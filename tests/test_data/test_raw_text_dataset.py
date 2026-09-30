import os
from types import SimpleNamespace

import torch
import pytest

from datasets import Dataset
from scripts.train_local_head import (
    StreamingRawTextDataset,
    _build_streaming_raw_text_dataloader,
    _infer_dataset_loader,
    _resolve_data_format,
)
from specforge.data.preprocessing import build_eagle3_dataset
from specforge.data.unigram import count_raw_text_unigrams


class DummyTokenizer:
    def __call__(
        self,
        text,
        max_length=None,
        truncation=False,
        return_tensors=None,
        add_special_tokens=False,
    ):
        def tokenize_one(value):
            ids = [idx + 1 for idx, _ in enumerate(value.split())]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            return ids

        if isinstance(text, list):
            return {"input_ids": [tokenize_one(value) for value in text]}
        return SimpleNamespace(
            input_ids=torch.tensor([tokenize_one(text)], dtype=torch.long)
        )


def test_build_eagle3_dataset_raw_text_uses_text_column():
    dataset = Dataset.from_dict({"text": ["alpha beta gamma delta"]})

    processed = build_eagle3_dataset(
        dataset=dataset,
        tokenizer=DummyTokenizer(),
        chat_template=None,
        max_length=3,
        data_format="raw_text",
        num_proc=1,
        cache_dir=None,
        cache_key=None,
    )

    assert processed.column_names == ["input_ids", "loss_mask", "attention_mask"]
    assert processed[0]["input_ids"].tolist() == [[1, 2, 3]]
    assert processed[0]["attention_mask"].tolist() == [[1, 1, 1]]
    assert processed[0]["loss_mask"].tolist() == [[1, 1, 0]]


def test_infer_dataset_loader_accepts_parquet_directory(tmp_path):
    (tmp_path / "000.parquet").write_text("placeholder")
    (tmp_path / "001.parquet").write_text("placeholder")

    loader, files = _infer_dataset_loader(str(tmp_path))

    assert loader == "parquet"
    assert files == [
        os.path.join(str(tmp_path), "000.parquet"),
        os.path.join(str(tmp_path), "001.parquet"),
    ]


def test_streaming_raw_text_dataset_tokenizes_and_filters_short_rows():
    rows = [
        {"text": "too short"},
        {"text": "alpha beta gamma delta"},
    ]
    dataset = StreamingRawTextDataset(
        dataset=rows,
        tokenizer=DummyTokenizer(),
        max_length=4,
        min_loss_tokens=3,
        rank=0,
        world_size=1,
    )

    items = list(dataset)

    assert len(items) == 1
    assert items[0]["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert items[0]["loss_mask"].tolist() == [[1, 1, 1, 0]]


def test_streaming_requires_raw_text_and_max_train_steps():
    args = SimpleNamespace(
        data_format="conversation",
        is_preformatted=False,
        streaming=True,
        max_train_steps=10,
        resume=False,
    )
    parser = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(ValueError(message)))

    with pytest.raises(ValueError, match="raw_text"):
        _resolve_data_format(parser, args)

    args.data_format = "raw_text"
    args.max_train_steps = None
    with pytest.raises(ValueError, match="max-train-steps"):
        _resolve_data_format(parser, args)


def test_streaming_raw_text_dataloader_forces_single_worker(monkeypatch, tmp_path):
    data_file = tmp_path / "data.jsonl"
    data_file.write_text('{"text": "alpha beta gamma delta"}\n')
    args = SimpleNamespace(
        train_data_path=str(data_file),
        streaming_shuffle_buffer=0,
        seed=0,
        block_size=2,
        local_head_type="rnn",
        max_length=8,
        batch_size=1,
        dataloader_num_workers=8,
    )

    class DistStub:
        @staticmethod
        def get_rank(_group=None):
            return 0

        @staticmethod
        def get_world_size(_group=None):
            return 1

    import scripts.train_local_head as train_local_head
    import specforge.data.utils as data_utils

    monkeypatch.setattr(train_local_head, "dist", DistStub)
    monkeypatch.setattr(train_local_head, "print_on_rank0", lambda message: None)
    monkeypatch.setattr(data_utils, "DataCollatorWithPadding", lambda: lambda batch: batch)
    dataloader = _build_streaming_raw_text_dataloader(
        args,
        DummyTokenizer(),
        lambda loader, data_files, streaming: {
            "train": [{"text": "alpha beta gamma delta"}]
        },
    )

    assert dataloader.num_workers == 0


def test_count_raw_text_unigrams_counts_loss_mask_positions_with_max_samples():
    rows = [
        {"text": "alpha beta gamma"},
        {"text": "delta epsilon"},
        {"text": "zeta"},
    ]

    stats = count_raw_text_unigrams(
        rows,
        tokenizer=DummyTokenizer(),
        vocab_size=5,
        max_length=8,
        batch_size=10,
        max_samples=1,
    )

    assert stats["counts"].tolist() == [0, 1, 1, 0, 0]
    assert stats["total_target_tokens"].item() == 2
    assert stats["num_samples"].item() == 1
