#!/usr/bin/env python3
# coding=utf-8
"""Count target-token unigram statistics without running local-head training."""

import argparse
import glob
import hashlib
import os
import sys
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count SpecForge target-token unigram statistics."
    )
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--data-format",
        type=str,
        default=None,
        choices=["conversation", "preformatted", "raw_text"],
        help=(
            "Input dataset format. Defaults to conversation, or preformatted when "
            "--is-preformatted is set."
        ),
    )
    parser.add_argument("--chat-template", type=str, default="qwen")
    parser.add_argument("--is-preformatted", action="store_true")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--unigram-top-k",
        type=int,
        default=200,
        help="Number of nonzero top tokens to write to top_tokens.json.",
    )
    parser.add_argument(
        "--plot-unigram-prior",
        action="store_true",
        help="Also write unigram-prior plots if matplotlib is installed.",
    )
    parser.add_argument(
        "--unigram-plot-top-k",
        type=int,
        default=50,
        help="Number of nonzero top tokens to show in the optional bar chart.",
    )

    args = parser.parse_args()
    if args.data_format is None:
        args.data_format = "preformatted" if args.is_preformatted else "conversation"
    elif args.is_preformatted and args.data_format != "preformatted":
        parser.error("--is-preformatted conflicts with --data-format.")
    args.is_preformatted = args.data_format == "preformatted"

    if args.streaming and args.data_format != "raw_text":
        parser.error("--streaming is currently only supported with raw_text data.")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive when set.")
    return args


def _resolve_dataset_files(data_path: str) -> List[str]:
    if any(char in data_path for char in "*?[]"):
        files = sorted(glob.glob(data_path))
    elif os.path.isdir(data_path):
        files = sorted(
            os.path.join(data_path, name)
            for name in os.listdir(data_path)
            if os.path.isfile(os.path.join(data_path, name))
        )
    else:
        files = [data_path]

    files = [
        path for path in files if path.endswith((".json", ".jsonl", ".parquet"))
    ]
    if not files:
        raise ValueError(f"No supported dataset files found for {data_path!r}.")
    return files


def _infer_dataset_loader(data_path: str) -> Tuple[str, List[str]]:
    files = _resolve_dataset_files(data_path)
    suffixes = {Path(path).suffix for path in files}
    if suffixes <= {".json", ".jsonl"}:
        return "json", files
    if suffixes == {".parquet"}:
        return "parquet", files
    raise ValueError(
        f"Cannot mix dataset file formats for {data_path!r}: {sorted(suffixes)}"
    )


def _target_vocab_size(args, tokenizer) -> int:
    config = AutoConfig.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    vocab_size = int(getattr(config, "vocab_size", len(tokenizer)))
    tokenizer_size = len(tokenizer)
    if vocab_size < tokenizer_size:
        raise ValueError(
            f"Target config vocab_size ({vocab_size}) is smaller than tokenizer "
            f"size ({tokenizer_size})."
        )
    if vocab_size != tokenizer_size:
        print(
            f"Using target config vocab_size={vocab_size}; tokenizer size is "
            f"{tokenizer_size}."
        )
    return vocab_size


def main():
    args = parse_args()

    from specforge.data import build_eagle3_dataset
    from specforge.data.unigram import (
        count_loss_mask_unigrams,
        count_raw_text_unigrams,
        save_unigram_outputs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    vocab_size = _target_vocab_size(args, tokenizer)
    loader, files = _infer_dataset_loader(args.train_data_path)

    if args.streaming:
        dataset = load_dataset(loader, data_files=files, streaming=True)["train"]
        stats = count_raw_text_unigrams(
            dataset,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            max_length=args.max_length,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            text_column=args.text_column,
        )
    else:
        dataset = load_dataset(loader, data_files=files)["train"]
        if args.max_samples is not None:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        cache_params_string = (
            f"unigram-{args.train_data_path}-{args.max_length}-"
            f"{args.chat_template}-{args.data_format}-{args.target_model_path}-"
            f"{args.max_samples}"
        )
        cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
        dataset = build_eagle3_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            data_format=args.data_format,
            is_preformatted=args.is_preformatted,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            num_proc=args.build_dataset_num_proc,
        )
        stats = count_loss_mask_unigrams(
            dataset,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
        )

    metadata = {
        "target_model_path": args.target_model_path,
        "train_data_path": args.train_data_path,
        "data_format": args.data_format,
        "streaming": bool(args.streaming),
        "chat_template": args.chat_template,
        "is_preformatted": bool(args.is_preformatted),
        "text_column": args.text_column,
        "max_length": int(args.max_length),
        "max_samples": args.max_samples,
    }
    save_unigram_outputs(
        stats,
        tokenizer=tokenizer,
        output_dir=Path(args.output_dir),
        metadata=metadata,
        top_k=args.unigram_top_k,
        plot=args.plot_unigram_prior,
        plot_top_k=args.unigram_plot_top_k,
    )


if __name__ == "__main__":
    main()
