#!/usr/bin/env python3
# coding=utf-8
"""Train standalone RNN or Markov local heads for DeLS-Spec."""

import argparse
import glob
import logging
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from specforge.args import TrackerArgs
from specforge.core.local_head import (
    OnlineDirectVocabMarkovLocalHeadModel,
    OnlineLocalHeadModel,
    OnlineMarkovLocalHeadModel,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.data.unigram import count_loss_mask_unigrams, save_unigram_outputs
from specforge.utils import get_last_checkpoint, get_local_device, print_on_rank0


logger = logging.getLogger(__name__)


DEFAULT_LOSS_DECAY_GAMMA_BY_BLOCK_SIZE = {
    16: 7.0,
    10: 5.0,
    8: 4.0,
}


def resolve_loss_decay_gamma(
    loss_decay_gamma: Optional[float], block_size: int
) -> float:
    if loss_decay_gamma is not None:
        return loss_decay_gamma

    block_size = int(block_size)
    if block_size in DEFAULT_LOSS_DECAY_GAMMA_BY_BLOCK_SIZE:
        return DEFAULT_LOSS_DECAY_GAMMA_BY_BLOCK_SIZE[block_size]

    raise ValueError(
        "No default --loss-decay-gamma is defined for "
        f"block_size={block_size}. Pass --loss-decay-gamma explicitly; "
        "use 0 to disable."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train standalone DeLS-Spec local head"
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--local-head-type",
        type=str,
        default="rnn",
        choices=["rnn", "markov"],
        help="Train a GRU/RNN local head or a dense Markov local head.",
    )
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help="Number of anchor positions per sequence",
    )
    model_group.add_argument(
        "--local-gru-hidden-dim",
        type=int,
        default=1024,
        help="GRU hidden dimension for the local head.",
    )
    model_group.add_argument(
        "--local-rank",
        type=int,
        default=256,
        help="Low-rank dimension r after GRU hidden-state projection.",
    )
    model_group.add_argument(
        "--local-input-rank",
        type=int,
        default=None,
        help=(
            "Input rank for RNN local-head token embeddings. Defaults to "
            "--local-rank."
        ),
    )
    model_group.add_argument(
        "--local-head-parameterization",
        type=str,
        default="target_factorized",
        choices=["target_factorized", "direct_vocab"],
        help=(
            "Parameterization for the selected local head. target_factorized "
            "freezes target embedding/lm_head and trains low-rank adapters; "
            "direct_vocab trains vocab-space factors directly."
        ),
    )
    model_group.add_argument(
        "--rnn-parameterization",
        type=str,
        default=None,
        choices=["target_factorized", "direct_vocab"],
        help=(
            "Deprecated alias for --local-head-parameterization when "
            "--local-head-type rnn."
        ),
    )
    model_group.add_argument(
        "--local-lm-head-init",
        type=str,
        default="random",
        choices=["random", "zero"],
        help="Initialization for the trainable low-rank lm head.",
    )
    model_group.add_argument(
        "--local-rank-activation",
        type=str,
        default="identity",
        choices=["identity", "silu"],
        help=(
            "Activation after rank_proj and before the local lm head. "
            "Use 'silu' to explicitly enable SiLU; default is no activation."
        ),
    )
    model_group.add_argument(
        "--local-up-proj-init",
        type=str,
        default="random",
        choices=["random", "zero"],
        help=(
            "Initialization for the trainable r->target-hidden projection used "
            "with target_factorized RNN local heads."
        ),
    )
    model_group.add_argument(
        "--markov-rank",
        type=int,
        default=256,
        help="Rank r for Markov local-head low-rank factors.",
    )
    model_group.add_argument(
        "--markov-parameterization",
        type=str,
        default=None,
        choices=["target_factorized", "direct_vocab"],
        help=(
            "Deprecated alias for --local-head-parameterization when "
            "--local-head-type markov."
        ),
    )
    model_group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=None,
        help=(
            "Gamma for exponential loss decay from the first supervised target. "
            "Defaults by block size: 16 -> 7, 10 -> 5, 8 -> 4. "
            "Pass 0 to disable."
        ),
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default=None,
        help="Embedding weight key in the target model.",
    )
    model_group.add_argument(
        "--lm-head-key",
        type=str,
        default=None,
        help="LM head weight key in the target model.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument(
        "--data-format",
        type=str,
        default=None,
        choices=["conversation", "preformatted", "raw_text"],
        help=(
            "Input dataset format. Defaults to conversation, or preformatted when "
            "--is-preformatted is set. raw_text reads a text column and trains "
            "next-token prediction directly."
        ),
    )
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Stream raw-text datasets instead of materializing an Arrow cache. "
            "Only supported with --data-format raw_text."
        ),
    )
    dataset_group.add_argument(
        "--streaming-shuffle-buffer",
        type=int,
        default=10000,
        help="Shuffle buffer size for --streaming raw-text datasets.",
    )
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Required with --streaming; maximum dataloader steps to run.",
    )
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument(
        "--init-local-head-path",
        type=str,
        default=None,
        help=(
            "Warm-start local-head weights from a local_head.pt file or a "
            "checkpoint directory, without restoring optimizer, epoch, or step."
        ),
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=10000)
    output_group.add_argument(
        "--save-merged-rnn-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For RNN local heads, save merged_rnn_local_head.pt with fused "
            "vocab-space embedding/head weights. Enabled by default."
        ),
    )
    output_group.add_argument(
        "--save-merged-markov-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For Markov local heads, save merged_markov_local_head.pt with "
            "embedding_weight=vocab x r and lm_head_weight=vocab x r."
        ),
    )
    output_group.add_argument(
        "--save-unigram-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save loss-mask unigram statistics after training on rank 0.",
    )
    output_group.add_argument(
        "--unigram-output-dir",
        type=str,
        default=None,
        help="Directory for unigram stats. Defaults to output_dir/loss_mask_unigram.",
    )
    output_group.add_argument(
        "--unigram-top-k",
        type=int,
        default=200,
        help="Number of nonzero top tokens to write to top_tokens.json.",
    )
    output_group.add_argument(
        "--plot-unigram-prior",
        action="store_true",
        help="Also write unigram-prior plots if matplotlib is installed.",
    )
    output_group.add_argument(
        "--unigram-plot-top-k",
        type=int,
        default=50,
        help="Number of nonzero top tokens to show in the optional bar chart.",
    )

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--tp-size", type=int, default=1)

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    args = parser.parse_args()
    if args.resume and args.init_local_head_path:
        parser.error("--resume cannot be used with --init-local-head-path.")
    _resolve_data_format(parser, args)
    return _resolve_local_head_parameterization(parser, args)


def _resolve_data_format(parser, args) -> None:
    if args.data_format is None:
        args.data_format = "preformatted" if args.is_preformatted else "conversation"
        return

    if args.is_preformatted and args.data_format != "preformatted":
        parser.error("--is-preformatted conflicts with --data-format.")
    args.is_preformatted = args.data_format == "preformatted"

    if args.streaming:
        if args.data_format != "raw_text":
            parser.error("--streaming is currently only supported with raw_text data.")
        if args.max_train_steps is None or args.max_train_steps <= 0:
            parser.error("--streaming requires --max-train-steps > 0.")
        if args.resume:
            parser.error("--streaming does not currently support --resume.")


def _resolve_local_head_parameterization(parser, args):
    active_alias = (
        args.rnn_parameterization
        if args.local_head_type == "rnn"
        else args.markov_parameterization
    )
    if active_alias is not None:
        if (
            args.local_head_parameterization != "target_factorized"
            and args.local_head_parameterization != active_alias
        ):
            parser.error(
                "--local-head-parameterization conflicts with the deprecated "
                f"--{args.local_head_type}-parameterization alias."
            )
        args.local_head_parameterization = active_alias

    args.rnn_parameterization = args.local_head_parameterization
    args.markov_parameterization = args.local_head_parameterization
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


class StreamingRawTextDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: int,
        min_loss_tokens: int,
        rank: int,
        world_size: int,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_loss_tokens = min_loss_tokens
        self.rank = rank
        self.world_size = world_size

    def _tokenize(self, text: str):
        if not text:
            return None
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        if input_ids.numel() == 0:
            return None
        loss_mask = torch.ones_like(input_ids, dtype=torch.long)
        loss_mask[-1] = 0
        if int(loss_mask.sum().item()) < self.min_loss_tokens:
            return None
        return {
            "input_ids": input_ids[None, :],
            "loss_mask": loss_mask[None, :],
            "attention_mask": torch.ones_like(loss_mask)[None, :],
        }

    def __iter__(self):
        dataset = self.dataset
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        num_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        if hasattr(dataset, "shard"):
            dataset = dataset.shard(num_shards=num_shards, index=shard_index)
            for row in dataset:
                item = self._tokenize(row.get("text", ""))
                if item is not None:
                    yield item
        else:
            for idx, row in enumerate(dataset):
                if idx % num_shards != shard_index:
                    continue
                item = self._tokenize(row.get("text", ""))
                if item is not None:
                    yield item


def _build_streaming_raw_text_dataloader(args, tokenizer, load_dataset):
    from specforge.data.utils import DataCollatorWithPadding
    from specforge.distributed import get_dp_group

    train_loader, train_files = _infer_dataset_loader(args.train_data_path)
    dataset = load_dataset(train_loader, data_files=train_files, streaming=True)["train"]
    if args.streaming_shuffle_buffer > 0:
        dataset = dataset.shuffle(
            buffer_size=args.streaming_shuffle_buffer, seed=args.seed
        )

    process_group = get_dp_group()
    min_loss_tokens = 2 * args.block_size if args.local_head_type == "rnn" else 1
    iterable = StreamingRawTextDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        min_loss_tokens=min_loss_tokens,
        rank=dist.get_rank(process_group),
        world_size=dist.get_world_size(process_group),
    )

    if args.dataloader_num_workers != 0:
        print_on_rank0(
            "Streaming raw_text uses num_workers=0 to avoid partial iteration "
            "from DataLoader worker sharding."
        )
    return DataLoader(
        iterable,
        batch_size=args.batch_size,
        num_workers=0,
        prefetch_factor=None,
        collate_fn=DataCollatorWithPadding(),
        drop_last=True,
    )


def build_dataloader(
    args, tokenizer
) -> Tuple[DataLoader, Optional[DataLoader], object]:
    import hashlib

    from datasets import load_dataset

    from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
    from specforge.distributed import get_dp_group

    if args.streaming:
        train_dataloader = _build_streaming_raw_text_dataloader(
            args, tokenizer, load_dataset
        )
        return train_dataloader, None, None

    cache_params_string = (
        f"{args.local_head_type}-local-head-{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.data_format}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_loader, train_files = _infer_dataset_loader(args.train_data_path)
    train_dataset = load_dataset(train_loader, data_files=train_files)["train"]
    train_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        data_format=args.data_format,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )
    unigram_dataset = train_dataset

    min_loss_tokens = 2 * args.block_size if args.local_head_type == "rnn" else 1
    original_size = len(train_dataset)
    train_dataset = train_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_dataset)} samples"
    )

    train_dataloader = prepare_dp_dataloaders(
        train_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_loader, eval_files = _infer_dataset_loader(args.eval_data_path)
        eval_dataset = load_dataset(eval_loader, data_files=eval_files)["train"]
        eval_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            data_format=args.data_format,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_dataloader, eval_dataloader, unigram_dataset


def _is_frozen_target_key(name: str, model=None) -> bool:
    if name.startswith("target_lm_head."):
        return True
    if not name.startswith("embed_tokens."):
        return False

    if getattr(model, "rnn_parameterization", None) == "direct_vocab":
        return False
    return True


def _resolve_local_head_checkpoint_path(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "local_head.pt")
    return path


def _load_local_head_checkpoint(
    local_head_model,
    checkpoint_path: str,
    local_head_type: str,
) -> dict:
    checkpoint_path = _resolve_local_head_checkpoint_path(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Local-head checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_type = checkpoint.get("model_config", {}).get("local_head_type")
    if checkpoint_type is not None and checkpoint_type != local_head_type:
        raise RuntimeError(
            f"Checkpoint local_head_type mismatch: "
            f"checkpoint={checkpoint_type}, current={local_head_type}"
        )
    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"{checkpoint_path} does not contain model_state_dict. "
            "Use a training local_head.pt checkpoint, not a merged inference artifact."
        )

    missing, unexpected = local_head_model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    missing = [
        name for name in missing if not _is_frozen_target_key(name, local_head_model)
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected local-head checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    checkpoint["_checkpoint_path"] = checkpoint_path
    return checkpoint


def _local_head_config(args, model) -> dict:
    common = {
        "local_head_type": args.local_head_type,
        "vocab_size": model.vocab_size,
        "target_model_path": args.target_model_path,
        "embedding_key": args.embedding_key,
        "lm_head_key": args.lm_head_key,
        "local_head_parameterization": args.local_head_parameterization,
    }
    if args.local_head_type == "markov":
        return {
            **common,
            "target_hidden_size": model.target_hidden_size,
            "rank": model.rank,
            "architecture": model.architecture,
            "markov_parameterization": args.markov_parameterization,
        }

    return {
        **common,
        "embed_dim": model.embed_dim,
        "input_rank_dim": model.input_rank_dim,
        "gru_input_dim": model.gru_input_dim,
        "gru_hidden_dim": model.gru_hidden_dim,
        "low_rank_dim": model.low_rank_dim,
        "block_size": model.block_size,
        "num_anchors": model.num_anchors,
        "loss_decay_gamma": model.loss_decay_gamma,
        "lm_head_mode": model.lm_head_mode,
        "rnn_parameterization": model.rnn_parameterization,
        "target_hidden_size": model.target_hidden_size,
        "rank_activation": model.rank_activation,
    }


def _rnn_lm_head_weight_from_state(model: OnlineLocalHeadModel, state_dict: dict):
    if model.lm_head_mode == "low_rank":
        return state_dict["low_rank_lm_head.weight"]

    target_weight = state_dict["target_lm_head.weight"].to(torch.float32)
    up_weight = state_dict["rank_to_hidden.weight"].to(torch.float32)
    return target_weight.matmul(up_weight).to(up_weight.dtype)


def _merged_rnn_state(args, model: OnlineLocalHeadModel, state_dict: dict) -> dict:
    if model.rnn_parameterization == "direct_vocab":
        embedding_weight = state_dict["embed_tokens.weight"]
        lm_head_weight = state_dict["low_rank_lm_head.weight"]
    else:
        embed_weight = state_dict["embed_tokens.weight"].to(torch.float32)
        down_weight = state_dict["embed_down_proj.weight"].to(torch.float32)
        embedding_weight = embed_weight.matmul(down_weight.t()).to(down_weight.dtype)
        lm_head_weight = _rnn_lm_head_weight_from_state(model, state_dict)

    model_state_dict = {
        k: v
        for k, v in state_dict.items()
        if k.startswith("prefix_gru.") or k.startswith("rank_proj.")
    }
    model_state_dict["low_rank_lm_head.weight"] = lm_head_weight
    model_state_dict["embed_tokens.weight"] = embedding_weight

    model_config = {
        **_local_head_config(args, model),
        "source_rnn_parameterization": model.rnn_parameterization,
        "source_lm_head_mode": model.lm_head_mode,
        "rnn_parameterization": "direct_vocab",
        "lm_head_mode": "low_rank",
        "architecture": "direct_vocab_rnn",
        "embed_dim": model.gru_input_dim,
        "target_hidden_size": None,
    }

    return {
        "embedding_weight": embedding_weight,
        "lm_head_weight": lm_head_weight,
        "vocab_size": model.vocab_size,
        "input_rank_dim": model.input_rank_dim,
        "low_rank_dim": model.low_rank_dim,
        "model_config": model_config,
        "model_state_dict": model_state_dict,
    }


def _merged_markov_state(args, model, state_dict: dict) -> dict:
    if model.architecture == "direct_vocab_low_rank":
        embedding_weight = state_dict["embedding_weight"]
        lm_head_weight = state_dict["lm_head_weight"]
    else:
        embed_weight = state_dict["embed_tokens.weight"].to(torch.float32)
        lm_head_weight_full = state_dict["target_lm_head.weight"].to(torch.float32)
        down_weight = state_dict["down_proj"].to(torch.float32)
        up_weight = state_dict["up_proj"].to(torch.float32)
        embedding_weight = embed_weight.matmul(down_weight).to(down_weight.dtype)
        lm_head_weight = lm_head_weight_full.matmul(up_weight.t()).to(up_weight.dtype)

    return {
        "embedding_weight": embedding_weight,
        "lm_head_weight": lm_head_weight,
        "rank": model.rank,
        "vocab_size": model.vocab_size,
        "model_config": _local_head_config(args, model),
    }


def save_checkpoint(args, epoch, step, fsdp_model, local_head_model, optimizer):
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
        state_dict = fsdp_model.state_dict()
        local_state_dict = {
            k: v
            for k, v in state_dict.items()
            if not _is_frozen_target_key(k, local_head_model)
        }

        if dist.get_rank() == 0:
            checkpoint_payload = {
                "epoch": epoch,
                "global_step": step,
                "args": args,
                "model_config": _local_head_config(args, local_head_model),
                "model_state_dict": local_state_dict,
                **optimizer.state_dict(),
            }
            torch.save(
                checkpoint_payload,
                os.path.join(save_dir, "local_head.pt"),
            )
            if args.local_head_type == "rnn" and args.save_merged_rnn_weights:
                torch.save(
                    _merged_rnn_state(args, local_head_model, state_dict),
                    os.path.join(save_dir, "merged_rnn_local_head.pt"),
                )
            if (
                args.local_head_type == "markov"
                and args.save_merged_markov_weights
            ):
                torch.save(
                    _merged_markov_state(args, local_head_model, state_dict),
                    os.path.join(save_dir, "merged_markov_local_head.pt"),
                )
            print_on_rank0(f"Saved checkpoint to {save_dir}")

    dist.barrier()


def reduce_metrics_dict(metrics):
    if not metrics:
        return {}

    world_size = dist.get_world_size()
    reduced = {}
    for k, v in metrics.items():
        if v is None:
            continue
        if torch.is_tensor(v):
            t = v.detach().clone()
            dist.all_reduce(t)
            reduced[k] = (t / world_size).item()
        else:
            reduced[k] = float(v)
    return reduced


def record_metrics(
    args,
    loss: float,
    accuracy: float,
    global_step: int,
    tracker,
    optimizer,
    train_dataloader=None,
    mode: str = "train",
    extra_metrics: dict = None,
    progress_bar=None,
) -> None:
    logdict = {}
    if mode == "train" and optimizer is not None:
        logdict[f"{mode}/lr"] = optimizer.get_learning_rate()
    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy
    if extra_metrics:
        logdict.update(
            {f"{mode}/{k}": float(v) for k, v in extra_metrics.items() if v is not None}
        )

    total_log_steps = (
        args.max_train_steps
        if getattr(args, "streaming", False)
        else args.num_epochs * len(train_dataloader) // args.accumulation_steps
    )
    print_msg = (
        f"{mode.capitalize()} - Step {global_step} "
        f"[{global_step}/{total_log_steps}?], "
        f"Loss: {loss:.4f}, Acc: {accuracy:.4f}"
    )
    if dist.get_rank() == 0 and progress_bar is not None and hasattr(
        progress_bar, "write"
    ):
        progress_bar.write(print_msg)
    else:
        print_on_rank0(print_msg)
    tracker.log(logdict, step=global_step)


def save_unigram_prior(args, tokenizer, dataset) -> None:
    output_dir = Path(
        args.unigram_output_dir or os.path.join(args.output_dir, "loss_mask_unigram")
    )
    stats = count_loss_mask_unigrams(
        dataset,
        tokenizer=tokenizer,
        vocab_size=_target_vocab_size(args, tokenizer),
    )
    metadata = {
        "target_model_path": args.target_model_path,
        "train_data_path": args.train_data_path,
        "chat_template": args.chat_template,
        "is_preformatted": bool(args.is_preformatted),
        "data_format": args.data_format,
        "max_length": int(args.max_length),
    }
    save_unigram_outputs(
        stats,
        tokenizer=tokenizer,
        output_dir=output_dir,
        metadata=metadata,
        top_k=args.unigram_top_k,
        plot=args.plot_unigram_prior,
        plot_top_k=args.unigram_plot_top_k,
        log_fn=print_on_rank0,
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
        message = (
            f"Using target config vocab_size={vocab_size}; tokenizer size is "
            f"{tokenizer_size}."
        )
        if dist.is_available() and dist.is_initialized():
            print_on_rank0(message)
        else:
            logger.info(message)
    return vocab_size


def build_local_head_model(args, target_components, tokenizer, device):
    if (
        args.local_head_type == "markov"
        and args.markov_parameterization == "direct_vocab"
    ):
        vocab_size = _target_vocab_size(args, tokenizer)
        return OnlineDirectVocabMarkovLocalHeadModel(
            vocab_size=vocab_size,
            rank=args.markov_rank,
        ).to(device=device, dtype=torch.bfloat16)

    if args.local_head_type == "rnn":
        local_lm_head_mode = (
            "target_lm_head"
            if args.rnn_parameterization == "target_factorized"
            else "low_rank"
        )
        input_rank_dim = (
            args.local_input_rank
            if args.local_input_rank is not None
            else args.local_rank
        )
        if args.rnn_parameterization == "direct_vocab":
            vocab_size = _target_vocab_size(args, tokenizer)
            return OnlineLocalHeadModel(
                target_embed_tokens=None,
                target_lm_head=None,
                vocab_size=vocab_size,
                embed_dim=input_rank_dim,
                input_rank_dim=input_rank_dim,
                gru_hidden_dim=args.local_gru_hidden_dim,
                low_rank_dim=args.local_rank,
                block_size=args.block_size,
                num_anchors=args.num_anchors,
                loss_decay_gamma=args.loss_decay_gamma,
                lm_head_mode=local_lm_head_mode,
                lm_head_init=args.local_lm_head_init,
                rnn_parameterization=args.rnn_parameterization,
                rank_activation=args.local_rank_activation,
                up_proj_init=args.local_up_proj_init,
            ).to(device=device, dtype=torch.bfloat16)

    embed_tokens = target_components.embed_tokens
    target_lm_head = target_components.lm_head
    vocab_size, target_hidden_size = target_lm_head.weight.shape
    embed_dim = embed_tokens.weight.shape[1]
    if embed_dim != target_hidden_size:
        raise ValueError(
            f"Embedding dim ({embed_dim}) must match target lm_head hidden size "
            f"({target_hidden_size}) for the local head."
        )

    if args.local_head_type == "markov":
        return OnlineMarkovLocalHeadModel(
            target_embed_tokens=embed_tokens,
            target_lm_head=target_lm_head,
            rank=args.markov_rank,
        ).to(device=device, dtype=torch.bfloat16)

    local_head_model = OnlineLocalHeadModel(
        target_embed_tokens=embed_tokens,
        target_lm_head=target_lm_head
        if local_lm_head_mode == "target_lm_head"
        else None,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        input_rank_dim=input_rank_dim,
        gru_hidden_dim=args.local_gru_hidden_dim,
        low_rank_dim=args.local_rank,
        block_size=args.block_size,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        lm_head_mode=local_lm_head_mode,
        lm_head_init=args.local_lm_head_init,
        rnn_parameterization=args.rnn_parameterization,
        rank_activation=args.local_rank_activation,
        up_proj_init=args.local_up_proj_init,
    )
    return local_head_model.to(device=device, dtype=torch.bfloat16)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()

    from specforge.distributed import destroy_distributed, init_distributed
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    device = get_local_device()
    set_seed(args.seed)

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    if args.local_head_type == "rnn":
        args.loss_decay_gamma = resolve_loss_decay_gamma(
            args.loss_decay_gamma, args.block_size
        )
        print_on_rank0(f"Using loss_decay_gamma: {args.loss_decay_gamma}")

    local_head_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        local_head_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {local_head_last_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    train_dataloader, eval_dataloader, unigram_dataset = build_dataloader(
        args, tokenizer
    )
    del eval_dataloader

    if args.streaming:
        steps_per_epoch = args.max_train_steps
        total_steps = math.ceil(args.max_train_steps / args.accumulation_steps)
    else:
        steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
        total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    target_components = None
    if not (
        (
            args.local_head_type == "markov"
            and args.markov_parameterization == "direct_vocab"
        )
        or (
            args.local_head_type == "rnn"
            and args.rnn_parameterization == "direct_vocab"
        )
    ):
        print_on_rank0("Loading target embeddings and head...")
        target_components = TargetEmbeddingsAndHead.from_pretrained(
            args.target_model_path,
            embed_key=args.embedding_key,
            lm_head_key=args.lm_head_key,
            device=device.type,
            trust_remote_code=args.trust_remote_code,
        )

    local_head_model = build_local_head_model(
        args, target_components, tokenizer, device
    )
    if args.local_head_type == "markov":
        print_on_rank0(
            "Markov local head config: "
            f"parameterization={args.markov_parameterization}, "
            f"rank={args.markov_rank}, "
            f"hidden_size={local_head_model.target_hidden_size}, "
            f"vocab_size={local_head_model.vocab_size}"
        )
    else:
        print_on_rank0(
            "RNN local head config: "
            f"parameterization={args.rnn_parameterization}, "
            f"block_size={args.block_size}, "
            f"input_rank={local_head_model.input_rank_dim}, "
            f"gru_hidden_dim={args.local_gru_hidden_dim}, "
            f"output_rank={args.local_rank}, "
            f"lm_head_mode={local_head_model.lm_head_mode}, "
            f"rank_activation={args.local_rank_activation}"
        )
    print_on_rank0(
        "Trainable local head parameters: "
        f"{sum(p.numel() for p in local_head_model.parameters() if p.requires_grad):,}"
    )

    if args.init_local_head_path:
        init_state = _load_local_head_checkpoint(
            local_head_model,
            args.init_local_head_path,
            args.local_head_type,
        )
        print_on_rank0(
            "Initialized local head weights from checkpoint: "
            f"{init_state['_checkpoint_path']}"
        )
        del init_state

    resume_state = None
    if local_head_last_checkpoint:
        resume_state = _load_local_head_checkpoint(
            local_head_model,
            local_head_last_checkpoint,
            args.local_head_type,
        )
        print_on_rank0(
            f"Loaded local head checkpoint: {resume_state['_checkpoint_path']}"
        )

    fsdp_model = FSDP(
        local_head_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )

    start_epoch = ckpt_info[0]
    global_step = ckpt_info[1]
    optimizer = BF16Optimizer(
        local_head_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    if resume_state is not None:
        optimizer.load_state_dict(resume_state)
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(
            f"Restored optimizer/scheduler state: epoch={start_epoch}, "
            f"step={global_step}, lr={optimizer.get_learning_rate():.6f}"
        )

    skip_steps = (
        0 if args.streaming else global_step - start_epoch * len(train_dataloader)
    )

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        if not args.streaming:
            train_dataloader.sampler.set_epoch(epoch)
        local_head_model.train()
        if hasattr(local_head_model, "embed_tokens"):
            local_head_model.embed_tokens.eval()
        if getattr(local_head_model, "target_lm_head", None) is not None:
            local_head_model.target_lm_head.eval()

        if dist.get_rank() == 0:
            tqdm_kwargs = {}
            if args.streaming:
                tqdm_kwargs = {
                    "total": args.max_train_steps,
                    "initial": global_step,
                }
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Training Epoch {epoch}",
                leave=True,
                **tqdm_kwargs,
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            if args.streaming and global_step >= args.max_train_steps:
                break
            global_step += 1

            input_ids = data["input_ids"].to(device, non_blocking=True)
            loss_mask = data["loss_mask"].to(device, non_blocking=True)

            loss, accuracy, metrics = fsdp_model(
                input_ids=input_ids,
                loss_mask=loss_mask,
            )

            (loss / args.accumulation_steps).backward()

            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                acc_log = accuracy.clone()
                dist.all_reduce(loss_log)
                dist.all_reduce(acc_log)
                loss_log = loss_log / dist.get_world_size()
                acc_log = acc_log / dist.get_world_size()
                metrics = reduce_metrics_dict(metrics)

                record_metrics(
                    args,
                    loss_log.item(),
                    acc_log.item(),
                    global_step,
                    tracker,
                    optimizer,
                    train_dataloader,
                    mode="train",
                    extra_metrics=metrics,
                    progress_bar=progress_bar,
                )

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args, epoch, global_step, fsdp_model, local_head_model, optimizer
                )
            if args.streaming and global_step >= args.max_train_steps:
                break
        if args.streaming and global_step >= args.max_train_steps:
            break

    save_checkpoint(
        args, args.num_epochs, global_step, fsdp_model, local_head_model, optimizer
    )

    tracker.close()
    if args.save_unigram_prior:
        if args.streaming:
            print_on_rank0("Skipping unigram prior save for streaming datasets.")
            destroy_distributed()
            return
        dist.barrier()
        if dist.get_rank() == 0:
            save_unigram_prior(args, tokenizer, unigram_dataset)
        dist.barrier()
    destroy_distributed()


if __name__ == "__main__":
    main()
