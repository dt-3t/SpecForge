#!/usr/bin/env python3
# coding=utf-8
"""Train standalone RNN or Markov local heads for DeLS-Spec."""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from specforge.args import TrackerArgs
from specforge.core.local_head import (
    OnlineDirectVocabMarkovLocalHeadModel,
    OnlineLocalHeadModel,
    OnlineMarkovLocalHeadModel,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import get_last_checkpoint, get_local_device, print_on_rank0


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
        "--local-lm-head-mode",
        type=str,
        default="low_rank",
        choices=["low_rank", "target_lm_head"],
        help=(
            "Use a trainable low-rank vocab head, or use a frozen target lm_head "
            "after a trainable r->target-hidden projection."
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
            "with --local-lm-head-mode target_lm_head."
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
        default="target_factorized",
        choices=["target_factorized", "direct_vocab"],
        help=(
            "target_factorized freezes target embedding/lm_head and trains "
            "hidden-size down/up projections. direct_vocab trains two vocab x rank "
            "matrices directly and does not load target embedding/lm_head."
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
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
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
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=10000)
    output_group.add_argument(
        "--save-merged-lm-head",
        action="store_true",
        help=(
            "Also save merged_lm_head.pt as vocab x r weights. In "
            "target_lm_head mode this is target_lm_head.weight @ up_proj.weight."
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

    return parser.parse_args()


def build_dataloader(
    args, tokenizer
) -> Tuple[DataLoader, Optional[DataLoader], object]:
    import hashlib

    from datasets import load_dataset

    from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
    from specforge.distributed import get_dp_group

    cache_params_string = (
        f"{args.local_head_type}-local-head-{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
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
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
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


def _is_frozen_target_key(name: str) -> bool:
    return name.startswith("embed_tokens.") or name.startswith("target_lm_head.")


def _local_head_config(args, model) -> dict:
    common = {
        "local_head_type": args.local_head_type,
        "vocab_size": model.vocab_size,
        "target_model_path": args.target_model_path,
        "embedding_key": args.embedding_key,
        "lm_head_key": args.lm_head_key,
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
        "gru_hidden_dim": model.gru_hidden_dim,
        "low_rank_dim": model.low_rank_dim,
        "block_size": model.block_size,
        "num_anchors": model.num_anchors,
        "loss_decay_gamma": model.loss_decay_gamma,
        "lm_head_mode": model.lm_head_mode,
        "rank_activation": model.rank_activation,
    }


def _merged_lm_head_from_state(model: OnlineLocalHeadModel, state_dict: dict):
    if model.lm_head_mode == "low_rank":
        return state_dict["low_rank_lm_head.weight"]

    target_weight = state_dict["target_lm_head.weight"].to(torch.float32)
    up_weight = state_dict["rank_to_hidden.weight"].to(torch.float32)
    return target_weight.matmul(up_weight).to(up_weight.dtype)


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
            k: v for k, v in state_dict.items() if not _is_frozen_target_key(k)
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
            if args.local_head_type == "rnn" and args.save_merged_lm_head:
                torch.save(
                    {
                        "weight": _merged_lm_head_from_state(
                            local_head_model, state_dict
                        )
                    },
                    os.path.join(save_dir, "merged_lm_head.pt"),
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

    print_msg = (
        f"{mode.capitalize()} - Step {global_step} "
        f"[{global_step}/"
        f"{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?], "
        f"Loss: {loss:.4f}, Acc: {accuracy:.4f}"
    )
    print_on_rank0(print_msg)
    tracker.log(logdict, step=global_step)


def _ensure_count_capacity(counts: torch.Tensor, max_token_id: int) -> torch.Tensor:
    if max_token_id < counts.numel():
        return counts
    expanded = torch.zeros(max_token_id + 1, dtype=counts.dtype)
    expanded[: counts.numel()] = counts
    return expanded


def count_loss_mask_unigrams(dataset, tokenizer) -> Dict[str, torch.Tensor]:
    vocab_size = len(tokenizer)
    counts = torch.zeros(vocab_size, dtype=torch.long)
    total_target_tokens = 0

    for idx in tqdm(range(len(dataset)), desc="Counting loss_mask target tokens"):
        row = dataset[idx]
        input_ids = torch.as_tensor(row["input_ids"]).view(-1).long()
        loss_mask = torch.as_tensor(row["loss_mask"]).view(-1) > 0
        if input_ids.numel() != loss_mask.numel():
            raise ValueError(
                f"Sample {idx} has mismatched input_ids/loss_mask lengths: "
                f"{input_ids.numel()} vs {loss_mask.numel()}"
            )

        target_ids = input_ids[loss_mask]
        if target_ids.numel() == 0:
            continue

        max_token_id = int(target_ids.max().item())
        counts = _ensure_count_capacity(counts, max_token_id)
        counts += torch.bincount(target_ids, minlength=counts.numel())
        total_target_tokens += int(target_ids.numel())

    if total_target_tokens == 0:
        raise ValueError("No tokens with loss_mask == 1 were found.")

    p = counts.to(torch.float64) / float(total_target_tokens)
    log_p = torch.full_like(p, -math.inf)
    nonzero = counts > 0
    log_p[nonzero] = torch.log(p[nonzero])

    return {
        "counts": counts,
        "p": p,
        "log_p": log_p,
        "total_target_tokens": torch.tensor(total_target_tokens, dtype=torch.long),
        "num_samples": torch.tensor(len(dataset), dtype=torch.long),
    }


def _token_strings(tokenizer, token_id: int) -> Dict[str, str]:
    token = tokenizer.convert_ids_to_tokens(token_id)
    text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    return {"token": token, "text": text}


def write_top_tokens(
    stats: Dict[str, torch.Tensor],
    tokenizer,
    output_path: Path,
    top_k: int,
) -> List[Dict]:
    counts = stats["counts"]
    p = stats["p"]
    log_p = stats["log_p"]
    positive_ids = torch.where(counts > 0)[0]
    rows = []
    if positive_ids.numel() > 0:
        k = min(top_k, int(positive_ids.numel()))
        top_counts, top_indices = torch.topk(counts[positive_ids], k=k)
        top_ids = positive_ids[top_indices]
        for token_id, count in zip(top_ids.tolist(), top_counts.tolist()):
            strings = _token_strings(tokenizer, token_id)
            rows.append(
                {
                    "token_id": token_id,
                    "count": int(count),
                    "p": float(p[token_id].item()),
                    "log_p": float(log_p[token_id].item()),
                    **strings,
                }
            )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return rows


def plot_unigram_distribution(
    stats: Dict[str, torch.Tensor],
    top_rows: List[Dict],
    output_dir: Path,
    plot_top_k: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print_on_rank0("matplotlib is not installed; skipping unigram plots.")
        return

    def safe_label(value) -> str:
        label = repr("" if value is None else value)
        label = label.replace("$", r"\$")
        if len(label) > 30:
            label = label[:27] + "..."
        return label

    rows = top_rows[:plot_top_k]
    if rows:
        labels = []
        values = []
        for row in rows:
            token_label = row["token"]
            if token_label is None:
                token_label = row["text"]
            labels.append(f"{row['token_id']}: {safe_label(token_label)}")
            values.append(row["p"])

        height = max(6.0, 0.25 * len(rows))
        fig, ax = plt.subplots(figsize=(12, height))
        ax.barh(range(len(rows)), values)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("p(token | loss_mask=1)")
        ax.set_title(f"Top {len(rows)} Loss-Mask Target Token Unigrams")
        fig.tight_layout()
        fig.savefig(output_dir / "loss_mask_unigram_top_tokens.png", dpi=200)
        plt.close(fig)

    positive_p = stats["p"][stats["counts"] > 0].cpu().numpy()
    positive_p = np.sort(positive_p)[::-1]
    ranks = np.arange(1, len(positive_p) + 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(ranks, positive_p)
    ax.set_xlabel("Rank")
    ax.set_ylabel("p(token | loss_mask=1)")
    ax.set_title("Loss-Mask Target Token Unigram Zipf Plot")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_mask_unigram_zipf.png", dpi=200)
    plt.close(fig)


def save_unigram_prior(args, tokenizer, dataset) -> None:
    output_dir = Path(
        args.unigram_output_dir or os.path.join(args.output_dir, "loss_mask_unigram")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = count_loss_mask_unigrams(dataset, tokenizer=tokenizer)
    metadata = {
        "target_model_path": args.target_model_path,
        "train_data_path": args.train_data_path,
        "chat_template": args.chat_template,
        "is_preformatted": bool(args.is_preformatted),
        "max_length": int(args.max_length),
        "vocab_size": int(stats["counts"].numel()),
        "num_samples": int(stats["num_samples"].item()),
        "total_target_tokens": int(stats["total_target_tokens"].item()),
    }

    torch.save({**stats, "metadata": metadata}, output_dir / "loss_mask_unigram.pt")
    np.savez_compressed(
        output_dir / "loss_mask_unigram.npz",
        counts=stats["counts"].cpu().numpy(),
        p=stats["p"].cpu().numpy(),
        log_p=stats["log_p"].cpu().numpy(),
    )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    top_rows = write_top_tokens(
        stats,
        tokenizer=tokenizer,
        output_path=output_dir / "top_tokens.json",
        top_k=args.unigram_top_k,
    )
    if args.plot_unigram_prior:
        plot_unigram_distribution(
            stats,
            top_rows,
            output_dir,
            plot_top_k=args.unigram_plot_top_k,
        )

    print_on_rank0(
        "Saved loss-mask target-token unigram stats to "
        f"{output_dir} with {metadata['total_target_tokens']} target tokens "
        f"from {metadata['num_samples']} samples."
    )


def build_local_head_model(args, target_components, tokenizer, device):
    if (
        args.local_head_type == "markov"
        and args.markov_parameterization == "direct_vocab"
    ):
        return OnlineDirectVocabMarkovLocalHeadModel(
            vocab_size=len(tokenizer),
            rank=args.markov_rank,
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
        if args.local_lm_head_mode == "target_lm_head"
        else None,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        gru_hidden_dim=args.local_gru_hidden_dim,
        low_rank_dim=args.local_rank,
        block_size=args.block_size,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        lm_head_mode=args.local_lm_head_mode,
        lm_head_init=args.local_lm_head_init,
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

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    target_components = None
    if not (
        args.local_head_type == "markov"
        and args.markov_parameterization == "direct_vocab"
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
            f"block_size={args.block_size}, "
            f"gru_hidden_dim={args.local_gru_hidden_dim}, rank={args.local_rank}, "
            f"lm_head_mode={args.local_lm_head_mode}, "
            f"rank_activation={args.local_rank_activation}"
        )
    print_on_rank0(
        "Trainable local head parameters: "
        f"{sum(p.numel() for p in local_head_model.parameters() if p.requires_grad):,}"
    )

    resume_state = None
    if local_head_last_checkpoint:
        checkpoint_path = os.path.join(local_head_last_checkpoint, "local_head.pt")
        resume_state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_type = resume_state.get("model_config", {}).get("local_head_type")
        if checkpoint_type is not None and checkpoint_type != args.local_head_type:
            raise RuntimeError(
                f"Checkpoint local_head_type mismatch: "
                f"checkpoint={checkpoint_type}, current={args.local_head_type}"
            )
        missing, unexpected = local_head_model.load_state_dict(
            resume_state["model_state_dict"], strict=False
        )
        missing = [name for name in missing if not _is_frozen_target_key(name)]
        if missing or unexpected:
            raise RuntimeError(
                f"Unexpected local-head checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        print_on_rank0(f"Loaded local head checkpoint: {checkpoint_path}")

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

    skip_steps = global_step - start_epoch * len(train_dataloader)

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        local_head_model.train()
        if hasattr(local_head_model, "embed_tokens"):
            local_head_model.embed_tokens.eval()
        if getattr(local_head_model, "target_lm_head", None) is not None:
            local_head_model.target_lm_head.eval()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
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

    save_checkpoint(
        args, args.num_epochs, global_step, fsdp_model, local_head_model, optimizer
    )

    tracker.close()
    if args.save_unigram_prior:
        dist.barrier()
        if dist.get_rank() == 0:
            save_unigram_prior(args, tokenizer, unigram_dataset)
        dist.barrier()
    destroy_distributed()


if __name__ == "__main__":
    main()
