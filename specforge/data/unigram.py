# coding=utf-8
"""Utilities for target-token unigram statistics."""

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from tqdm import tqdm


def ensure_count_capacity(counts: torch.Tensor, max_token_id: int) -> torch.Tensor:
    if max_token_id < counts.numel():
        return counts
    expanded = torch.zeros(max_token_id + 1, dtype=counts.dtype)
    expanded[: counts.numel()] = counts
    return expanded


def finalize_unigram_counts(
    counts: torch.Tensor,
    total_target_tokens: int,
    num_samples: int,
) -> Dict[str, torch.Tensor]:
    if total_target_tokens == 0:
        raise ValueError("No target tokens were found.")

    p = counts.to(torch.float64) / float(total_target_tokens)
    log_p = torch.full_like(p, -math.inf)
    nonzero = counts > 0
    log_p[nonzero] = torch.log(p[nonzero])

    return {
        "counts": counts,
        "p": p,
        "log_p": log_p,
        "total_target_tokens": torch.tensor(total_target_tokens, dtype=torch.long),
        "num_samples": torch.tensor(num_samples, dtype=torch.long),
    }


def count_loss_mask_unigrams(
    dataset,
    tokenizer,
    vocab_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if vocab_size is None:
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
        counts = ensure_count_capacity(counts, max_token_id)
        counts += torch.bincount(target_ids, minlength=counts.numel())
        total_target_tokens += int(target_ids.numel())

    return finalize_unigram_counts(
        counts=counts,
        total_target_tokens=total_target_tokens,
        num_samples=len(dataset),
    )


def count_raw_text_unigrams(
    dataset: Iterable[dict],
    tokenizer,
    *,
    vocab_size: int,
    max_length: int,
    batch_size: int = 512,
    max_samples: Optional[int] = None,
    text_column: str = "text",
) -> Dict[str, torch.Tensor]:
    counts = torch.zeros(vocab_size, dtype=torch.long)
    total_target_tokens = 0
    num_samples = 0
    batch: List[str] = []

    def flush(texts: List[str]) -> None:
        nonlocal counts, total_target_tokens, num_samples
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        for ids in encoded["input_ids"]:
            if max_samples is not None and num_samples >= max_samples:
                break
            if len(ids) <= 1:
                continue
            target_ids = torch.tensor(ids[:-1], dtype=torch.long)
            if target_ids.numel() == 0:
                continue
            max_token_id = int(target_ids.max().item())
            counts = ensure_count_capacity(counts, max_token_id)
            counts += torch.bincount(target_ids, minlength=counts.numel())
            total_target_tokens += int(target_ids.numel())
            num_samples += 1

    for row in tqdm(dataset, desc="Counting raw-text target tokens"):
        text = row.get(text_column) or ""
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            flush(batch)
            batch.clear()
            if max_samples is not None and num_samples >= max_samples:
                break

    if batch and (max_samples is None or num_samples < max_samples):
        if max_samples is not None:
            batch = batch[: max_samples - num_samples]
        flush(batch)

    return finalize_unigram_counts(
        counts=counts,
        total_target_tokens=total_target_tokens,
        num_samples=num_samples,
    )


def token_strings(tokenizer, token_id: int) -> Dict[str, str]:
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
            strings = token_strings(tokenizer, token_id)
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
    log_fn=print,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log_fn("matplotlib is not installed; skipping unigram plots.")
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


def save_unigram_outputs(
    stats: Dict[str, torch.Tensor],
    tokenizer,
    output_dir: Path,
    metadata: Dict,
    *,
    top_k: int = 200,
    plot: bool = False,
    plot_top_k: int = 50,
    log_fn=print,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **metadata,
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
        top_k=top_k,
    )
    if plot:
        plot_unigram_distribution(
            stats,
            top_rows,
            output_dir,
            plot_top_k=plot_top_k,
            log_fn=log_fn,
        )

    log_fn(
        "Saved target-token unigram stats to "
        f"{output_dir} with {metadata['total_target_tokens']} target tokens "
        f"from {metadata['num_samples']} samples."
    )
