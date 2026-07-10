#!/usr/bin/env python3
# coding=utf-8
"""Convert training local_head.pt checkpoints to fused vocab-space checkpoints."""

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a DeLS-Spec local_head.pt to merged_local_head.pt."
    )
    parser.add_argument("checkpoint_path", type=str, help="Path to local_head.pt.")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output checkpoint path. Defaults to checkpoint_dir/merged_local_head.pt.",
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        default=None,
        help=(
            "Target model path used to fuse target_factorized checkpoints. "
            "Defaults to model_config.target_model_path."
        ),
    )
    parser.add_argument("--embedding-key", type=str, default=None)
    parser.add_argument("--lm-head-key", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading target embedding/lm_head during fusion.",
    )
    return parser.parse_args()


def _infer_local_head_type(config: dict, state_dict: dict) -> str:
    local_head_type = config.get("local_head_type")
    if local_head_type in {"rnn", "markov"}:
        return local_head_type
    if "down_proj" in state_dict and "up_proj" in state_dict:
        return "markov"
    if any(name.startswith("prefix_gru.") for name in state_dict):
        return "rnn"
    raise ValueError("Cannot infer local head type from checkpoint.")


def _parameterization(config: dict, local_head_type: str) -> str:
    value = config.get("local_head_parameterization")
    if value is None:
        value = config.get(f"{local_head_type}_parameterization")
    if value is None and str(config.get("architecture", "")).startswith("direct_vocab"):
        value = "direct_vocab"
    return value or "target_factorized"


def _load_target_components(args, config: dict, *, required: bool):
    if not required:
        return None

    target_model_path = args.target_model_path or config.get("target_model_path")
    if not target_model_path:
        raise ValueError(
            "target_factorized conversion requires --target-model-path or "
            "model_config.target_model_path."
        )
    return TargetEmbeddingsAndHead.from_pretrained(
        target_model_path,
        embed_key=args.embedding_key or config.get("embedding_key"),
        lm_head_key=args.lm_head_key or config.get("lm_head_key"),
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )


def _target_embed_weight(target_components) -> torch.Tensor:
    return target_components.embed_tokens.weight.detach().cpu()


def _target_lm_head_weight(target_components) -> torch.Tensor:
    return target_components.lm_head.weight.detach().cpu()


def _convert_rnn(config: dict, state_dict: dict, target_components) -> dict:
    parameterization = _parameterization(config, "rnn")
    lm_head_mode = config.get("lm_head_mode", "low_rank")

    if parameterization == "direct_vocab":
        embedding_weight = state_dict["embed_tokens.weight"]
        lm_head_weight = state_dict["low_rank_lm_head.weight"]
    else:
        if target_components is None:
            raise ValueError("RNN target_factorized conversion requires target weights.")
        embed_weight = _target_embed_weight(target_components).to(torch.float32)
        if "embed_down_proj.weight" in state_dict:
            down_weight = state_dict["embed_down_proj.weight"].cpu().to(torch.float32)
            embedding_weight = embed_weight.matmul(down_weight.t()).to(down_weight.dtype)
        else:
            embedding_weight = embed_weight.to(state_dict["prefix_gru.weight_ih_l0"].dtype)

        if lm_head_mode == "target_lm_head":
            target_weight = _target_lm_head_weight(target_components).to(torch.float32)
            up_weight = state_dict["rank_to_hidden.weight"].cpu().to(torch.float32)
            lm_head_weight = target_weight.matmul(up_weight).to(up_weight.dtype)
        else:
            lm_head_weight = state_dict["low_rank_lm_head.weight"]

    model_state_dict = {
        k: v
        for k, v in state_dict.items()
        if k.startswith("prefix_gru.") or k.startswith("rank_proj.")
    }
    model_state_dict["embed_tokens.weight"] = embedding_weight
    model_state_dict["low_rank_lm_head.weight"] = lm_head_weight

    merged_config = copy.deepcopy(config)
    merged_config.update(
        {
            "local_head_type": "rnn",
            "local_head_parameterization": "direct_vocab",
            "source_local_head_parameterization": parameterization,
            "source_lm_head_mode": lm_head_mode,
            "rnn_parameterization": "direct_vocab",
            "lm_head_mode": "low_rank",
            "architecture": "direct_vocab_rnn",
            "vocab_size": int(config.get("vocab_size", embedding_weight.shape[0])),
            "embed_dim": int(embedding_weight.shape[1]),
            "input_rank_dim": int(embedding_weight.shape[1]),
            "gru_input_dim": int(embedding_weight.shape[1]),
            "low_rank_dim": int(lm_head_weight.shape[1]),
            "target_hidden_size": None,
        }
    )

    return {
        "embedding_weight": embedding_weight,
        "lm_head_weight": lm_head_weight,
        "vocab_size": int(embedding_weight.shape[0]),
        "input_rank_dim": int(embedding_weight.shape[1]),
        "low_rank_dim": int(lm_head_weight.shape[1]),
        "model_config": merged_config,
        "model_state_dict": model_state_dict,
    }


def _convert_markov(config: dict, state_dict: dict, target_components) -> dict:
    parameterization = _parameterization(config, "markov")
    if parameterization == "direct_vocab":
        embedding_weight = state_dict["embedding_weight"]
        lm_head_weight = state_dict["lm_head_weight"]
    else:
        if target_components is None:
            raise ValueError("Markov target_factorized conversion requires target weights.")
        embed_weight = _target_embed_weight(target_components).to(torch.float32)
        lm_head_weight_full = _target_lm_head_weight(target_components).to(torch.float32)
        down_weight = state_dict["down_proj"].cpu().to(torch.float32)
        up_weight = state_dict["up_proj"].cpu().to(torch.float32)
        embedding_weight = embed_weight.matmul(down_weight).to(down_weight.dtype)
        lm_head_weight = lm_head_weight_full.matmul(up_weight.t()).to(up_weight.dtype)

    merged_config = copy.deepcopy(config)
    merged_config.update(
        {
            "local_head_type": "markov",
            "local_head_parameterization": "direct_vocab",
            "source_local_head_parameterization": parameterization,
            "markov_parameterization": "direct_vocab",
            "architecture": "direct_vocab_low_rank",
            "target_hidden_size": None,
            "rank": int(embedding_weight.shape[1]),
            "vocab_size": int(embedding_weight.shape[0]),
        }
    )
    return {
        "embedding_weight": embedding_weight,
        "lm_head_weight": lm_head_weight,
        "rank": int(embedding_weight.shape[1]),
        "vocab_size": int(embedding_weight.shape[0]),
        "model_config": merged_config,
    }


def convert_checkpoint(checkpoint: dict, args) -> dict:
    config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Expected checkpoint with model_config and model_state_dict.")

    local_head_type = _infer_local_head_type(config, state_dict)
    parameterization = _parameterization(config, local_head_type)
    target_components = _load_target_components(
        args, config, required=parameterization == "target_factorized"
    )
    if local_head_type == "rnn":
        return _convert_rnn(config, state_dict, target_components)
    return _convert_markov(config, state_dict, target_components)


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else checkpoint_path.with_name("merged_local_head.pt")
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload = convert_checkpoint(checkpoint, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Wrote merged local head to {output_path}")


if __name__ == "__main__":
    main()
