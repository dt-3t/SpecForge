# coding=utf-8
"""Local-head online training wrapper."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_LM_HEAD_MODES = {"low_rank", "target_lm_head"}
_VALID_RNN_PARAMETERIZATIONS = {"target_factorized", "direct_vocab"}
_VALID_INIT_MODES = {"random", "zero"}
_VALID_RANK_ACTIVATIONS = {"identity", "silu"}


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Compute per-block consecutive correct predictions from the first target."""
    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


def _init_linear_weight(module: nn.Linear, init_mode: str) -> None:
    if init_mode not in _VALID_INIT_MODES:
        raise ValueError(f"init_mode={init_mode!r}; must be one of {_VALID_INIT_MODES}")
    if init_mode == "zero":
        nn.init.zeros_(module.weight)


class OnlineLocalHeadModel(nn.Module):
    """Train an embedding-GRU-low-rank local head with Domino-style block sampling."""

    def __init__(
        self,
        target_embed_tokens: Optional[nn.Module],
        vocab_size: int,
        embed_dim: int,
        gru_hidden_dim: int = 1024,
        low_rank_dim: int = 256,
        input_rank_dim: Optional[int] = None,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        lm_head_mode: str = "target_lm_head",
        lm_head_init: str = "random",
        rnn_parameterization: str = "target_factorized",
        rank_activation: str = "identity",
        target_lm_head: Optional[nn.Module] = None,
        up_proj_init: str = "random",
    ):
        super().__init__()
        if rnn_parameterization not in _VALID_RNN_PARAMETERIZATIONS:
            raise ValueError(
                f"rnn_parameterization={rnn_parameterization!r}; must be one of "
                f"{_VALID_RNN_PARAMETERIZATIONS}"
            )
        if lm_head_mode not in _VALID_LM_HEAD_MODES:
            raise ValueError(
                f"lm_head_mode={lm_head_mode!r}; must be one of {_VALID_LM_HEAD_MODES}"
            )
        if rnn_parameterization == "direct_vocab" and lm_head_mode != "low_rank":
            raise ValueError(
                "direct_vocab RNN local heads require lm_head_mode='low_rank'"
            )
        if (
            rnn_parameterization == "target_factorized"
            and lm_head_mode != "target_lm_head"
        ):
            raise ValueError(
                "target_factorized RNN local heads require "
                "lm_head_mode='target_lm_head'"
            )
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        if low_rank_dim <= 0:
            raise ValueError("low_rank_dim must be positive")
        if input_rank_dim is None:
            input_rank_dim = low_rank_dim
        if input_rank_dim <= 0:
            raise ValueError("input_rank_dim must be positive")
        if rank_activation not in _VALID_RANK_ACTIVATIONS:
            raise ValueError(
                f"rank_activation={rank_activation!r}; must be one of "
                f"{_VALID_RANK_ACTIVATIONS}"
            )

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.input_rank_dim = int(input_rank_dim)
        self.gru_input_dim = int(input_rank_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.low_rank_dim = int(low_rank_dim)
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.lm_head_mode = lm_head_mode
        self.rnn_parameterization = rnn_parameterization
        self.rank_activation = rank_activation
        self.target_hidden_size = None

        self.embed_down_proj = None
        if rnn_parameterization == "direct_vocab":
            self.embed_tokens = nn.Embedding(self.vocab_size, self.input_rank_dim)
            self.embed_dim = self.input_rank_dim
        else:
            if target_embed_tokens is None:
                raise ValueError(
                    "target_embed_tokens is required for target_factorized RNN "
                    "local heads"
                )
            self.embed_tokens = target_embed_tokens
            self.embed_tokens.requires_grad_(False)
            self.embed_tokens.eval()
            self.embed_down_proj = nn.Linear(
                self.embed_dim, self.input_rank_dim, bias=False
            )

        self.prefix_gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.rank_proj = nn.Linear(gru_hidden_dim, low_rank_dim, bias=False)

        self.low_rank_lm_head = None
        self.rank_to_hidden = None
        self.target_lm_head = None
        if lm_head_mode == "low_rank":
            self.low_rank_lm_head = nn.Linear(low_rank_dim, vocab_size, bias=False)
            _init_linear_weight(self.low_rank_lm_head, lm_head_init)
        else:
            if target_lm_head is None:
                raise ValueError("target_lm_head is required for target_lm_head mode")
            target_hidden_size = target_lm_head.weight.shape[1]
            self.target_hidden_size = int(target_hidden_size)
            self.rank_to_hidden = nn.Linear(
                low_rank_dim, target_hidden_size, bias=False
            )
            _init_linear_weight(self.rank_to_hidden, up_proj_init)
            self.target_lm_head = target_lm_head
            self.target_lm_head.requires_grad_(False)
            self.target_lm_head.eval()

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; matches OnlineDominoModel."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = max(1, min(self.num_anchors, int(valid_counts.max().item()) - 1))

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _gather_block_ids(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        label_indices = anchor_positions.unsqueeze(-1) + offsets
        valid_label_mask = label_indices < input_ids.size(1)
        safe_indices = label_indices.clamp(max=input_ids.size(1) - 1)
        gathered = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_indices,
        )
        return gathered, safe_indices, valid_label_mask

    def _build_ntp_inputs(
        self, input_ids: torch.Tensor, anchor_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = input_ids.device
        bs = self.block_size

        prev_offsets = torch.arange(0, bs - 1, device=device).view(1, 1, -1)
        target_offsets = torch.arange(1, bs, device=device).view(1, 1, -1)

        prev_ids, _, _ = self._gather_block_ids(
            input_ids, anchor_positions, prev_offsets
        )
        target_ids, safe_target_indices, valid_label_mask = self._gather_block_ids(
            input_ids, anchor_positions, target_offsets
        )

        return (
            prev_ids,
            target_ids,
            safe_target_indices,
            valid_label_mask,
        )

    def _compute_logits(self, prev_ids: torch.Tensor) -> torch.Tensor:
        bsz, n, t = prev_ids.shape
        block_emb = self.embed_tokens(prev_ids)
        if self.embed_down_proj is not None:
            block_emb = self.embed_down_proj(block_emb)
        gru_inputs = block_emb.reshape(bsz * n, t, -1)
        gru_out = self.prefix_gru(gru_inputs)[0]
        rank_states = self.rank_proj(gru_out).reshape(bsz, n, t, -1)
        if self.rank_activation == "silu":
            rank_states = F.silu(rank_states)

        if self.lm_head_mode == "low_rank":
            return self.low_rank_lm_head(rank_states)

        hidden_states = self.rank_to_hidden(rank_states)
        return self.target_lm_head(hidden_states)

    def _compute_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)
        valid_token_count = flat_weights.sum() + 1e-6

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        return loss, flat_logits, flat_targets

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        flat_logits: torch.Tensor,
        flat_targets: torch.Tensor,
        target_ids: torch.Tensor,
        eval_weight_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        binary_eval_mask = eval_weight_mask.reshape(-1)
        pred_ids = torch.argmax(flat_logits, dim=-1)
        correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
        actual_token_count = binary_eval_mask.sum() + 1e-6
        accuracy = correct.sum().float() / actual_token_count

        valid_mask_4d = (eval_weight_mask > 0).bool()
        pred_accept_len = compute_accept_len(
            pred_ids.view_as(target_ids), target_ids, valid_mask_4d
        )
        valid_block_mask = valid_mask_4d.any(dim=2)
        num_valid_blocks = valid_block_mask.sum().float() + 1e-6
        avg_accept_len = (
            (pred_accept_len + 1.0) * valid_block_mask.float()
        ).sum() / num_valid_blocks

        return accuracy, {
            "final_loss": F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction="none",
            )
            .mul(binary_eval_mask)
            .sum()
            .div(actual_token_count)
            .detach(),
            "accept_len": avg_accept_len.detach(),
        }

    def forward(self, input_ids: torch.Tensor, loss_mask: torch.Tensor):
        """Parallel block-wise local-head next-token-prediction loss."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        prev_ids, target_ids, safe_target_indices, valid_label_mask = (
            self._build_ntp_inputs(input_ids, anchor_positions)
        )

        logits = self._compute_logits(prev_ids)

        weight_mask = (
            block_keep_mask.unsqueeze(-1)
            .expand(-1, -1, target_ids.size(2))
            .float()
        )
        weight_mask = weight_mask * valid_label_mask.float()
        gathered_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )
        weight_mask = weight_mask * gathered_loss_mask
        eval_weight_mask = weight_mask.clone()

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(target_ids.size(2), device=device).view(1, 1, -1)
            decay_weights = torch.exp(-k.float() / self.loss_decay_gamma)
            weight_mask = weight_mask * decay_weights

        loss, flat_logits, flat_targets = self._compute_loss(
            logits=logits,
            target_ids=target_ids,
            weight_mask=weight_mask,
        )

        with torch.no_grad():
            accuracy, metrics = self._compute_metrics(
                logits=logits,
                flat_logits=flat_logits,
                flat_targets=flat_targets,
                target_ids=target_ids,
                eval_weight_mask=eval_weight_mask,
            )

        return loss, accuracy, metrics

    def fused_lm_head_weight(self) -> torch.Tensor:
        """Return a vocab x rank low-rank head usable after training."""
        if self.lm_head_mode == "low_rank":
            return self.low_rank_lm_head.weight.detach()

        target_weight = self.target_lm_head.weight.detach().to(torch.float32)
        up_weight = self.rank_to_hidden.weight.detach().to(torch.float32)
        return target_weight.matmul(up_weight).to(self.rank_to_hidden.weight.dtype)

    def merged_embedding_weight(self) -> torch.Tensor:
        """Return a vocab x input-rank embedding usable after training."""
        if self.rnn_parameterization == "direct_vocab":
            return self.embed_tokens.weight.detach()

        embed_weight = self.embed_tokens.weight.detach().to(torch.float32)
        down_weight = self.embed_down_proj.weight.detach().to(torch.float32)
        return embed_weight.matmul(down_weight.t()).to(self.embed_down_proj.weight.dtype)


class OnlineMarkovLocalHeadModel(nn.Module):
    """Train token_i -> token_{i+1} through frozen target embedding/lm_head."""

    def __init__(
        self,
        target_embed_tokens: nn.Module,
        target_lm_head: nn.Module,
        rank: int = 256,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        target_hidden_size = target_lm_head.weight.shape[1]
        embed_dim = target_embed_tokens.weight.shape[1]
        if embed_dim != target_hidden_size:
            raise ValueError(
                f"Embedding dim ({embed_dim}) must match target lm_head hidden size "
                f"({target_hidden_size}) for Markov local head."
            )

        self.embed_tokens = target_embed_tokens
        self.embed_tokens.requires_grad_(False)
        self.embed_tokens.eval()

        self.target_lm_head = target_lm_head
        self.target_lm_head.requires_grad_(False)
        self.target_lm_head.eval()

        self.vocab_size = target_lm_head.weight.shape[0]
        self.target_hidden_size = target_hidden_size
        self.rank = rank
        self.architecture = "target_embedding_down_up_target_lm_head"

        self.down_proj = nn.Parameter(torch.empty(target_hidden_size, rank))
        self.up_proj = nn.Parameter(torch.zeros(rank, target_hidden_size))
        nn.init.uniform_(
            self.down_proj,
            -target_hidden_size**-0.5,
            target_hidden_size**-0.5,
        )

    def _build_dense_ntp(
        self, input_ids: torch.Tensor, loss_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.size(1) < 2:
            raise ValueError("input_ids must contain at least two tokens")
        prev_ids = input_ids[:, :-1]
        target_ids = input_ids[:, 1:]
        target_weight = loss_mask[:, 1:].to(dtype=torch.float32)
        return prev_ids, target_ids, target_weight

    def _compute_logits(self, prev_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(prev_ids)
        rank_states = hidden.matmul(self.down_proj)
        projected_hidden = rank_states.matmul(self.up_proj)
        return self.target_lm_head(projected_hidden)

    def _compute_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        flat_weights = target_weight.reshape(-1)
        valid_token_count = flat_weights.sum() + 1e-6

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss = (loss_per_token * flat_weights).sum() / valid_token_count
        return loss, flat_logits, flat_targets

    def _compute_metrics(
        self,
        flat_logits: torch.Tensor,
        flat_targets: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flat_weights = target_weight.reshape(-1)
        pred_ids = torch.argmax(flat_logits, dim=-1)
        valid = flat_weights > 0.5
        valid_token_count = valid.float().sum() + 1e-6
        accuracy = (
            ((pred_ids == flat_targets) & valid).float().sum() / valid_token_count
        )
        final_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        final_loss = (final_loss * flat_weights).sum() / (flat_weights.sum() + 1e-6)
        return accuracy.detach(), {
            "final_loss": final_loss.detach(),
            "valid_tokens": flat_weights.sum().detach(),
        }

    def forward(self, input_ids: torch.Tensor, loss_mask: torch.Tensor):
        """Dense next-token-prediction loss over valid target positions."""
        prev_ids, target_ids, target_weight = self._build_dense_ntp(
            input_ids, loss_mask
        )
        logits = self._compute_logits(prev_ids)
        loss, flat_logits, flat_targets = self._compute_loss(
            logits=logits,
            target_ids=target_ids,
            target_weight=target_weight,
        )
        with torch.no_grad():
            accuracy, metrics = self._compute_metrics(
                flat_logits=flat_logits,
                flat_targets=flat_targets,
                target_weight=target_weight,
            )
        return loss, accuracy, metrics

    def merged_embedding_weight(self) -> torch.Tensor:
        """Return vocab x rank embedding after folding target embeddings."""
        embed_weight = self.embed_tokens.weight.detach().to(torch.float32)
        down_weight = self.down_proj.detach().to(torch.float32)
        return embed_weight.matmul(down_weight).to(self.down_proj.dtype)

    def fused_lm_head_weight(self) -> torch.Tensor:
        """Return vocab x rank lm_head after folding up_proj into target lm_head."""
        target_weight = self.target_lm_head.weight.detach().to(torch.float32)
        up_weight = self.up_proj.detach().to(torch.float32)
        return target_weight.matmul(up_weight.t()).to(self.up_proj.dtype)


class OnlineDirectVocabMarkovLocalHeadModel(nn.Module):
    """Train token_i -> token_{i+1} with direct vocab x rank factors."""

    def __init__(
        self,
        vocab_size: int,
        rank: int = 256,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.vocab_size = int(vocab_size)
        self.target_hidden_size = None
        self.rank = int(rank)
        self.architecture = "direct_vocab_low_rank"

        self.embedding_weight = nn.Parameter(torch.empty(self.vocab_size, self.rank))
        self.lm_head_weight = nn.Parameter(torch.zeros(self.vocab_size, self.rank))
        nn.init.uniform_(
            self.embedding_weight,
            -self.rank**-0.5,
            self.rank**-0.5,
        )

    def _build_dense_ntp(
        self, input_ids: torch.Tensor, loss_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.size(1) < 2:
            raise ValueError("input_ids must contain at least two tokens")
        prev_ids = input_ids[:, :-1]
        target_ids = input_ids[:, 1:]
        target_weight = loss_mask[:, 1:].to(dtype=torch.float32)
        return prev_ids, target_ids, target_weight

    def _compute_logits(self, prev_ids: torch.Tensor) -> torch.Tensor:
        rank_states = F.embedding(prev_ids, self.embedding_weight)
        return torch.matmul(rank_states, self.lm_head_weight.t())

    def _compute_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        flat_weights = target_weight.reshape(-1)
        valid_token_count = flat_weights.sum() + 1e-6

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss = (loss_per_token * flat_weights).sum() / valid_token_count
        return loss, flat_logits, flat_targets

    def _compute_metrics(
        self,
        flat_logits: torch.Tensor,
        flat_targets: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flat_weights = target_weight.reshape(-1)
        pred_ids = torch.argmax(flat_logits, dim=-1)
        valid = flat_weights > 0.5
        valid_token_count = valid.float().sum() + 1e-6
        accuracy = (
            ((pred_ids == flat_targets) & valid).float().sum() / valid_token_count
        )
        final_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        final_loss = (final_loss * flat_weights).sum() / (flat_weights.sum() + 1e-6)
        return accuracy.detach(), {
            "final_loss": final_loss.detach(),
            "valid_tokens": flat_weights.sum().detach(),
        }

    def forward(self, input_ids: torch.Tensor, loss_mask: torch.Tensor):
        """Dense next-token-prediction loss over valid target positions."""
        prev_ids, target_ids, target_weight = self._build_dense_ntp(
            input_ids, loss_mask
        )
        logits = self._compute_logits(prev_ids)
        loss, flat_logits, flat_targets = self._compute_loss(
            logits=logits,
            target_ids=target_ids,
            target_weight=target_weight,
        )
        with torch.no_grad():
            accuracy, metrics = self._compute_metrics(
                flat_logits=flat_logits,
                flat_targets=flat_targets,
                target_weight=target_weight,
            )
        return loss, accuracy, metrics

    def merged_embedding_weight(self) -> torch.Tensor:
        """Return the trainable vocab x rank embedding factor."""
        return self.embedding_weight.detach()

    def fused_lm_head_weight(self) -> torch.Tensor:
        """Return the trainable vocab x rank output factor."""
        return self.lm_head_weight.detach()
