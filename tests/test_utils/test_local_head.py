from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.train_local_head import _merged_rnn_state
from scripts.convert_local_head import convert_checkpoint
from specforge.core.local_head import OnlineLocalHeadModel


def _args():
    return SimpleNamespace(
        local_head_type="rnn",
        target_model_path="dummy-target",
        embedding_key=None,
        lm_head_key=None,
        local_head_parameterization="direct_vocab",
    )


def test_target_factorized_rnn_fusion_matches_training_path():
    torch.manual_seed(0)
    vocab_size = 11
    target_hidden = 7
    input_rank = 3
    gru_hidden = 5
    output_rank = 4
    target_embed = nn.Embedding(vocab_size, target_hidden)
    target_lm_head = nn.Linear(target_hidden, vocab_size, bias=False)
    model = OnlineLocalHeadModel(
        target_embed_tokens=target_embed,
        target_lm_head=target_lm_head,
        vocab_size=vocab_size,
        embed_dim=target_hidden,
        input_rank_dim=input_rank,
        gru_hidden_dim=gru_hidden,
        low_rank_dim=output_rank,
        block_size=4,
        lm_head_mode="target_lm_head",
        rnn_parameterization="target_factorized",
    )
    prev_ids = torch.tensor([[[1, 2, 3], [4, 5, 6]]])

    training_logits = model._compute_logits(prev_ids)
    embedding_weight = model.merged_embedding_weight()
    lm_head_weight = model.fused_lm_head_weight()

    fused_embeds = F.embedding(prev_ids, embedding_weight)
    bsz, n, t = prev_ids.shape
    gru_out = model.prefix_gru(fused_embeds.reshape(bsz * n, t, -1))[0]
    rank_states = model.rank_proj(gru_out).reshape(bsz, n, t, -1)
    fused_logits = torch.matmul(rank_states, lm_head_weight.t())

    assert torch.allclose(training_logits, fused_logits, atol=1e-5, rtol=1e-5)


def test_merged_rnn_state_is_vocab_space_checkpoint():
    torch.manual_seed(0)
    vocab_size = 13
    model = OnlineLocalHeadModel(
        target_embed_tokens=None,
        target_lm_head=None,
        vocab_size=vocab_size,
        embed_dim=3,
        input_rank_dim=3,
        gru_hidden_dim=5,
        low_rank_dim=4,
        block_size=4,
        lm_head_mode="low_rank",
        rnn_parameterization="direct_vocab",
    )

    payload = _merged_rnn_state(_args(), model, model.state_dict())

    assert payload["embedding_weight"].shape == (vocab_size, 3)
    assert payload["lm_head_weight"].shape == (vocab_size, 4)
    assert payload["model_config"]["rnn_parameterization"] == "direct_vocab"
    assert payload["model_config"]["architecture"] == "direct_vocab_rnn"
    assert payload["model_state_dict"]["embed_tokens.weight"].shape == (vocab_size, 3)


def test_convert_direct_vocab_rnn_local_head():
    checkpoint = {
        "model_config": {
            "local_head_type": "rnn",
            "local_head_parameterization": "direct_vocab",
            "vocab_size": 7,
            "embed_dim": 3,
            "gru_hidden_dim": 5,
            "low_rank_dim": 4,
            "lm_head_mode": "low_rank",
        },
        "model_state_dict": {
            "embed_tokens.weight": torch.randn(7, 3),
            "prefix_gru.weight_ih_l0": torch.randn(15, 3),
            "prefix_gru.weight_hh_l0": torch.randn(15, 5),
            "rank_proj.weight": torch.randn(4, 5),
            "low_rank_lm_head.weight": torch.randn(7, 4),
        },
    }

    payload = convert_checkpoint(checkpoint, _args())

    assert payload["model_config"]["architecture"] == "direct_vocab_rnn"
    assert payload["embedding_weight"].shape == (7, 3)
    assert payload["lm_head_weight"].shape == (7, 4)


def test_convert_direct_vocab_markov_local_head():
    checkpoint = {
        "model_config": {
            "local_head_type": "markov",
            "local_head_parameterization": "direct_vocab",
            "vocab_size": 7,
            "rank": 3,
        },
        "model_state_dict": {
            "embedding_weight": torch.randn(7, 3),
            "lm_head_weight": torch.randn(7, 3),
        },
    }

    payload = convert_checkpoint(checkpoint, _args())

    assert payload["model_config"]["architecture"] == "direct_vocab_low_rank"
    assert payload["embedding_weight"].shape == (7, 3)
    assert payload["lm_head_weight"].shape == (7, 3)
