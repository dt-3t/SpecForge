from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.train_local_head import (
    _load_local_head_checkpoint,
    _merged_rnn_state,
    build_local_head_model,
)
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


def test_direct_vocab_rnn_uses_target_config_vocab_size(monkeypatch):
    class TinyTokenizer:
        def __len__(self):
            return 11

    class DummyAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return SimpleNamespace(vocab_size=13)

    import scripts.train_local_head as train_local_head

    monkeypatch.setattr(train_local_head, "AutoConfig", DummyAutoConfig)
    monkeypatch.setattr(train_local_head, "print_on_rank0", lambda message: None)
    args = SimpleNamespace(
        local_head_type="rnn",
        rnn_parameterization="direct_vocab",
        target_model_path="dummy-target",
        trust_remote_code=False,
        local_input_rank=None,
        local_rank=3,
        local_gru_hidden_dim=5,
        block_size=4,
        num_anchors=2,
        loss_decay_gamma=None,
        local_lm_head_init="random",
        local_rank_activation="identity",
        local_up_proj_init="random",
    )

    model = build_local_head_model(
        args,
        target_components=None,
        tokenizer=TinyTokenizer(),
        device=torch.device("cpu"),
    )

    assert model.vocab_size == 13
    assert model.embed_tokens.weight.shape == (13, 3)
    assert model.low_rank_lm_head.weight.shape == (13, 3)


def test_load_local_head_checkpoint_accepts_checkpoint_directory(tmp_path):
    source = OnlineLocalHeadModel(
        target_embed_tokens=None,
        target_lm_head=None,
        vocab_size=7,
        embed_dim=3,
        input_rank_dim=3,
        gru_hidden_dim=5,
        low_rank_dim=4,
        block_size=4,
        lm_head_mode="low_rank",
        rnn_parameterization="direct_vocab",
    )
    target = OnlineLocalHeadModel(
        target_embed_tokens=None,
        target_lm_head=None,
        vocab_size=7,
        embed_dim=3,
        input_rank_dim=3,
        gru_hidden_dim=5,
        low_rank_dim=4,
        block_size=4,
        lm_head_mode="low_rank",
        rnn_parameterization="direct_vocab",
    )
    with torch.no_grad():
        for param in source.parameters():
            param.fill_(0.25)
        for param in target.parameters():
            param.zero_()

    checkpoint_dir = tmp_path / "epoch_0_step_100"
    checkpoint_dir.mkdir()
    torch.save(
        {
            "epoch": 3,
            "global_step": 100,
            "model_config": {"local_head_type": "rnn"},
            "model_state_dict": source.state_dict(),
        },
        checkpoint_dir / "local_head.pt",
    )

    payload = _load_local_head_checkpoint(target, str(checkpoint_dir), "rnn")

    assert payload["epoch"] == 3
    assert payload["global_step"] == 100
    assert payload["_checkpoint_path"] == str(checkpoint_dir / "local_head.pt")
    for name, tensor in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], tensor)


def test_load_local_head_checkpoint_rejects_merged_artifact(tmp_path):
    checkpoint_path = tmp_path / "merged_rnn_local_head.pt"
    torch.save({"embedding_weight": torch.randn(7, 3)}, checkpoint_path)
    target = OnlineLocalHeadModel(
        target_embed_tokens=None,
        target_lm_head=None,
        vocab_size=7,
        embed_dim=3,
        input_rank_dim=3,
        gru_hidden_dim=5,
        low_rank_dim=4,
        block_size=4,
        lm_head_mode="low_rank",
        rnn_parameterization="direct_vocab",
    )

    try:
        _load_local_head_checkpoint(target, str(checkpoint_path), "rnn")
    except RuntimeError as exc:
        assert "model_state_dict" in str(exc)
    else:
        raise AssertionError("Expected merged artifact to be rejected")
