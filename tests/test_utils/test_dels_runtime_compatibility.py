"""CPU integration checks against a neighboring DeLS-Spec runtime checkout.

Set DELS_RUNTIME_ROOT to the runtime source root when running these tests.
"""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.train_local_head import _merged_rnn_state
from specforge.core.local_head import OnlineLocalHeadModel


@pytest.fixture(scope="module")
def runtime():
    root = Path(os.environ.get("DELS_RUNTIME_ROOT", Path(__file__).resolve().parents[3] / "DeLS-Spec"))
    path = root / "code" / "dels.py"
    if not path.is_file():
        pytest.skip("Set DELS_RUNTIME_ROOT to the matching DeLS-Spec checkout")
    spec = importlib.util.spec_from_file_location("dels_runtime_integration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _target():
    return SimpleNamespace(
        model=SimpleNamespace(embed_tokens=nn.Embedding(13, 7)),
        lm_head=nn.Linear(7, 13, bias=False),
    )


def _model(target, parameterization, activation):
    return OnlineLocalHeadModel(
        target_embed_tokens=target.model.embed_tokens if parameterization == "target_factorized" else None,
        target_lm_head=target.lm_head if parameterization == "target_factorized" else None,
        vocab_size=13,
        embed_dim=7 if parameterization == "target_factorized" else 3,
        input_rank_dim=3,
        gru_hidden_dim=5,
        low_rank_dim=4,
        block_size=4,
        lm_head_mode="target_lm_head" if parameterization == "target_factorized" else "low_rank",
        rnn_parameterization=parameterization,
        rank_activation=activation,
    )


def _export(model, parameterization):
    args = SimpleNamespace(
        local_head_type="rnn", target_model_path="dummy-target",
        embedding_key=None, lm_head_key=None,
        local_head_parameterization=parameterization,
    )
    return _merged_rnn_state(args, model, model.state_dict())


@pytest.mark.parametrize("parameterization", ["target_factorized", "direct_vocab"])
@pytest.mark.parametrize("activation", ["identity", "silu"])
def test_export_load_and_causal_logits(runtime, tmp_path, parameterization, activation):
    torch.manual_seed(17)
    target = _target()
    model = _model(target, parameterization, activation)
    original_embedding = target.model.embed_tokens.weight.detach().clone()
    path = tmp_path / "merged_rnn_local_head.pt"
    torch.save(_export(model, parameterization), path)
    assert runtime.detect_local_head_checkpoint_kind(str(path)) == "dels"
    loaded = runtime.DeLSLocalHead.from_checkpoint(
        checkpoint_path=str(path), target_model=target,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    ids = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]])
    expected = model._compute_logits(ids).detach()[:, 0]
    hidden = loaded.init_empty_hidden(2, device=torch.device("cpu"))
    logits = []
    for step in range(3):
        hidden = loaded.advance(ids[:, 0, step:step + 1], hidden)
        logits.append(loaded.logits_from_hidden(hidden)[:, 0])
    torch.testing.assert_close(torch.stack(logits, dim=1), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(target.model.embed_tokens.weight, original_embedding)
    assert loaded.embed_tokens is not target.model.embed_tokens
    assert loaded.embed_tokens.weight.shape == (13, 3)


@pytest.mark.parametrize("mode", ["low_rank", "target_lm_head"])
def test_legacy_rnn_checkpoint_still_loads(runtime, tmp_path, mode):
    torch.manual_seed(9)
    target = _target()
    gru = nn.GRU(7, 5, batch_first=True, bias=False)
    proj = nn.Linear(5, 4, bias=False)
    head = nn.Linear(4, 13 if mode == "low_rank" else 7, bias=False)
    state = {"prefix_gru." + k: v for k, v in gru.state_dict().items()}
    state["rank_proj.weight"] = proj.weight
    state["low_rank_lm_head.weight" if mode == "low_rank" else "rank_to_hidden.weight"] = head.weight
    config = dict(vocab_size=13, embed_dim=7, gru_hidden_dim=5, low_rank_dim=4,
                  lm_head_mode=mode, rank_activation="silu", shift_label=True,
                  pure_draft_prefix_len=0)
    path = tmp_path / "local_head.pt"
    torch.save(dict(model_config=config, model_state_dict=state), path)
    loaded = runtime.DeLSLocalHead.from_checkpoint(
        checkpoint_path=str(path), target_model=target,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    tokens = torch.tensor([[1, 2]])
    hidden = gru(target.model.embed_tokens(tokens))[1]
    expected = head(torch.nn.functional.silu(proj(hidden.transpose(0, 1))))
    if mode == "target_lm_head":
        expected = target.lm_head(expected)
    torch.testing.assert_close(loaded.logits_from_hidden(hidden), expected)
    assert loaded.shift_label and loaded.pure_draft_prefix_len == 0


def test_unmerged_projected_rnn_has_actionable_error(runtime, tmp_path):
    target = _target()
    model = _model(target, "target_factorized", "silu")
    payload = _export(model, "target_factorized")
    payload["model_config"]["rnn_parameterization"] = "target_factorized"
    payload["model_config"].pop("architecture")
    payload["model_state_dict"] = model.state_dict()
    path = tmp_path / "local_head.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="merged_rnn_local_head.pt"):
        runtime.DeLSLocalHead.from_checkpoint(
            checkpoint_path=str(path), target_model=target,
            dtype=torch.float32, device=torch.device("cpu"),
        )


def test_markov_export_keeps_markov_detection(runtime, tmp_path):
    path = tmp_path / "merged_markov_local_head.pt"
    torch.save(dict(embedding_weight=torch.rand(13, 3), lm_head_weight=torch.rand(13, 3)), path)
    assert runtime.detect_local_head_checkpoint_kind(str(path)) == "markov"


def test_corrupt_embedding_is_rejected(runtime, tmp_path):
    target = _target()
    payload = _export(_model(target, "direct_vocab", "identity"), "direct_vocab")
    payload["model_state_dict"]["embed_tokens.weight"] = torch.rand(13, 2)
    path = tmp_path / "broken.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="embedding shape"):
        runtime.DeLSLocalHead.from_checkpoint(
            checkpoint_path=str(path), target_model=target,
            dtype=torch.float32, device=torch.device("cpu"),
        )
