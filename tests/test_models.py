"""Unit tests for PolicyModel and RewardModel.

These tests use tiny in-process models (random weights) to keep them fast
and dependency-free at CI time.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, GPT2Model


# ---------------------------------------------------------------------------
# Fixtures: build tiny GPT-2 style models without downloading weights
# ---------------------------------------------------------------------------


def _tiny_gpt2_config() -> GPT2Config:
    return GPT2Config(
        vocab_size=100,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_positions=64,
    )


@pytest.fixture
def tiny_policy(tmp_path):
    """A :class:`~rlhf.models.PolicyModel` backed by a tiny random GPT-2."""
    config = _tiny_gpt2_config()
    lm = GPT2LMHeadModel(config)
    model_dir = str(tmp_path / "policy")
    lm.save_pretrained(model_dir)

    from rlhf.models.policy_model import PolicyModel

    pm = PolicyModel.__new__(PolicyModel)
    nn.Module.__init__(pm)
    pm.device = "cpu"
    pm.model = lm

    # Build a minimal tokenizer stub
    from tests.test_data import _DummyTokenizer

    tok = _DummyTokenizer()
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    pm.tokenizer = tok
    return pm


@pytest.fixture
def tiny_reward():
    """A :class:`~rlhf.models.RewardModel` backed by a tiny random GPT-2."""
    config = _tiny_gpt2_config()
    backbone = GPT2Model(config)

    from rlhf.models.reward_model import RewardModel

    rm = RewardModel.__new__(RewardModel)
    nn.Module.__init__(rm)
    rm.device = "cpu"
    rm.backbone = backbone
    rm.value_head = nn.Linear(config.n_embd, 1)

    from tests.test_data import _DummyTokenizer

    tok = _DummyTokenizer()
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    rm.tokenizer = tok
    return rm


# ---------------------------------------------------------------------------
# PolicyModel tests
# ---------------------------------------------------------------------------


def test_policy_forward_returns_loss(tiny_policy):
    B, T = 2, 16
    ids = torch.randint(0, 100, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    labels = ids.clone()
    labels[mask == 0] = -100

    out = tiny_policy(ids, mask, labels)
    assert out.loss is not None
    assert out.loss.item() > 0


def test_policy_log_probs_shape(tiny_policy):
    B, T = 2, 16
    ids = torch.randint(0, 100, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    log_probs = tiny_policy.log_probs_of(ids, mask)
    assert log_probs.shape == (B, T - 1)


def test_policy_log_probs_are_non_positive(tiny_policy):
    ids = torch.randint(0, 100, (1, 16))
    mask = torch.ones(1, 16, dtype=torch.long)
    log_probs = tiny_policy.log_probs_of(ids, mask)
    assert (log_probs <= 0).all()


# ---------------------------------------------------------------------------
# RewardModel tests
# ---------------------------------------------------------------------------


def test_reward_model_forward_shape(tiny_reward):
    B, T = 3, 16
    ids = torch.randint(0, 100, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    scores = tiny_reward(ids, mask)
    assert scores.shape == (B,)


def test_reward_model_forward_no_mask(tiny_reward):
    ids = torch.randint(0, 100, (2, 16))
    scores = tiny_reward(ids)
    assert scores.shape == (2,)


def test_reward_model_scores_are_scalar(tiny_reward):
    ids = torch.randint(0, 100, (1, 8))
    scores = tiny_reward(ids)
    assert scores.ndim == 1


def test_reward_model_save_load(tiny_reward, tmp_path):
    out_dir = str(tmp_path / "reward")
    # Patch backbone.save_pretrained to avoid file-system calls to HF hub
    import types

    tiny_reward.backbone.save_pretrained = lambda d: None
    tiny_reward.tokenizer.save_pretrained = lambda d: None

    import os

    os.makedirs(out_dir, exist_ok=True)
    # Only test value head serialisation
    import torch

    torch.save(tiny_reward.value_head.state_dict(), os.path.join(out_dir, "value_head.pt"))
    loaded = torch.load(os.path.join(out_dir, "value_head.pt"))
    assert "weight" in loaded
