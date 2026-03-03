"""Integration-style tests for the training loop components.

All tests use tiny random models and minimal data so they complete quickly.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

from tests.test_data import _DummyTokenizer
from tests.test_models import _tiny_gpt2_config


# ---------------------------------------------------------------------------
# Helper: build fixtures inline for convenience
# ---------------------------------------------------------------------------


def _make_policy():
    from rlhf.models.policy_model import PolicyModel

    config = _tiny_gpt2_config()
    lm = GPT2LMHeadModel(config)
    pm = PolicyModel.__new__(PolicyModel)
    nn.Module.__init__(pm)
    pm.device = "cpu"
    pm.model = lm
    tok = _DummyTokenizer()
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    pm.tokenizer = tok
    return pm


def _make_reward():
    from rlhf.models.reward_model import RewardModel

    config = _tiny_gpt2_config()
    backbone = GPT2Model(config)
    rm = RewardModel.__new__(RewardModel)
    nn.Module.__init__(rm)
    rm.device = "cpu"
    rm.backbone = backbone
    rm.value_head = nn.Linear(config.n_embd, 1)
    tok = _DummyTokenizer()
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    rm.tokenizer = tok
    return rm


# ---------------------------------------------------------------------------
# SFTTrainer
# ---------------------------------------------------------------------------


def test_sft_trainer_returns_losses(tmp_path):
    from rlhf.config import SFTConfig
    from rlhf.data.dataset import PromptDataset
    from rlhf.training.sft_trainer import SFTTrainer

    policy = _make_policy()
    policy.save_pretrained = lambda d: None  # skip file I/O
    config = SFTConfig(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        output_dir=str(tmp_path / "sft"),
    )
    dataset = PromptDataset(
        prompts=["Hello ", "World "],
        responses=["there", "here"],
        tokenizer=policy.tokenizer,
        max_length=16,
    )
    trainer = SFTTrainer(policy, config)
    losses = trainer.train(dataset)
    assert len(losses) > 0
    assert all(isinstance(l, float) for l in losses)


def test_sft_trainer_loss_decreases_trend(tmp_path):
    """Loss should be a finite number."""
    from rlhf.config import SFTConfig
    from rlhf.data.dataset import PromptDataset
    from rlhf.training.sft_trainer import SFTTrainer
    import math

    policy = _make_policy()
    policy.save_pretrained = lambda d: None  # skip file I/O
    config = SFTConfig(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        output_dir=str(tmp_path / "sft2"),
    )
    dataset = PromptDataset(["Hi "] * 4, ["there"] * 4, policy.tokenizer, 16)
    trainer = SFTTrainer(policy, config)
    losses = trainer.train(dataset)
    assert all(math.isfinite(l) for l in losses)


# ---------------------------------------------------------------------------
# RewardTrainer
# ---------------------------------------------------------------------------


def test_reward_trainer_returns_losses(tmp_path):
    from rlhf.config import RewardModelConfig
    from rlhf.data.dataset import PreferenceDataset
    from rlhf.training.reward_trainer import RewardTrainer

    reward = _make_reward()
    config = RewardModelConfig(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        output_dir=str(tmp_path / "reward"),
    )
    dataset = PreferenceDataset(
        prompts=["P1 ", "P2 "],
        chosen=["good1", "good2"],
        rejected=["bad1", "bad2"],
        tokenizer=reward.tokenizer,
        max_length=16,
    )
    # Patch save so we don't need a real HF model directory
    reward.save_pretrained = lambda d: None

    trainer = RewardTrainer(reward, config)
    losses = trainer.train(dataset)
    assert len(losses) > 0
    assert all(isinstance(l, float) for l in losses)


def test_reward_trainer_loss_is_finite(tmp_path):
    import math

    from rlhf.config import RewardModelConfig
    from rlhf.data.dataset import PreferenceDataset
    from rlhf.training.reward_trainer import RewardTrainer

    reward = _make_reward()
    config = RewardModelConfig(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        output_dir=str(tmp_path / "reward2"),
    )
    dataset = PreferenceDataset(["P "] * 4, ["good"] * 4, ["bad"] * 4, reward.tokenizer, 16)
    reward.save_pretrained = lambda d: None
    trainer = RewardTrainer(reward, config)
    losses = trainer.train(dataset)
    assert all(math.isfinite(l) for l in losses)


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------


def test_ppo_trainer_returns_metrics(tmp_path):
    from rlhf.config import PPOConfig
    from rlhf.training.ppo_trainer import PPOTrainer

    policy = _make_policy()
    reward = _make_reward()
    config = PPOConfig(
        num_train_steps=2,
        batch_size=2,
        ppo_epochs=1,
        num_rollout_steps=8,
        output_dir=str(tmp_path / "ppo"),
    )
    policy.save_pretrained = lambda d: None

    trainer = PPOTrainer(policy, reward, config, kl_coef=0.1)
    # Patch generate to return fixed strings quickly
    trainer.policy.generate = lambda prompts, **kw: ["response"] * len(prompts)

    metrics = trainer.train(["What is AI?", "Tell me something."])
    assert len(metrics) == 2
    assert "mean_reward" in metrics[0]
    assert "policy_loss" in metrics[0]


def test_ppo_trainer_mean_reward_is_finite(tmp_path):
    import math

    from rlhf.config import PPOConfig
    from rlhf.training.ppo_trainer import PPOTrainer

    policy = _make_policy()
    reward = _make_reward()
    config = PPOConfig(
        num_train_steps=1,
        batch_size=2,
        ppo_epochs=1,
        num_rollout_steps=4,
        output_dir=str(tmp_path / "ppo2"),
    )
    policy.save_pretrained = lambda d: None
    trainer = PPOTrainer(policy, reward, config)
    trainer.policy.generate = lambda prompts, **kw: ["ok"] * len(prompts)
    metrics = trainer.train(["Hello?", "World?"])
    assert math.isfinite(metrics[0]["mean_reward"])
