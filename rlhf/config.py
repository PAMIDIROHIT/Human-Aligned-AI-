"""Configuration dataclasses for the RLHF pipeline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFTConfig:
    """Supervised Fine-Tuning configuration."""

    model_name: str = "gpt2"
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    max_seq_length: int = 512
    warmup_steps: int = 100
    weight_decay: float = 0.01
    output_dir: str = "outputs/sft"


@dataclass
class RewardModelConfig:
    """Reward model training configuration."""

    model_name: str = "gpt2"
    learning_rate: float = 1e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    max_seq_length: int = 512
    warmup_steps: int = 100
    weight_decay: float = 0.01
    output_dir: str = "outputs/reward_model"


@dataclass
class PPOConfig:
    """PPO training configuration."""

    learning_rate: float = 1e-5
    ppo_epochs: int = 4
    batch_size: int = 8
    mini_batch_size: int = 4
    gamma: float = 1.0
    lam: float = 0.95
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    num_rollout_steps: int = 128
    num_train_steps: int = 1000
    output_dir: str = "outputs/ppo"


@dataclass
class RLHFConfig:
    """Top-level configuration for the full RLHF pipeline."""

    sft: SFTConfig = field(default_factory=SFTConfig)
    reward: RewardModelConfig = field(default_factory=RewardModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    seed: int = 42
