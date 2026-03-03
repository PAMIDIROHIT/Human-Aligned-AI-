# Human-Aligned-AI

An **end-to-end Reinforcement Learning from Human Feedback (RLHF) pipeline** for training human-aligned language models.

---

## Overview

This repository implements the three-stage RLHF pipeline introduced in *InstructGPT* and widely used to align large language models with human preferences:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `SFTTrainer` | **Supervised Fine-Tuning** – warm-start a causal LM on human-written demonstrations |
| 2 | `RewardTrainer` | **Reward Model Training** – learn a scalar reward function from preference comparisons (Bradley-Terry loss) |
| 3 | `PPOTrainer` | **PPO Alignment** – optimise the policy against the reward model while constraining KL divergence from the SFT baseline |

---

## Project Structure

```
rlhf/
├── config.py          # Dataclass configurations for all three stages
├── pipeline.py        # RLHFPipeline – end-to-end orchestrator
├── data/
│   └── dataset.py     # PromptDataset, PreferenceDataset, loader utilities
├── models/
│   ├── policy_model.py   # PolicyModel (causal LM wrapper)
│   └── reward_model.py   # RewardModel (encoder + scalar value head)
└── training/
    ├── sft_trainer.py     # Supervised Fine-Tuning loop
    ├── reward_trainer.py  # Reward model training loop
    └── ppo_trainer.py     # PPO training loop with KL penalty

scripts/
├── train_sft.py       # CLI: run SFT stage
├── train_reward.py    # CLI: run reward-model stage
└── train_ppo.py       # CLI: run PPO stage

tests/
├── test_data.py       # Dataset and loader tests
├── test_models.py     # PolicyModel and RewardModel unit tests
└── test_training.py   # Trainer integration tests
```

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### End-to-End Pipeline

```python
from rlhf import RLHFPipeline, RLHFConfig

config = RLHFConfig()  # uses defaults; override as needed

pipeline = RLHFPipeline(
    config=config,
    policy_model_name="gpt2",
    reward_model_name="gpt2",
)

# Stage 1 – SFT
sft_data = [
    {"prompt": "What is the capital of France?", "response": "Paris."},
    # ... more examples
]

# Stage 2 – Reward model
preference_data = [
    {
        "prompt": "Tell me a joke.",
        "chosen": "Why did the chicken cross the road? To get to the other side!",
        "rejected": "I don't know.",
    },
    # ... more pairs
]

# Stage 3 – PPO
rl_prompts = ["Tell me about AI.", "What is machine learning?"]

results = pipeline.run(sft_data, preference_data, rl_prompts)
```

### Individual Stages via CLI

```bash
# Stage 1: SFT
python scripts/train_sft.py \
    --model_name gpt2 \
    --data_path data/sft_data.json \
    --output_dir outputs/sft

# Stage 2: Reward model
python scripts/train_reward.py \
    --model_name gpt2 \
    --data_path data/preference_data.json \
    --output_dir outputs/reward_model

# Stage 3: PPO
python scripts/train_ppo.py \
    --policy_path outputs/sft \
    --reward_path outputs/reward_model \
    --prompts_path data/prompts.json \
    --output_dir outputs/ppo
```

### Data Formats

**SFT data** (`data/sft_data.json`):
```json
[
  {"prompt": "Q: What is AI?  A:", "response": " Artificial Intelligence."},
  ...
]
```

**Preference data** (`data/preference_data.json`):
```json
[
  {
    "prompt": "Tell me a joke.",
    "chosen": "Why did the chicken cross the road? To get to the other side!",
    "rejected": "Ugh, jokes are boring."
  },
  ...
]
```

**Prompts** (`data/prompts.json`):
```json
["What is machine learning?", "Explain neural networks.", ...]
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Configuration

All hyper-parameters are exposed via dataclasses in `rlhf/config.py`:

```python
from rlhf.config import RLHFConfig, SFTConfig, RewardModelConfig, PPOConfig

config = RLHFConfig(
    sft=SFTConfig(
        model_name="gpt2",
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        max_seq_length=512,
        output_dir="outputs/sft",
    ),
    reward=RewardModelConfig(
        learning_rate=1e-5,
        num_train_epochs=3,
        output_dir="outputs/reward_model",
    ),
    ppo=PPOConfig(
        learning_rate=1e-5,
        ppo_epochs=4,
        clip_range=0.2,
        kl_coef=0.1,        # set via PPOTrainer
        num_train_steps=1000,
        output_dir="outputs/ppo",
    ),
)
```
