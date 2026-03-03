"""PPO trainer for reinforcement learning from human feedback."""

from __future__ import annotations

import copy
import os
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from ..config import PPOConfig
from ..models.policy_model import PolicyModel
from ..models.reward_model import RewardModel


class RolloutBatch(NamedTuple):
    """A batch of on-policy rollouts collected for a PPO update."""

    input_ids: torch.Tensor          # (B, T)
    attention_mask: torch.Tensor     # (B, T)
    old_log_probs: torch.Tensor      # (B, T-1) log π_old
    rewards: torch.Tensor            # (B,)  scalar reward per sequence
    advantages: torch.Tensor         # (B, T-1)
    returns: torch.Tensor            # (B, T-1)


class PPOTrainer:
    """Update a policy model using Proximal Policy Optimisation (PPO).

    This implementation uses a *token-level* advantage estimate based on the
    Monte-Carlo return from the scalar end-of-sequence reward.  A reference
    copy of the policy (frozen) is kept to compute a KL-divergence penalty
    that prevents the policy from drifting too far from the SFT initialisation.

    Args:
        policy: The live policy to be updated.
        reward_model: Trained reward model providing scalar feedback.
        config: PPO hyper-parameters.
        kl_coef: Coefficient for the KL penalty term.
    """

    def __init__(
        self,
        policy: PolicyModel,
        reward_model: RewardModel,
        config: PPOConfig,
        kl_coef: float = 0.1,
    ) -> None:
        self.policy = policy
        self.reward_model = reward_model
        self.config = config
        self.kl_coef = kl_coef

        # Frozen reference policy (SFT initialisation)
        self.ref_policy = copy.deepcopy(policy)
        for param in self.ref_policy.model.parameters():
            param.requires_grad_(False)

        self.optimizer = AdamW(
            self.policy.model.parameters(),
            lr=config.learning_rate,
        )

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_rollouts(self, prompts: list[str]) -> RolloutBatch:
        """Generate responses and compute rewards + advantages.

        Args:
            prompts: List of plain-text prompts for the current mini-batch.

        Returns:
            A :class:`RolloutBatch` ready for PPO updates.
        """
        responses = self.policy.generate(
            prompts,
            max_new_tokens=self.config.num_rollout_steps,
        )

        texts = [p + r for p, r in zip(prompts, responses)]
        enc = self.policy.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.policy.device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Compute log-probs under the current policy
        old_log_probs = self.policy.log_probs_of(input_ids, attention_mask)

        # Scalar reward from the reward model
        rewards_raw = self.reward_model(input_ids, attention_mask)  # (B,)

        # KL penalty per sequence (mean over token positions)
        ref_log_probs = self.ref_policy.log_probs_of(input_ids, attention_mask)
        kl_penalty = (old_log_probs - ref_log_probs).mean(dim=1)  # (B,)

        rewards = rewards_raw - self.kl_coef * kl_penalty  # (B,)

        # Broadcast scalar reward to token level and compute advantages
        # (simple Monte-Carlo: same value for every token in the sequence)
        T = old_log_probs.size(1)
        token_rewards = rewards.unsqueeze(1).expand(-1, T)  # (B, T)
        advantages = (token_rewards - token_rewards.mean()) / (
            token_rewards.std() + 1e-8
        )
        returns = token_rewards  # MC return = reward itself

        return RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            old_log_probs=old_log_probs,
            rewards=rewards,
            advantages=advantages,
            returns=returns,
        )

    # ------------------------------------------------------------------
    # PPO update step
    # ------------------------------------------------------------------

    def _ppo_update(self, batch: RolloutBatch) -> dict[str, float]:
        """Perform *ppo_epochs* gradient updates on a rollout batch.

        Args:
            batch: A :class:`RolloutBatch` from :meth:`_collect_rollouts`.

        Returns:
            Dict with ``policy_loss``, ``entropy``, and ``approx_kl`` metrics.
        """
        metrics: dict[str, float] = {
            "policy_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        n_updates = 0

        for _ in range(self.config.ppo_epochs):
            # Compute current log-probs
            outputs = self.policy.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )
            logits = outputs.logits[:, :-1, :]  # (B, T-1, V)
            log_probs_all = F.log_softmax(logits, dim=-1)
            next_token_ids = batch.input_ids[:, 1:].unsqueeze(-1)
            curr_log_probs = log_probs_all.gather(-1, next_token_ids).squeeze(-1)

            # Policy ratio and clipped surrogate objective
            ratio = torch.exp(curr_log_probs - batch.old_log_probs)
            surr1 = ratio * batch.advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
                * batch.advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus (encourage exploration)
            probs = torch.exp(log_probs_all)
            entropy = -(probs * log_probs_all).sum(-1).mean()

            total_loss = policy_loss - self.config.entropy_coef * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.model.parameters(),
                max_norm=self.config.max_grad_norm,
            )
            self.optimizer.step()

            with torch.no_grad():
                approx_kl = (batch.old_log_probs - curr_log_probs).mean().item()

            metrics["policy_loss"] += policy_loss.item()
            metrics["entropy"] += entropy.item()
            metrics["approx_kl"] += approx_kl
            n_updates += 1

        return {k: v / n_updates for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, prompts: list[str]) -> list[dict[str, float]]:
        """Run the PPO training loop.

        Iterates for ``config.num_train_steps`` steps.  On each step a fresh
        batch of *prompts* is sampled (cycling if necessary), responses are
        generated, rewards are computed, and PPO updates are performed.

        Args:
            prompts: Pool of plain-text prompts to draw from.

        Returns:
            List of per-step metric dicts.
        """
        self.policy.model.train()
        self.reward_model.eval()

        all_metrics: list[dict[str, float]] = []
        prompt_pool = prompts

        bar = tqdm(range(self.config.num_train_steps), desc="PPO training")
        for step in bar:
            # Sample a mini-batch of prompts (cycling through the pool)
            start = (step * self.config.batch_size) % len(prompt_pool)
            batch_prompts = (prompt_pool * 2)[start : start + self.config.batch_size]

            rollout = self._collect_rollouts(batch_prompts)
            step_metrics = self._ppo_update(rollout)
            step_metrics["mean_reward"] = rollout.rewards.mean().item()
            all_metrics.append(step_metrics)

            bar.set_postfix(
                reward=f"{step_metrics['mean_reward']:.3f}",
                kl=f"{step_metrics['approx_kl']:.4f}",
            )

        os.makedirs(self.config.output_dir, exist_ok=True)
        self.policy.save_pretrained(self.config.output_dir)
        return all_metrics
