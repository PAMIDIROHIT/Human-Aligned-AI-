"""End-to-end RLHF pipeline orchestrator."""

from __future__ import annotations

from typing import Dict, List, Optional

from .config import RLHFConfig
from .data.dataset import load_preference_dataset, load_sft_dataset
from .models.policy_model import PolicyModel
from .models.reward_model import RewardModel
from .training.ppo_trainer import PPOTrainer
from .training.reward_trainer import RewardTrainer
from .training.sft_trainer import SFTTrainer


class RLHFPipeline:
    """Orchestrates all three stages of the RLHF training pipeline.

    Stages
    ------
    1. **Supervised Fine-Tuning (SFT)** – warm-start the policy on
       human-written demonstrations.
    2. **Reward Model Training** – learn a reward function from human
       preference comparisons.
    3. **PPO Fine-Tuning** – use the reward model as a signal to further
       align the policy via reinforcement learning.

    Args:
        config: A :class:`~rlhf.config.RLHFConfig` with all hyper-parameters.
        policy_model_name: HuggingFace model identifier used for the policy.
        reward_model_name: HuggingFace model identifier used for the reward
                           model.  Defaults to *policy_model_name* if not
                           specified.
        device: Compute device (``"cpu"`` / ``"cuda"``).
    """

    def __init__(
        self,
        config: Optional[RLHFConfig] = None,
        policy_model_name: str = "gpt2",
        reward_model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or RLHFConfig()
        self.policy_model_name = policy_model_name
        self.reward_model_name = reward_model_name or policy_model_name
        self.device = device

        self.policy: Optional[PolicyModel] = None
        self.reward_model: Optional[RewardModel] = None

    # ------------------------------------------------------------------
    # Stage 1: SFT
    # ------------------------------------------------------------------

    def run_sft(self, sft_data: List[Dict[str, str]]) -> List[float]:
        """Run the Supervised Fine-Tuning stage.

        Args:
            sft_data: List of ``{"prompt": …, "response": …}`` dicts.

        Returns:
            Per-step training losses.
        """
        print("=== Stage 1: Supervised Fine-Tuning ===")
        self.policy = PolicyModel(self.policy_model_name, device=self.device)
        dataset = load_sft_dataset(
            sft_data,
            self.policy.tokenizer,
            max_length=self.config.sft.max_seq_length,
        )
        trainer = SFTTrainer(self.policy, self.config.sft)
        losses = trainer.train(dataset)
        print(f"SFT complete. Final loss: {losses[-1]:.4f}")
        return losses

    # ------------------------------------------------------------------
    # Stage 2: Reward Model
    # ------------------------------------------------------------------

    def run_reward_training(
        self, preference_data: List[Dict[str, str]]
    ) -> List[float]:
        """Train the reward model on human preference pairs.

        Args:
            preference_data: List of ``{"prompt": …, "chosen": …,
                             "rejected": …}`` dicts.

        Returns:
            Per-step training losses.
        """
        print("=== Stage 2: Reward Model Training ===")
        self.reward_model = RewardModel(self.reward_model_name, device=self.device)
        tokenizer = self.reward_model.tokenizer
        dataset = load_preference_dataset(
            preference_data,
            tokenizer,
            max_length=self.config.reward.max_seq_length,
        )
        trainer = RewardTrainer(self.reward_model, self.config.reward)
        losses = trainer.train(dataset)
        print(f"Reward model training complete. Final loss: {losses[-1]:.4f}")
        return losses

    # ------------------------------------------------------------------
    # Stage 3: PPO
    # ------------------------------------------------------------------

    def run_ppo(self, prompts: List[str]) -> List[Dict[str, float]]:
        """Run the PPO alignment stage.

        Requires that :meth:`run_sft` and :meth:`run_reward_training` have
        been called first (or that ``self.policy`` / ``self.reward_model``
        have been set manually).

        Args:
            prompts: Pool of plain-text prompts used to generate on-policy
                     rollouts.

        Returns:
            List of per-step metric dicts.
        """
        if self.policy is None:
            raise RuntimeError(
                "Policy model is not initialised. Call run_sft() first or "
                "set self.policy manually."
            )
        if self.reward_model is None:
            raise RuntimeError(
                "Reward model is not initialised. Call run_reward_training() "
                "first or set self.reward_model manually."
            )
        print("=== Stage 3: PPO Fine-Tuning ===")
        trainer = PPOTrainer(self.policy, self.reward_model, self.config.ppo)
        metrics = trainer.train(prompts)
        final_reward = metrics[-1]["mean_reward"] if metrics else float("nan")
        print(f"PPO complete. Final mean reward: {final_reward:.4f}")
        return metrics

    # ------------------------------------------------------------------
    # Convenience: run all stages
    # ------------------------------------------------------------------

    def run(
        self,
        sft_data: List[Dict[str, str]],
        preference_data: List[Dict[str, str]],
        rl_prompts: List[str],
    ) -> Dict:
        """Execute all three pipeline stages end-to-end.

        Args:
            sft_data: Training data for the SFT stage.
            preference_data: Human preference pairs for reward-model training.
            rl_prompts: Prompts for PPO rollouts.

        Returns:
            Dict with keys ``"sft_losses"``, ``"reward_losses"``, and
            ``"ppo_metrics"``.
        """
        sft_losses = self.run_sft(sft_data)
        reward_losses = self.run_reward_training(preference_data)
        ppo_metrics = self.run_ppo(rl_prompts)
        return {
            "sft_losses": sft_losses,
            "reward_losses": reward_losses,
            "ppo_metrics": ppo_metrics,
        }
