"""Stage 3 PPO — Adaptive KL Penalty Controller (PID-style).

Adjusts the KL penalty coefficient β dynamically during PPO training
based on the current KL divergence relative to the target.

The controller uses a proportional feedback mechanism:
  - If KL > target: increase β to penalize divergence
  - If KL < target: decrease β to allow more exploration
"""

from __future__ import annotations

import logging
import math

import mlflow

logger = logging.getLogger(__name__)


class AdaptiveKLController:
    """PID-style adaptive KL penalty controller for PPO.

    Adjusts β (the KL penalty coefficient) to keep KL divergence
    near the target value. Uses a proportional control law:
        β_new = β_old * (1 + kl_lr * (kl_current / kl_target - 1))

    Attributes:
        kl_coef: Current KL penalty coefficient (β).
        target_kl: Target KL divergence.
        horizon: Smoothing horizon for the controller.
        kl_lr: Learning rate for adjusting the KL coefficient.
        _step_count: Number of update steps taken.
        _kl_history: History of KL values for monitoring.
    """

    def __init__(
        self,
        init_kl_coef: float = 0.2,
        target_kl: float = 0.1,
        horizon: int = 10000,
        kl_lr: float = 0.1,
    ) -> None:
        """Initialize the adaptive KL controller.

        Args:
            init_kl_coef: Initial value of β (KL penalty coefficient).
            target_kl: Target KL divergence to maintain.
            horizon: Smoothing horizon for proportional control.
            kl_lr: Learning rate (sensitivity) of the controller.
        """
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon
        self.kl_lr = kl_lr
        self._step_count = 0
        self._kl_history: list[float] = []

        logger.info(
            "Initialized AdaptiveKLController: β=%.4f, target_kl=%.4f, kl_lr=%.4f",
            init_kl_coef,
            target_kl,
            kl_lr,
        )

    def update(self, current_kl: float, step: int | None = None) -> float:
        """Update the KL coefficient based on current KL divergence.

        Uses proportional control:
            β_new = β_old * (1 + kl_lr * (kl_current / kl_target - 1))

        The coefficient is clamped to [1e-6, 100] to prevent instability.

        Args:
            current_kl: The current measured KL divergence.
            step: Optional step number for logging (defaults to internal counter).

        Returns:
            The updated KL coefficient β.
        """
        if step is None:
            step = self._step_count

        # Proportional control
        proportional_error = current_kl / max(self.target_kl, 1e-8) - 1.0
        self.kl_coef = self.kl_coef * (1.0 + self.kl_lr * proportional_error)

        # Clamp to prevent extreme values
        self.kl_coef = max(1e-6, min(100.0, self.kl_coef))

        self._kl_history.append(current_kl)
        self._step_count += 1

        logger.debug(
            "KL Controller step %d: kl=%.6f, β=%.6f, error=%.4f",
            step,
            current_kl,
            self.kl_coef,
            proportional_error,
        )

        return self.kl_coef

    def log_metrics(self, step: int) -> None:
        """Log KL controller metrics to MLflow.

        Args:
            step: The global training step.
        """
        mlflow.log_metric("kl_coef_beta", self.kl_coef, step=step)
        mlflow.log_metric("kl_target", self.target_kl, step=step)
        if self._kl_history:
            mlflow.log_metric("kl_current", self._kl_history[-1], step=step)

    @property
    def kl_history(self) -> list[float]:
        """Get the history of KL values.

        Returns:
            List of recorded KL divergence values.
        """
        return self._kl_history.copy()

    def get_stats(self) -> dict[str, float]:
        """Get summary statistics of the KL controller.

        Returns:
            Dictionary with controller statistics including current beta,
            mean KL, and step count.
        """
        stats = {
            "kl_coef": self.kl_coef,
            "target_kl": self.target_kl,
            "total_steps": self._step_count,
        }
        if self._kl_history:
            stats["mean_kl"] = sum(self._kl_history) / len(self._kl_history)
            stats["max_kl"] = max(self._kl_history)
            stats["min_kl"] = min(self._kl_history)
            stats["last_kl"] = self._kl_history[-1]
        return stats

    def is_kl_exploding(self, multiplier: float = 5.0) -> bool:
        """Check if KL divergence is exploding (reward hacking indicator).

        Args:
            multiplier: Factor above target_kl considered dangerous.

        Returns:
            True if recent KL is significantly above target.
        """
        if not self._kl_history:
            return False
        recent_kl = self._kl_history[-1]
        return recent_kl > self.target_kl * multiplier
