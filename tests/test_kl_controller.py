"""Unit tests for the Adaptive KL Controller.

Tests the proportional control law, clamping, history tracking,
and KL explosion detection.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from src.stage3_ppo.kl_controller import AdaptiveKLController


class TestAdaptiveKLController:
    """Tests for AdaptiveKLController."""

    def test_initialization(self) -> None:
        """Controller should initialize with given parameters."""
        ctrl = AdaptiveKLController(
            init_kl_coef=0.3, target_kl=0.1, kl_lr=0.2
        )
        assert ctrl.kl_coef == 0.3
        assert ctrl.target_kl == 0.1
        assert ctrl.kl_lr == 0.2
        assert ctrl._step_count == 0
        assert ctrl._kl_history == []

    def test_update_increases_beta_when_kl_above_target(self) -> None:
        """β should increase when KL > target (penalize divergence)."""
        ctrl = AdaptiveKLController(
            init_kl_coef=0.2, target_kl=0.1, kl_lr=0.1
        )
        initial_beta = ctrl.kl_coef
        # KL = 0.2 > target = 0.1 → should increase β
        new_beta = ctrl.update(current_kl=0.2)
        assert new_beta > initial_beta

    def test_update_decreases_beta_when_kl_below_target(self) -> None:
        """β should decrease when KL < target (allow more exploration)."""
        ctrl = AdaptiveKLController(
            init_kl_coef=0.2, target_kl=0.1, kl_lr=0.1
        )
        initial_beta = ctrl.kl_coef
        # KL = 0.05 < target = 0.1 → should decrease β
        new_beta = ctrl.update(current_kl=0.05)
        assert new_beta < initial_beta

    def test_update_stable_at_target(self) -> None:
        """β should remain unchanged when KL == target."""
        ctrl = AdaptiveKLController(
            init_kl_coef=0.2, target_kl=0.1, kl_lr=0.1
        )
        # KL = target → proportional error = 0 → β unchanged
        new_beta = ctrl.update(current_kl=0.1)
        assert abs(new_beta - 0.2) < 1e-10

    def test_clamping_upper(self) -> None:
        """β should be clamped to 100.0 maximum."""
        ctrl = AdaptiveKLController(
            init_kl_coef=90.0, target_kl=0.001, kl_lr=0.5
        )
        # Very high KL relative to target → try to push β past 100
        ctrl.update(current_kl=10.0)
        assert ctrl.kl_coef <= 100.0

    def test_clamping_lower(self) -> None:
        """β should be clamped to 1e-6 minimum."""
        ctrl = AdaptiveKLController(
            init_kl_coef=1e-5, target_kl=10.0, kl_lr=0.99
        )
        # Very low KL relative to target → try to push β below 1e-6
        ctrl.update(current_kl=0.0001)
        assert ctrl.kl_coef >= 1e-6

    def test_step_count_increments(self) -> None:
        """Step count should increment with each update."""
        ctrl = AdaptiveKLController()
        ctrl.update(0.1)
        ctrl.update(0.2)
        ctrl.update(0.15)
        assert ctrl._step_count == 3

    def test_kl_history_tracking(self) -> None:
        """KL history should record every KL value passed to update."""
        ctrl = AdaptiveKLController()
        values = [0.05, 0.1, 0.08, 0.12]
        for v in values:
            ctrl.update(v)
        assert ctrl.kl_history == values

    def test_kl_history_is_copy(self) -> None:
        """kl_history property should return a copy, not the internal list."""
        ctrl = AdaptiveKLController()
        ctrl.update(0.1)
        history = ctrl.kl_history
        history.append(999.0)
        assert 999.0 not in ctrl.kl_history

    def test_get_stats(self) -> None:
        """get_stats should include all expected fields after updates."""
        ctrl = AdaptiveKLController(init_kl_coef=0.2, target_kl=0.1)
        ctrl.update(0.05)
        ctrl.update(0.15)

        stats = ctrl.get_stats()
        assert "kl_coef" in stats
        assert "target_kl" in stats
        assert "total_steps" in stats
        assert stats["total_steps"] == 2
        assert "mean_kl" in stats
        assert abs(stats["mean_kl"] - 0.1) < 1e-10
        assert stats["max_kl"] == 0.15
        assert stats["min_kl"] == 0.05
        assert stats["last_kl"] == 0.15

    def test_get_stats_empty(self) -> None:
        """get_stats with no updates should not have KL summary keys."""
        ctrl = AdaptiveKLController()
        stats = ctrl.get_stats()
        assert "mean_kl" not in stats
        assert stats["total_steps"] == 0

    def test_is_kl_exploding_true(self) -> None:
        """Should return True when recent KL is far above target."""
        ctrl = AdaptiveKLController(target_kl=0.1)
        ctrl.update(0.6)  # 6× target, default multiplier is 5×
        assert ctrl.is_kl_exploding() is True

    def test_is_kl_exploding_false(self) -> None:
        """Should return False when KL is near target."""
        ctrl = AdaptiveKLController(target_kl=0.1)
        ctrl.update(0.12)
        assert ctrl.is_kl_exploding() is False

    def test_is_kl_exploding_empty_history(self) -> None:
        """Should return False with no KL history."""
        ctrl = AdaptiveKLController(target_kl=0.1)
        assert ctrl.is_kl_exploding() is False

    def test_is_kl_exploding_custom_multiplier(self) -> None:
        """Custom multiplier should change the explosion threshold."""
        ctrl = AdaptiveKLController(target_kl=0.1)
        ctrl.update(0.25)
        # 0.25 < 0.1 * 3.0 = 0.3 → not exploding
        assert ctrl.is_kl_exploding(multiplier=3.0) is False
        # 0.25 > 0.1 * 2.0 = 0.2 → exploding
        assert ctrl.is_kl_exploding(multiplier=2.0) is True

    @patch("src.stage3_ppo.kl_controller.mlflow")
    def test_log_metrics(self, mock_mlflow) -> None:
        """log_metrics should call mlflow.log_metric for key stats."""
        ctrl = AdaptiveKLController(init_kl_coef=0.2, target_kl=0.1)
        ctrl.update(0.08)
        ctrl.log_metrics(step=10)

        calls = mock_mlflow.log_metric.call_args_list
        logged_names = {c[0][0] for c in calls}
        assert "kl_coef_beta" in logged_names
        assert "kl_target" in logged_names
        assert "kl_current" in logged_names

    def test_proportional_control_formula(self) -> None:
        """Verify the exact proportional control math."""
        init_beta = 0.5
        target = 0.1
        lr = 0.2
        kl = 0.15

        ctrl = AdaptiveKLController(
            init_kl_coef=init_beta, target_kl=target, kl_lr=lr
        )
        new_beta = ctrl.update(kl)

        # Expected: β_new = β_old * (1 + kl_lr * (kl / target - 1))
        expected = init_beta * (1.0 + lr * (kl / target - 1.0))
        assert abs(new_beta - expected) < 1e-10
