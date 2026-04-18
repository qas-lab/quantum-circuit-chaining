"""Tests for the reward functions module."""

from __future__ import annotations

import pytest
from benchmarks.ai_transpile.rl_trajectory import (
    RewardConfig,
    RewardSet,
    compute_all_rewards,
    compute_efficiency_reward,
    compute_improvement_only_reward,
    compute_improvement_percentage,
    compute_multi_objective_reward,
    compute_sparse_final_reward,
    get_default_config,
    summarize_trajectory_rewards,
)
from benchmarks.ai_transpile.transpilers import CircuitMetrics

# --- Fixtures ---


@pytest.fixture
def initial_metrics() -> CircuitMetrics:
    """Create initial circuit metrics."""
    return CircuitMetrics(
        depth=20,
        two_qubit_gates=10,
        two_qubit_depth=8,
        total_gates=30,
    )


@pytest.fixture
def improved_metrics() -> CircuitMetrics:
    """Create improved circuit metrics."""
    return CircuitMetrics(
        depth=15,
        two_qubit_gates=6,
        two_qubit_depth=5,
        total_gates=20,
    )


@pytest.fixture
def zero_metrics() -> CircuitMetrics:
    """Create zero metrics (empty circuit)."""
    return CircuitMetrics(
        depth=0,
        two_qubit_gates=0,
        two_qubit_depth=0,
        total_gates=0,
    )


# --- Tests for compute_improvement_only_reward ---


def test_improvement_only_basic(
    initial_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test basic improvement-only reward computation."""
    # (10 - 6) / 10 = 0.4
    reward = compute_improvement_only_reward(initial_metrics, improved_metrics)
    assert reward == pytest.approx(0.4)


def test_improvement_only_with_alpha(
    initial_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test improvement-only reward with custom alpha."""
    config = RewardConfig(alpha=2.0)
    reward = compute_improvement_only_reward(initial_metrics, improved_metrics, config)
    assert reward == pytest.approx(0.8)  # 2.0 * 0.4


def test_improvement_only_zero_initial(
    zero_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test improvement-only with zero initial gates."""
    reward = compute_improvement_only_reward(zero_metrics, improved_metrics)
    assert reward == 0.0


def test_improvement_only_no_improvement(
    initial_metrics: CircuitMetrics,
) -> None:
    """Test improvement-only when no improvement."""
    reward = compute_improvement_only_reward(initial_metrics, initial_metrics)
    assert reward == 0.0


def test_improvement_only_worse_result(
    improved_metrics: CircuitMetrics,
    initial_metrics: CircuitMetrics,
) -> None:
    """Test improvement-only with worse result (negative improvement)."""
    # Going from 6 to 10 gates is -4/6 improvement
    reward = compute_improvement_only_reward(improved_metrics, initial_metrics)
    assert reward == pytest.approx(-4 / 6)


# --- Tests for compute_efficiency_reward ---


def test_efficiency_basic(
    initial_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test basic efficiency reward computation."""
    # R = alpha * improvement - beta * time - gamma
    # R = 1.0 * 0.4 - 0.1 * 1.0 - 0.01 = 0.29
    reward = compute_efficiency_reward(initial_metrics, improved_metrics, time_cost=1.0)
    assert reward == pytest.approx(0.29)


def test_efficiency_with_config(
    initial_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test efficiency reward with custom config."""
    config = RewardConfig(alpha=2.0, beta=0.2, gamma=0.02)
    # R = 2.0 * 0.4 - 0.2 * 1.0 - 0.02 = 0.58
    reward = compute_efficiency_reward(initial_metrics, improved_metrics, time_cost=1.0, config=config)
    assert reward == pytest.approx(0.58)


def test_efficiency_high_time_cost(
    initial_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test efficiency reward with high time cost."""
    # R = 1.0 * 0.4 - 0.1 * 10.0 - 0.01 = -0.61
    reward = compute_efficiency_reward(initial_metrics, improved_metrics, time_cost=10.0)
    assert reward == pytest.approx(-0.61)


def test_efficiency_zero_time(
    initial_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test efficiency reward with zero time cost."""
    # R = 1.0 * 0.4 - 0.1 * 0.0 - 0.01 = 0.39
    reward = compute_efficiency_reward(initial_metrics, improved_metrics, time_cost=0.0)
    assert reward == pytest.approx(0.39)


# --- Tests for compute_multi_objective_reward ---


def test_multi_objective_basic(
    initial_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test basic multi-objective reward."""
    # 2Q improvement: (10-6)/10 = 0.4
    # depth improvement: (20-15)/20 = 0.25
    # combined = 0.7 * 0.4 + 0.3 * 0.25 = 0.355
    # R = 1.0 * 0.355 - 0.1 * 1.0 - 0.01 = 0.245
    reward = compute_multi_objective_reward(initial_metrics, improved_metrics, time_cost=1.0)
    assert reward == pytest.approx(0.245)


def test_multi_objective_with_weights(
    initial_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test multi-objective reward with custom weights."""
    config = RewardConfig(two_qubit_weight=0.5, depth_weight=0.5)
    # combined = 0.5 * 0.4 + 0.5 * 0.25 = 0.325
    # R = 1.0 * 0.325 - 0.1 * 1.0 - 0.01 = 0.215
    reward = compute_multi_objective_reward(initial_metrics, improved_metrics, time_cost=1.0, config=config)
    assert reward == pytest.approx(0.215)


def test_multi_objective_zero_depth(
    zero_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test multi-objective with zero initial depth."""
    reward = compute_multi_objective_reward(zero_metrics, improved_metrics, time_cost=1.0)
    # Both improvements should be 0, so R = -0.1 - 0.01 = -0.11
    assert reward == pytest.approx(-0.11)


# --- Tests for compute_sparse_final_reward ---


def test_sparse_final_not_final(
    initial_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test sparse reward on non-final step."""
    reward = compute_sparse_final_reward(initial_metrics, improved_metrics, is_final_step=False)
    assert reward == 0.0


def test_sparse_final_is_final(
    initial_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test sparse reward on final step."""
    # (10 - 6) / 10 = 0.4
    reward = compute_sparse_final_reward(initial_metrics, improved_metrics, is_final_step=True)
    assert reward == pytest.approx(0.4)


def test_sparse_final_zero_initial(
    zero_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test sparse reward with zero initial."""
    reward = compute_sparse_final_reward(zero_metrics, improved_metrics, is_final_step=True)
    assert reward == 0.0


# --- Tests for compute_all_rewards ---


def test_compute_all_rewards(
    initial_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test computing all reward variants."""
    rewards = compute_all_rewards(
        prev_metrics=initial_metrics,
        new_metrics=improved_metrics,
        time_cost=1.0,
        initial_metrics=initial_metrics,
        is_final_step=True,
    )

    assert isinstance(rewards, RewardSet)
    assert rewards.improvement_only == pytest.approx(0.4)
    assert rewards.efficiency == pytest.approx(0.29)
    assert rewards.sparse_final == pytest.approx(0.4)


def test_compute_all_rewards_not_final(
    initial_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
) -> None:
    """Test computing all rewards on non-final step."""
    rewards = compute_all_rewards(
        prev_metrics=initial_metrics,
        new_metrics=improved_metrics,
        time_cost=1.0,
        is_final_step=False,
    )

    assert rewards.sparse_final == 0.0


# --- Tests for helper functions ---


def test_get_default_config() -> None:
    """Test getting default config."""
    config = get_default_config()

    assert config.alpha == 1.0
    assert config.beta == 0.1
    assert config.gamma == 0.01
    assert config.two_qubit_weight == 0.7
    assert config.depth_weight == 0.3


def test_compute_improvement_percentage() -> None:
    """Test improvement percentage computation."""
    # (10 - 6) / 10 * 100 = 40%
    pct = compute_improvement_percentage(10, 6)
    assert pct == pytest.approx(40.0)


def test_compute_improvement_percentage_zero_initial() -> None:
    """Test improvement percentage with zero initial."""
    pct = compute_improvement_percentage(0, 5)
    assert pct == 0.0


def test_compute_improvement_percentage_no_change() -> None:
    """Test improvement percentage with no change."""
    pct = compute_improvement_percentage(10, 10)
    assert pct == pytest.approx(0.0)


def test_compute_improvement_percentage_worse() -> None:
    """Test improvement percentage when worse."""
    # (6 - 10) / 6 * 100 = -66.67%
    pct = compute_improvement_percentage(6, 10)
    assert pct == pytest.approx(-66.666, rel=0.01)


def test_summarize_trajectory_rewards_empty() -> None:
    """Test summarizing empty rewards list."""
    summary = summarize_trajectory_rewards([])

    assert summary["total_improvement_only"] == 0.0
    assert summary["total_efficiency"] == 0.0
    assert summary["mean_improvement_only"] == 0.0


def test_summarize_trajectory_rewards() -> None:
    """Test summarizing trajectory rewards."""
    rewards = [
        RewardSet(improvement_only=0.2, efficiency=0.15, multi_objective=0.18, sparse_final=0.0),
        RewardSet(improvement_only=0.1, efficiency=0.08, multi_objective=0.09, sparse_final=0.0),
        RewardSet(improvement_only=0.3, efficiency=0.25, multi_objective=0.27, sparse_final=0.6),
    ]

    summary = summarize_trajectory_rewards(rewards)

    assert summary["total_improvement_only"] == pytest.approx(0.6)
    assert summary["total_efficiency"] == pytest.approx(0.48)
    assert summary["total_sparse_final"] == pytest.approx(0.6)  # Last step's sparse
    assert summary["mean_improvement_only"] == pytest.approx(0.2)


# --- Tests for reward config ---


def test_reward_config_defaults() -> None:
    """Test RewardConfig default values."""
    config = RewardConfig()

    assert config.alpha == 1.0
    assert config.beta == 0.1
    assert config.gamma == 0.01
    assert config.depth_weight == 0.3
    assert config.two_qubit_weight == 0.7


def test_reward_config_custom() -> None:
    """Test RewardConfig with custom values."""
    config = RewardConfig(
        alpha=0.5,
        beta=0.2,
        gamma=0.05,
        depth_weight=0.4,
        two_qubit_weight=0.6,
    )

    assert config.alpha == 0.5
    assert config.beta == 0.2
    assert config.gamma == 0.05
    assert config.depth_weight == 0.4
    assert config.two_qubit_weight == 0.6
