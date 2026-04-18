"""Tests for rl_training evaluation and baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from benchmarks.ai_transpile.rl_training.algorithms.behavioral_cloning import BehavioralCloning
from benchmarks.ai_transpile.rl_training.config import TrainingConfig
from benchmarks.ai_transpile.rl_training.dataset import OfflineRLDataset
from benchmarks.ai_transpile.rl_training.evaluation import (
    compute_baselines,
    evaluate_policy,
    generate_comparison_table,
    save_evaluation_results,
)
from benchmarks.ai_transpile.rl_training.normalization import NormalizationStats


def _make_dataset(n: int = 50, action_dim: int = 5) -> OfflineRLDataset:
    """Create a synthetic dataset for evaluation testing."""
    rng = np.random.RandomState(42)
    return OfflineRLDataset(
        observations=rng.randn(n, 26).astype(np.float32),
        actions=rng.randint(0, action_dim, n).astype(np.int64),
        rewards=rng.randn(n).astype(np.float32),
        next_observations=rng.randn(n, 26).astype(np.float32),
        terminals=np.ones(n, dtype=np.float32),
        norm_stats=NormalizationStats(
            means=np.zeros(26, dtype=np.float32),
            stds=np.ones(26, dtype=np.float32),
            count=n,
        ),
        action_map={i + 1: i for i in range(action_dim)},
        action_names=[f"opt_{i}" for i in range(action_dim)],
    )


class TestEvaluatePolicy:
    def test_evaluate_returns_metrics(self) -> None:
        """evaluate_policy should return expected metric keys."""
        config = TrainingConfig(algorithm="bc", device="cpu")
        trainer = BehavioralCloning(config)
        dataset = _make_dataset()

        metrics = evaluate_policy(trainer, dataset)
        assert "action_agreement" in metrics
        assert "policy_entropy" in metrics
        assert "num_samples" in metrics
        assert 0 <= metrics["action_agreement"] <= 1

    def test_action_agreement_range(self) -> None:
        """Action agreement should be between 0 and 1."""
        config = TrainingConfig(algorithm="bc", device="cpu")
        trainer = BehavioralCloning(config)
        dataset = _make_dataset()

        metrics = evaluate_policy(trainer, dataset)
        assert 0 <= metrics["action_agreement"] <= 1

    def test_policy_entropy_non_negative(self) -> None:
        """Policy entropy should be non-negative."""
        config = TrainingConfig(algorithm="bc", device="cpu")
        trainer = BehavioralCloning(config)
        dataset = _make_dataset()

        metrics = evaluate_policy(trainer, dataset)
        assert metrics["policy_entropy"] >= 0


class TestBaselines:
    def test_compute_baselines(self) -> None:
        """Baselines should include random, greedy, and best_single."""
        dataset = _make_dataset()
        baselines = compute_baselines(dataset)

        assert "random" in baselines
        assert "greedy_most_common" in baselines
        assert "best_single" in baselines

    def test_random_baseline_agreement(self) -> None:
        """Random baseline agreement should be near 1/action_dim."""
        dataset = _make_dataset(n=1000)
        baselines = compute_baselines(dataset)

        random_agreement = baselines["random"]["action_agreement"]
        expected = baselines["random"]["expected_agreement"]
        # Should be within 5% of expected
        assert abs(random_agreement - expected) < 0.05

    def test_greedy_baseline(self) -> None:
        """Greedy baseline should pick the most common action."""
        dataset = _make_dataset()
        baselines = compute_baselines(dataset)

        greedy = baselines["greedy_most_common"]
        assert "chosen_action" in greedy
        assert "action_agreement" in greedy
        # Greedy should have higher agreement than random (usually)
        assert greedy["action_agreement"] >= baselines["random"]["expected_agreement"]


class TestComparisonTable:
    def test_generate_table(self) -> None:
        """Comparison table should be a non-empty string."""
        policy_metrics = {
            "action_agreement": 0.65,
            "policy_entropy": 1.2,
            "num_samples": 100,
            "policy_freq_action_0": 0.3,
            "policy_freq_action_1": 0.2,
            "policy_freq_action_2": 0.2,
            "policy_freq_action_3": 0.15,
            "policy_freq_action_4": 0.15,
        }
        baselines = {
            "random": {"action_agreement": 0.2, "data_freq_action_0": 0.25},
            "greedy_most_common": {"action_agreement": 0.35, "data_freq_action_0": 0.25},
        }
        action_names = ["wisq_rules", "wisq_bqskit", "tket", "qiskit_ai", "qiskit_standard"]

        table = generate_comparison_table(policy_metrics, baselines, action_names)
        assert "Action Agreement" in table
        assert "0.65" in table


class TestSaveResults:
    def test_save_results(self, tmp_path: Path) -> None:
        """Results should be saved to JSON."""
        results = {"accuracy": 0.75, "loss": 0.5}
        path = tmp_path / "results.json"
        save_evaluation_results(results, path)
        assert path.exists()

        import json
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["accuracy"] == 0.75
