"""Tests for the statistics module - TDD Red Phase.

These tests define the expected behavior of the statistics module
for computing metrics across benchmark experiment runs.
"""

from __future__ import annotations

from typing import Any

import pytest
from benchmarks.ai_transpile.statistics import (
    BenchmarkStatistics,
    RunnerComparison,
    aggregate_runner_stats,
    compare_runners,
    compute_confidence_interval,
    compute_improvement_percentage,
)

# --- Test Data Fixtures ---


@pytest.fixture
def sample_results() -> list[dict[str, Any]]:
    """Sample experiment results for testing statistics computation."""
    return [
        # qft_8 circuit with qiskit_ai runner - 3 iterations
        {
            "circuit": "qft_8",
            "runner": "qiskit_ai",
            "label": "ai_level_3_iter_1",
            "metrics": {"depth": 34, "two_qubit_gates": 52, "two_qubit_depth": 53, "total_gates": 60},
        },
        {
            "circuit": "qft_8",
            "runner": "qiskit_ai",
            "label": "ai_level_3_iter_2",
            "metrics": {"depth": 35, "two_qubit_gates": 52, "two_qubit_depth": 52, "total_gates": 60},
        },
        {
            "circuit": "qft_8",
            "runner": "qiskit_ai",
            "label": "ai_level_3_iter_3",
            "metrics": {"depth": 32, "two_qubit_gates": 52, "two_qubit_depth": 52, "total_gates": 60},
        },
        # qft_8 circuit with wisq_rules_only runner - 1 result
        {
            "circuit": "qft_8",
            "runner": "wisq_rules_only",
            "label": "wisq_two_q",
            "metrics": {"depth": 55, "two_qubit_gates": 47, "two_qubit_depth": 35, "total_gates": 99},
        },
        # efficient_su2_12 circuit with qiskit_ai runner - 3 iterations
        {
            "circuit": "efficient_su2_12",
            "runner": "qiskit_ai",
            "label": "ai_level_1_iter_1",
            "metrics": {"depth": 68, "two_qubit_gates": 100, "two_qubit_depth": 113, "total_gates": 148},
        },
        {
            "circuit": "efficient_su2_12",
            "runner": "qiskit_ai",
            "label": "ai_level_1_iter_2",
            "metrics": {"depth": 66, "two_qubit_gates": 91, "two_qubit_depth": 99, "total_gates": 139},
        },
        {
            "circuit": "efficient_su2_12",
            "runner": "qiskit_ai",
            "label": "ai_level_1_iter_3",
            "metrics": {"depth": 58, "two_qubit_gates": 96, "two_qubit_depth": 93, "total_gates": 144},
        },
        # efficient_su2_12 circuit with wisq_rules_only runner
        {
            "circuit": "efficient_su2_12",
            "runner": "wisq_rules_only",
            "label": "wisq_two_q",
            "metrics": {"depth": 38, "two_qubit_gates": 11, "two_qubit_depth": 11, "total_gates": 107},
        },
    ]


# --- Tests for BenchmarkStatistics dataclass ---


class TestBenchmarkStatistics:
    """Tests for the BenchmarkStatistics dataclass."""

    def test_benchmark_statistics_creation(self) -> None:
        """Test creating a BenchmarkStatistics instance."""
        stats = BenchmarkStatistics(
            circuit="qft_8",
            runner="qiskit_ai",
            metric="two_qubit_gates",
            mean=52.0,
            std=0.0,
            min_val=52,
            max_val=52,
            count=3,
        )
        assert stats.circuit == "qft_8"
        assert stats.runner == "qiskit_ai"
        assert stats.metric == "two_qubit_gates"
        assert stats.mean == 52.0
        assert stats.std == 0.0
        assert stats.min_val == 52
        assert stats.max_val == 52
        assert stats.count == 3

    def test_benchmark_statistics_with_confidence_interval(self) -> None:
        """Test BenchmarkStatistics with optional confidence interval."""
        stats = BenchmarkStatistics(
            circuit="qft_8",
            runner="qiskit_ai",
            metric="depth",
            mean=33.67,
            std=1.53,
            min_val=32,
            max_val=35,
            count=3,
            ci_lower=30.05,
            ci_upper=37.29,
        )
        assert stats.ci_lower == 30.05
        assert stats.ci_upper == 37.29


# --- Tests for aggregate_runner_stats ---


class TestAggregateRunnerStats:
    """Tests for the aggregate_runner_stats function."""

    def test_aggregate_single_runner_single_circuit(self, sample_results: list[dict[str, Any]]) -> None:
        """Test aggregation for a single runner on a single circuit."""
        # Filter to just qiskit_ai on qft_8
        qiskit_ai_qft8 = [r for r in sample_results if r["runner"] == "qiskit_ai" and r["circuit"] == "qft_8"]

        stats = aggregate_runner_stats(qiskit_ai_qft8)

        # Should have stats for each metric
        assert len(stats) == 4  # depth, two_qubit_gates, two_qubit_depth, total_gates

        # Find the two_qubit_gates stats
        tqg_stats = next(s for s in stats if s.metric == "two_qubit_gates")
        assert tqg_stats.circuit == "qft_8"
        assert tqg_stats.runner == "qiskit_ai"
        assert tqg_stats.mean == 52.0  # All three are 52
        assert tqg_stats.std == 0.0
        assert tqg_stats.min_val == 52
        assert tqg_stats.max_val == 52
        assert tqg_stats.count == 3

    def test_aggregate_with_variance(self, sample_results: list[dict[str, Any]]) -> None:
        """Test aggregation when there is variance in the metric."""
        qiskit_ai_qft8 = [r for r in sample_results if r["runner"] == "qiskit_ai" and r["circuit"] == "qft_8"]

        stats = aggregate_runner_stats(qiskit_ai_qft8)

        # Find the depth stats (which has variance: 34, 35, 32)
        depth_stats = next(s for s in stats if s.metric == "depth")
        assert depth_stats.mean == pytest.approx(33.67, rel=0.01)
        assert depth_stats.std == pytest.approx(1.53, rel=0.1)
        assert depth_stats.min_val == 32
        assert depth_stats.max_val == 35
        assert depth_stats.count == 3

    def test_aggregate_all_results(self, sample_results: list[dict[str, Any]]) -> None:
        """Test aggregation across all results."""
        stats = aggregate_runner_stats(sample_results)

        # Should have stats for each (circuit, runner, metric) combination
        # 2 circuits × 2 runners × 4 metrics = 16 stats (but each circuit-runner might not have all runners)
        # qft_8: qiskit_ai (3 results), wisq_rules_only (1 result)
        # efficient_su2_12: qiskit_ai (3 results), wisq_rules_only (1 result)
        # So 4 circuit-runner combinations × 4 metrics = 16 stats
        assert len(stats) == 16

    def test_aggregate_empty_results(self) -> None:
        """Test aggregation with empty results."""
        stats = aggregate_runner_stats([])
        assert stats == []

    def test_aggregate_single_result(self) -> None:
        """Test aggregation with a single result (std should be 0)."""
        single_result = [
            {
                "circuit": "test",
                "runner": "test_runner",
                "label": "v1",
                "metrics": {"depth": 10, "two_qubit_gates": 5, "two_qubit_depth": 3, "total_gates": 15},
            }
        ]
        stats = aggregate_runner_stats(single_result)
        assert len(stats) == 4
        for s in stats:
            assert s.count == 1
            assert s.std == 0.0
            assert s.min_val == s.max_val


# --- Tests for compute_confidence_interval ---


class TestComputeConfidenceInterval:
    """Tests for confidence interval computation."""

    def test_confidence_interval_basic(self) -> None:
        """Test basic 95% confidence interval computation."""
        values = [32, 34, 35]  # depth values from sample data
        lower, upper = compute_confidence_interval(values, confidence=0.95)

        # With n=3, t-value for 95% CI (2-tailed, df=2) ≈ 4.303
        # mean = 33.67, std = 1.53, se = 1.53/sqrt(3) ≈ 0.88
        # CI = mean ± t * se = 33.67 ± 4.303 * 0.88 ≈ 33.67 ± 3.79
        assert lower < 33.67 < upper
        assert lower == pytest.approx(29.88, rel=0.15)
        assert upper == pytest.approx(37.46, rel=0.15)

    def test_confidence_interval_single_value(self) -> None:
        """Test CI with single value returns the value for both bounds."""
        values = [42]
        lower, upper = compute_confidence_interval(values, confidence=0.95)
        assert lower == 42
        assert upper == 42

    def test_confidence_interval_identical_values(self) -> None:
        """Test CI with identical values returns those values."""
        values = [52, 52, 52]
        lower, upper = compute_confidence_interval(values, confidence=0.95)
        assert lower == 52
        assert upper == 52

    def test_confidence_interval_empty_raises(self) -> None:
        """Test that empty values raises ValueError."""
        with pytest.raises(ValueError, match="at least one value"):
            compute_confidence_interval([], confidence=0.95)


# --- Tests for compute_improvement_percentage ---


class TestComputeImprovementPercentage:
    """Tests for improvement percentage computation."""

    def test_improvement_percentage_positive(self) -> None:
        """Test improvement when optimized is better (lower)."""
        # WISQ achieves 47 vs Qiskit AI's 52 two_qubit_gates
        improvement = compute_improvement_percentage(baseline=52, optimized=47)
        # (52 - 47) / 52 * 100 = 9.62%
        assert improvement == pytest.approx(9.62, rel=0.01)

    def test_improvement_percentage_large(self) -> None:
        """Test large improvement percentage."""
        # WISQ achieves 11 vs Qiskit AI's 91 two_qubit_gates on efficient_su2_12
        improvement = compute_improvement_percentage(baseline=91, optimized=11)
        # (91 - 11) / 91 * 100 = 87.91%
        assert improvement == pytest.approx(87.91, rel=0.01)

    def test_improvement_percentage_negative(self) -> None:
        """Test negative improvement (regression)."""
        improvement = compute_improvement_percentage(baseline=50, optimized=60)
        # (50 - 60) / 50 * 100 = -20%
        assert improvement == pytest.approx(-20.0, rel=0.01)

    def test_improvement_percentage_zero(self) -> None:
        """Test zero improvement (no change)."""
        improvement = compute_improvement_percentage(baseline=52, optimized=52)
        assert improvement == 0.0

    def test_improvement_percentage_zero_baseline_raises(self) -> None:
        """Test that zero baseline raises ValueError."""
        with pytest.raises(ValueError, match="baseline.*zero"):
            compute_improvement_percentage(baseline=0, optimized=10)


# --- Tests for RunnerComparison dataclass ---


class TestRunnerComparison:
    """Tests for the RunnerComparison dataclass."""

    def test_runner_comparison_creation(self) -> None:
        """Test creating a RunnerComparison instance."""
        comparison = RunnerComparison(
            circuit="qft_8",
            metric="two_qubit_gates",
            baseline_runner="qiskit_ai",
            optimized_runner="wisq_rules_only",
            baseline_mean=52.0,
            optimized_mean=47.0,
            improvement_pct=9.62,
            baseline_std=0.0,
            optimized_std=0.0,
        )
        assert comparison.circuit == "qft_8"
        assert comparison.improvement_pct == 9.62


# --- Tests for compare_runners ---


class TestCompareRunners:
    """Tests for the compare_runners function."""

    def test_compare_two_runners(self, sample_results: list[dict[str, Any]]) -> None:
        """Test comparison between two runners."""
        comparisons = compare_runners(
            sample_results,
            baseline_runner="qiskit_ai",
            optimized_runner="wisq_rules_only",
        )

        # Should have comparison for each (circuit, metric) combination
        # 2 circuits × 4 metrics = 8 comparisons
        assert len(comparisons) == 8

        # Find the qft_8 two_qubit_gates comparison
        qft8_tqg = next(
            c for c in comparisons if c.circuit == "qft_8" and c.metric == "two_qubit_gates"
        )
        assert qft8_tqg.baseline_runner == "qiskit_ai"
        assert qft8_tqg.optimized_runner == "wisq_rules_only"
        assert qft8_tqg.baseline_mean == 52.0
        assert qft8_tqg.optimized_mean == 47.0
        assert qft8_tqg.improvement_pct == pytest.approx(9.62, rel=0.01)

    def test_compare_runners_large_improvement(self, sample_results: list[dict[str, Any]]) -> None:
        """Test comparison with large improvement."""
        comparisons = compare_runners(
            sample_results,
            baseline_runner="qiskit_ai",
            optimized_runner="wisq_rules_only",
        )

        # Find the efficient_su2_12 two_qubit_gates comparison
        su2_tqg = next(
            c for c in comparisons if c.circuit == "efficient_su2_12" and c.metric == "two_qubit_gates"
        )
        # WISQ: 11, Qiskit AI mean: (100+91+96)/3 = 95.67
        assert su2_tqg.optimized_mean == 11.0
        assert su2_tqg.baseline_mean == pytest.approx(95.67, rel=0.01)
        # Improvement: (95.67 - 11) / 95.67 * 100 ≈ 88.5%
        assert su2_tqg.improvement_pct == pytest.approx(88.5, rel=0.01)

    def test_compare_runners_missing_runner(self, sample_results: list[dict[str, Any]]) -> None:
        """Test comparison when one runner is missing for a circuit."""
        # Add a result for a circuit that only has one runner
        results_with_missing = sample_results + [
            {
                "circuit": "grover_4",
                "runner": "qiskit_ai",
                "label": "v1",
                "metrics": {"depth": 20, "two_qubit_gates": 15, "two_qubit_depth": 10, "total_gates": 30},
            }
        ]

        comparisons = compare_runners(
            results_with_missing,
            baseline_runner="qiskit_ai",
            optimized_runner="wisq_rules_only",
        )

        # grover_4 should not have comparisons (no wisq results)
        grover_comparisons = [c for c in comparisons if c.circuit == "grover_4"]
        assert len(grover_comparisons) == 0

    def test_compare_runners_specific_metrics(self, sample_results: list[dict[str, Any]]) -> None:
        """Test comparison with specific metrics filter."""
        comparisons = compare_runners(
            sample_results,
            baseline_runner="qiskit_ai",
            optimized_runner="wisq_rules_only",
            metrics=["two_qubit_gates", "depth"],
        )

        # Should have 2 circuits × 2 metrics = 4 comparisons
        assert len(comparisons) == 4
        metrics_in_results = {c.metric for c in comparisons}
        assert metrics_in_results == {"two_qubit_gates", "depth"}


# ============================================================================
# Tests for Chain Statistics
# ============================================================================


class TestChainStatistics:
    """Test chain experiment statistics functions."""

    @pytest.fixture
    def sample_chain_result_dict(self) -> dict[str, Any]:
        """Sample chain result dictionary for testing."""
        return {
            "chain_name": "wisq_then_tket",
            "steps": [
                {"runner_type": "wisq", "options": {"approx_epsilon": 0}, "name": "wisq_rules"},
                {"runner_type": "tket", "options": {"gate_set": "IBMN"}, "name": "tket_peephole"},
            ],
            "initial_metrics": {
                "depth": 20,
                "two_qubit_gates": 50,
                "two_qubit_depth": 15,
                "total_gates": 80,
            },
            "final_metrics": {
                "depth": 15,
                "two_qubit_gates": 35,
                "two_qubit_depth": 12,
                "total_gates": 60,
            },
            "step_results": [
                {
                    "step_name": "wisq_rules",
                    "step_index": 0,
                    "input_metrics": {
                        "depth": 20,
                        "two_qubit_gates": 50,
                        "two_qubit_depth": 15,
                        "total_gates": 80,
                    },
                    "output_metrics": {
                        "depth": 18,
                        "two_qubit_gates": 40,
                        "two_qubit_depth": 14,
                        "total_gates": 70,
                    },
                    "duration_seconds": 2.5,
                    "artifact_path": "/tmp/step0.qasm",
                },
                {
                    "step_name": "tket_peephole",
                    "step_index": 1,
                    "input_metrics": {
                        "depth": 18,
                        "two_qubit_gates": 40,
                        "two_qubit_depth": 14,
                        "total_gates": 70,
                    },
                    "output_metrics": {
                        "depth": 15,
                        "two_qubit_gates": 35,
                        "two_qubit_depth": 12,
                        "total_gates": 60,
                    },
                    "duration_seconds": 1.5,
                    "artifact_path": "/tmp/step1.qasm",
                },
            ],
            "total_duration_seconds": 4.0,
            "metadata": {"output_dir": "/tmp/chain"},
        }

    @pytest.fixture
    def sample_chain_benchmark_results(self) -> list[dict[str, Any]]:
        """Sample benchmark results including chain runner."""
        return [
            {
                "circuit": "qft_8",
                "runner": "wisq_then_tket",
                "optimizer": "chain",
                "label": "chain_wisq_rules_then_tket_peephole",
                "metrics": {
                    "depth": 15,
                    "two_qubit_gates": 35,
                    "two_qubit_depth": 12,
                    "total_gates": 60,
                },
                "metadata": {
                    "chain_name": "wisq_then_tket",
                    "total_duration_seconds": 4.0,
                },
            },
        ]

    @pytest.fixture
    def sample_individual_results(self) -> list[dict[str, Any]]:
        """Sample individual optimizer results."""
        return [
            {
                "circuit": "qft_8",
                "runner": "wisq_rules_only",
                "optimizer": "wisq",
                "label": "wisq_two_q",
                "metrics": {
                    "depth": 18,
                    "two_qubit_gates": 40,
                    "two_qubit_depth": 14,
                    "total_gates": 70,
                },
                "metadata": {"duration_seconds": 2.5},
            },
            {
                "circuit": "qft_8",
                "runner": "tket_full_peephole",
                "optimizer": "tket",
                "label": "tket_ibmn",
                "metrics": {
                    "depth": 17,
                    "two_qubit_gates": 42,
                    "two_qubit_depth": 13,
                    "total_gates": 65,
                },
                "metadata": {"duration_seconds": 1.0},
            },
            {
                "circuit": "qft_8",
                "runner": "qiskit_standard",
                "optimizer": "qiskit_standard",
                "label": "qiskit_opt_level_3",
                "metrics": {
                    "depth": 22,
                    "two_qubit_gates": 52,
                    "two_qubit_depth": 18,
                    "total_gates": 85,
                },
                "metadata": {"duration_seconds": 0.5},
            },
        ]

    def test_analyze_chain_results(self, sample_chain_result_dict: dict[str, Any]) -> None:
        """Test analyzing chain result dictionary."""
        from benchmarks.ai_transpile.statistics import analyze_chain_results

        analysis = analyze_chain_results(sample_chain_result_dict)

        assert analysis["chain_name"] == "wisq_then_tket"
        assert analysis["num_steps"] == 2
        assert len(analysis["step_statistics"]) == 2

        # Check first step statistics
        step0 = analysis["step_statistics"][0]
        assert step0["step_name"] == "wisq_rules"
        assert step0["input_two_qubit_gates"] == 50
        assert step0["output_two_qubit_gates"] == 40
        assert step0["improvement_pct"] == pytest.approx(20.0)  # (50-40)/50 * 100
        assert step0["duration_seconds"] == 2.5

        # Check second step statistics
        step1 = analysis["step_statistics"][1]
        assert step1["step_name"] == "tket_peephole"
        assert step1["input_two_qubit_gates"] == 40
        assert step1["output_two_qubit_gates"] == 35
        assert step1["improvement_pct"] == pytest.approx(12.5)  # (40-35)/40 * 100

        # Check total improvement
        # Initial: 50, Final: 35 -> (50-35)/50 * 100 = 30%
        assert analysis["total_improvement_pct"] == pytest.approx(30.0)
        assert analysis["total_duration_seconds"] == 4.0

    def test_compare_chain_vs_individual(
        self,
        sample_chain_benchmark_results: list[dict[str, Any]],
        sample_individual_results: list[dict[str, Any]],
    ) -> None:
        """Test comparing chain results against individual optimizers."""
        from benchmarks.ai_transpile.statistics import compare_chain_vs_individual

        comparison = compare_chain_vs_individual(
            chain_results=sample_chain_benchmark_results,
            individual_results=sample_individual_results,
            circuit_name="qft_8",
            metric="two_qubit_gates",
        )

        assert comparison is not None
        assert comparison.circuit == "qft_8"
        assert comparison.chain_name == "wisq_then_tket"
        assert comparison.chain_final_metric == 35

        # Check individual results were collected
        assert "wisq_rules_only" in comparison.individual_results
        assert "tket_full_peephole" in comparison.individual_results
        assert "qiskit_standard" in comparison.individual_results

        # Best individual should be wisq_rules_only with 40 gates
        assert comparison.best_individual == "wisq_rules_only"
        assert comparison.best_individual_metric == 40

        # Chain improvement over best: (40-35)/40 * 100 = 12.5%
        assert comparison.chain_improvement_over_best == pytest.approx(12.5)

    def test_compare_chain_vs_individual_no_chain(
        self, sample_individual_results: list[dict[str, Any]]
    ) -> None:
        """Test comparison returns None when no chain result found."""
        from benchmarks.ai_transpile.statistics import compare_chain_vs_individual

        comparison = compare_chain_vs_individual(
            chain_results=[],
            individual_results=sample_individual_results,
            circuit_name="qft_8",
        )

        assert comparison is None

    def test_compare_chain_vs_individual_no_individuals(
        self, sample_chain_benchmark_results: list[dict[str, Any]]
    ) -> None:
        """Test comparison returns None when no individual results found."""
        from benchmarks.ai_transpile.statistics import compare_chain_vs_individual

        comparison = compare_chain_vs_individual(
            chain_results=sample_chain_benchmark_results,
            individual_results=[],
            circuit_name="qft_8",
        )

        assert comparison is None

    def test_compute_chain_efficiency(self) -> None:
        """Test computing chain efficiency metrics."""
        from benchmarks.ai_transpile.statistics import compute_chain_efficiency

        efficiency = compute_chain_efficiency(
            chain_duration=4.0,
            chain_improvement=30.0,
            individual_durations={
                "wisq_rules_only": 2.5,
                "tket_full_peephole": 1.0,
                "qiskit_standard": 0.5,
            },
            individual_improvements={
                "wisq_rules_only": 20.0,
                "tket_full_peephole": 16.0,
                "qiskit_standard": 0.0,
            },
        )

        # Chain efficiency: 30.0 / 4.0 = 7.5 improvement per second
        assert efficiency["chain_efficiency"] == pytest.approx(7.5)

        # Individual efficiencies
        assert efficiency["individual_efficiencies"]["wisq_rules_only"] == pytest.approx(8.0)  # 20/2.5
        assert efficiency["individual_efficiencies"]["tket_full_peephole"] == pytest.approx(16.0)  # 16/1
        assert efficiency["individual_efficiencies"]["qiskit_standard"] == pytest.approx(0.0)

        # Best individual efficiency is TKET with 16.0
        assert efficiency["best_efficient_runner"] == "tket_full_peephole"
        assert efficiency["best_individual_efficiency"] == pytest.approx(16.0)

        # Chain vs best ratio: 7.5 / 16.0 = 0.46875
        assert efficiency["chain_vs_best_efficiency_ratio"] == pytest.approx(0.46875)

    def test_compute_chain_efficiency_zero_duration(self) -> None:
        """Test chain efficiency handles zero duration gracefully."""
        from benchmarks.ai_transpile.statistics import compute_chain_efficiency

        efficiency = compute_chain_efficiency(
            chain_duration=0.0,
            chain_improvement=30.0,
            individual_durations={"optimizer": 0.0},
            individual_improvements={"optimizer": 10.0},
        )

        assert efficiency["chain_efficiency"] == 0.0
        assert efficiency["individual_efficiencies"]["optimizer"] == 0.0

