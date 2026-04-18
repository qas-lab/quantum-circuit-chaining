"""Tests for the visualization module."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from benchmarks.ai_transpile.statistics import BenchmarkStatistics, RunnerComparison
from benchmarks.ai_transpile.visualization import (
    create_comparison_table,
    create_summary_table,
    plot_comparison_heatmap,
    plot_improvement_bars,
    plot_pareto_frontier,
    plot_runtime_vs_improvement,
    plot_runtime_vs_improvement_scatter,
    plot_variance_boxplot,
    plot_variance_boxplot_raw,
    setup_matplotlib_style,
)
from matplotlib.figure import Figure

# --- Fixtures ---


@pytest.fixture(autouse=True)
def close_figures() -> None:  # type: ignore[misc]
    """Close all matplotlib figures after each test."""
    yield  # type: ignore[misc]
    plt.close("all")


@pytest.fixture
def sample_stats() -> list[BenchmarkStatistics]:
    """Create sample BenchmarkStatistics list."""
    return [
        BenchmarkStatistics(
            circuit="qft_4",
            runner="wisq_rules",
            metric="two_qubit_gates",
            mean=8.0,
            std=1.0,
            min_val=6.0,
            max_val=10.0,
            count=3,
        ),
        BenchmarkStatistics(
            circuit="qaoa_4",
            runner="tket",
            metric="two_qubit_gates",
            mean=12.0,
            std=2.0,
            min_val=9.0,
            max_val=14.0,
            count=3,
        ),
    ]


@pytest.fixture
def sample_comparisons() -> list[RunnerComparison]:
    """Create sample RunnerComparison list."""
    return [
        RunnerComparison(
            circuit="qft_4",
            metric="two_qubit_gates",
            baseline_runner="qiskit_ai",
            optimized_runner="wisq_rules",
            baseline_mean=10.0,
            optimized_mean=8.0,
            improvement_pct=20.0,
            baseline_std=1.0,
            optimized_std=0.5,
        ),
    ]


# --- Tests for setup_matplotlib_style ---


def test_setup_matplotlib_style() -> None:
    """Test that setup_matplotlib_style sets expected rcParams."""
    setup_matplotlib_style()
    assert plt.rcParams["figure.dpi"] == 300


# --- Tests for plot_variance_boxplot ---


def test_plot_variance_boxplot(sample_stats: list[BenchmarkStatistics]) -> None:
    """Test plot_variance_boxplot returns a Figure."""
    fig = plot_variance_boxplot(sample_stats, metric="two_qubit_gates")
    assert isinstance(fig, Figure)


def test_plot_variance_boxplot_bad_metric(sample_stats: list[BenchmarkStatistics]) -> None:
    """Test plot_variance_boxplot raises ValueError for unknown metric."""
    with pytest.raises(ValueError, match="No data found"):
        plot_variance_boxplot(sample_stats, metric="nonexistent_metric")


# --- Tests for plot_variance_boxplot_raw ---


def test_plot_variance_boxplot_raw() -> None:
    """Test plot_variance_boxplot_raw returns a Figure."""
    raw_results = [
        {"metrics": {"depth": 10}, "metadata": {"optimization_level": 1}},
        {"metrics": {"depth": 12}, "metadata": {"optimization_level": 1}},
        {"metrics": {"depth": 8}, "metadata": {"optimization_level": 2}},
        {"metrics": {"depth": 9}, "metadata": {"optimization_level": 2}},
    ]
    fig = plot_variance_boxplot_raw(raw_results, metric="depth")
    assert isinstance(fig, Figure)


# --- Tests for plot_improvement_bars ---


def test_plot_improvement_bars(sample_comparisons: list[RunnerComparison]) -> None:
    """Test plot_improvement_bars returns a Figure."""
    fig = plot_improvement_bars(sample_comparisons, metric="two_qubit_gates")
    assert isinstance(fig, Figure)


def test_plot_improvement_bars_bad_metric(
    sample_comparisons: list[RunnerComparison],
) -> None:
    """Test plot_improvement_bars raises ValueError for unknown metric."""
    with pytest.raises(ValueError, match="No data found"):
        plot_improvement_bars(sample_comparisons, metric="nonexistent_metric")


# --- Tests for plot_runtime_vs_improvement ---


def test_plot_runtime_vs_improvement() -> None:
    """Test plot_runtime_vs_improvement returns a Figure."""
    data = [
        {"duration_seconds": 1.0, "improvement_pct": 10.0, "label": "run1"},
        {"duration_seconds": 2.0, "improvement_pct": -5.0, "label": "run2"},
    ]
    fig = plot_runtime_vs_improvement(data)
    assert isinstance(fig, Figure)


# --- Tests for plot_runtime_vs_improvement_scatter ---


def test_plot_runtime_vs_improvement_scatter() -> None:
    """Test plot_runtime_vs_improvement_scatter returns a Figure."""
    data = [
        {"duration_seconds": 1.0, "improvement_pct": 10.0, "runner": "wisq_rules", "circuit": "qft_4"},
        {"duration_seconds": 2.0, "improvement_pct": 20.0, "runner": "tket", "circuit": "qft_4"},
        {"duration_seconds": 1.5, "improvement_pct": 5.0, "runner": "wisq_rules", "circuit": "qaoa_4"},
    ]
    fig = plot_runtime_vs_improvement_scatter(data)
    assert isinstance(fig, Figure)


# --- Tests for plot_pareto_frontier ---


def test_plot_pareto_frontier() -> None:
    """Test plot_pareto_frontier returns a Figure."""
    data = [
        {"duration_seconds": 1.0, "improvement_pct": 10.0, "label": "cfg1"},
        {"duration_seconds": 3.0, "improvement_pct": 20.0, "label": "cfg2"},
        {"duration_seconds": 2.0, "improvement_pct": 5.0, "label": "cfg3"},
    ]
    fig = plot_pareto_frontier(data)
    assert isinstance(fig, Figure)


# --- Tests for plot_comparison_heatmap ---


def test_plot_comparison_heatmap(sample_comparisons: list[RunnerComparison]) -> None:
    """Test plot_comparison_heatmap returns a Figure."""
    fig = plot_comparison_heatmap(sample_comparisons)
    assert isinstance(fig, Figure)


# --- Tests for create_summary_table ---


def test_create_summary_table(sample_stats: list[BenchmarkStatistics]) -> None:
    """Test create_summary_table returns DataFrame with correct columns."""
    df = create_summary_table(sample_stats)
    assert isinstance(df, pd.DataFrame)
    assert "Circuit" in df.columns
    assert "Runner" in df.columns
    assert "Metric" in df.columns
    assert "Mean" in df.columns
    assert len(df) == 2


def test_create_summary_table_with_output_path(
    sample_stats: list[BenchmarkStatistics], tmp_path: Path
) -> None:
    """Test create_summary_table writes file to disk."""
    output = tmp_path / "summary.csv"
    create_summary_table(sample_stats, output_path=output)
    assert output.exists()
    assert output.stat().st_size > 0


# --- Tests for create_comparison_table ---


def test_create_comparison_table(sample_comparisons: list[RunnerComparison]) -> None:
    """Test create_comparison_table returns DataFrame with correct columns."""
    df = create_comparison_table(sample_comparisons)
    assert isinstance(df, pd.DataFrame)
    assert "Circuit" in df.columns
    assert "Metric" in df.columns
    assert "Baseline" in df.columns
    assert "Optimized" in df.columns
    assert "Improvement %" in df.columns
    assert len(df) == 1


def test_create_comparison_table_with_output_path(
    sample_comparisons: list[RunnerComparison], tmp_path: Path
) -> None:
    """Test create_comparison_table writes file to disk."""
    output = tmp_path / "comparison.csv"
    create_comparison_table(sample_comparisons, output_path=output)
    assert output.exists()
    assert output.stat().st_size > 0
