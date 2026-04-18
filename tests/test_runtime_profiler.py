"""Tests for runtime profiling utilities."""

from __future__ import annotations

import time
from pathlib import Path

import pytest


def test_timer_context_manager() -> None:
    """Test the Timer context manager for measuring execution time."""
    from benchmarks.ai_transpile.runtime_profiler import Timer

    with Timer() as timer:
        time.sleep(0.01)  # Sleep for 10ms

    assert timer.elapsed >= 0.01
    assert timer.elapsed < 0.1  # Should be much less than 100ms


def test_timer_manual_start_stop() -> None:
    """Test Timer with manual start and stop."""
    from benchmarks.ai_transpile.runtime_profiler import Timer

    timer = Timer()
    timer.start()
    time.sleep(0.01)
    timer.stop()

    assert timer.elapsed >= 0.01
    assert timer.elapsed < 0.1


def test_timer_multiple_measurements() -> None:
    """Test multiple timing measurements."""
    from benchmarks.ai_transpile.runtime_profiler import Timer

    measurements = []
    for _ in range(3):
        with Timer() as timer:
            time.sleep(0.01)
        measurements.append(timer.elapsed)

    assert all(t >= 0.01 for t in measurements)
    assert len(measurements) == 3


def test_profile_function_decorator() -> None:
    """Test the profile_function decorator."""
    from benchmarks.ai_transpile.runtime_profiler import profile_function

    call_count = 0

    @profile_function
    def sample_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        time.sleep(0.01)
        return x * 2

    result = sample_function(5)

    assert result == 10
    assert call_count == 1
    # The decorator should have recorded timing information
    # (checked via profiler stats in later tests)


def test_runtime_statistics() -> None:
    """Test RuntimeStatistics dataclass."""
    from benchmarks.ai_transpile.runtime_profiler import RuntimeStatistics

    stats = RuntimeStatistics(
        operation="test_op",
        mean_seconds=1.5,
        std_seconds=0.2,
        min_seconds=1.2,
        max_seconds=1.9,
        total_seconds=15.0,
        count=10,
    )

    assert stats.operation == "test_op"
    assert stats.mean_seconds == 1.5
    assert stats.count == 10


def test_runtime_profiler_record() -> None:
    """Test recording timing measurements."""
    from benchmarks.ai_transpile.runtime_profiler import RuntimeProfiler

    profiler = RuntimeProfiler()
    profiler.record("operation_a", 1.5)
    profiler.record("operation_a", 2.0)
    profiler.record("operation_b", 0.5)

    assert "operation_a" in profiler.timings
    assert len(profiler.timings["operation_a"]) == 2
    assert "operation_b" in profiler.timings
    assert len(profiler.timings["operation_b"]) == 1


def test_runtime_profiler_context_manager() -> None:
    """Test RuntimeProfiler as context manager."""
    from benchmarks.ai_transpile.runtime_profiler import RuntimeProfiler

    profiler = RuntimeProfiler()

    with profiler.measure("sleep_operation"):
        time.sleep(0.01)

    assert "sleep_operation" in profiler.timings
    assert len(profiler.timings["sleep_operation"]) == 1
    assert profiler.timings["sleep_operation"][0] >= 0.01


def test_runtime_profiler_get_statistics() -> None:
    """Test computing statistics from profiler."""
    from benchmarks.ai_transpile.runtime_profiler import RuntimeProfiler

    profiler = RuntimeProfiler()
    profiler.record("op1", 1.0)
    profiler.record("op1", 2.0)
    profiler.record("op1", 1.5)
    profiler.record("op2", 0.5)

    stats = profiler.get_statistics()

    assert len(stats) == 2
    op1_stats = next(s for s in stats if s.operation == "op1")
    assert op1_stats.count == 3
    assert op1_stats.mean_seconds == pytest.approx(1.5, rel=0.01)
    assert op1_stats.total_seconds == pytest.approx(4.5, rel=0.01)


def test_runtime_profiler_export_json(tmp_path: Path) -> None:
    """Test exporting profiler data to JSON."""
    from benchmarks.ai_transpile.runtime_profiler import RuntimeProfiler

    profiler = RuntimeProfiler()
    profiler.record("op1", 1.0)
    profiler.record("op1", 2.0)
    profiler.record("op2", 0.5)

    output_file = tmp_path / "profiler_data.json"
    profiler.export_json(output_file)

    assert output_file.exists()

    # Load and verify the JSON
    import json

    data = json.loads(output_file.read_text())
    assert "statistics" in data
    assert len(data["statistics"]) == 2


def test_runtime_profiler_clear() -> None:
    """Test clearing profiler data."""
    from benchmarks.ai_transpile.runtime_profiler import RuntimeProfiler

    profiler = RuntimeProfiler()
    profiler.record("op1", 1.0)
    profiler.record("op2", 2.0)

    assert len(profiler.timings) == 2

    profiler.clear()

    assert len(profiler.timings) == 0


def test_aggregate_timing_data() -> None:
    """Test aggregating timing data from multiple profiler runs."""
    from benchmarks.ai_transpile.runtime_profiler import aggregate_timing_data

    profiler1 = {"op1": [1.0, 2.0], "op2": [0.5]}
    profiler2 = {"op1": [1.5], "op3": [3.0]}

    aggregated = aggregate_timing_data([profiler1, profiler2])

    assert "op1" in aggregated
    assert len(aggregated["op1"]) == 3
    assert aggregated["op1"] == [1.0, 2.0, 1.5]
    assert "op2" in aggregated
    assert "op3" in aggregated


def test_timing_context_with_profiler() -> None:
    """Test timing context that records to a profiler."""
    from benchmarks.ai_transpile.runtime_profiler import RuntimeProfiler

    profiler = RuntimeProfiler()

    with profiler.measure("test_op"):
        _ = 1 + 1  # Simple operation
        time.sleep(0.01)

    assert "test_op" in profiler.timings
    assert profiler.timings["test_op"][0] >= 0.01


def test_profiler_with_nested_operations() -> None:
    """Test profiler with nested operation measurements."""
    from benchmarks.ai_transpile.runtime_profiler import RuntimeProfiler

    profiler = RuntimeProfiler()

    with profiler.measure("outer_op"):
        time.sleep(0.01)
        with profiler.measure("inner_op"):
            time.sleep(0.01)

    assert "outer_op" in profiler.timings
    assert "inner_op" in profiler.timings
    assert profiler.timings["outer_op"][0] >= 0.02  # Should include inner time
    assert profiler.timings["inner_op"][0] >= 0.01


def test_cost_benefit_ratio() -> None:
    """Test computing cost-benefit ratio."""
    from benchmarks.ai_transpile.runtime_profiler import compute_cost_benefit_ratio

    # 10% improvement in 5 seconds = 2% improvement per second
    ratio = compute_cost_benefit_ratio(improvement_pct=10.0, duration_seconds=5.0)
    assert ratio == pytest.approx(2.0, rel=0.01)

    # Negative improvement (regression)
    ratio = compute_cost_benefit_ratio(improvement_pct=-5.0, duration_seconds=2.0)
    assert ratio == pytest.approx(-2.5, rel=0.01)


def test_cost_benefit_zero_duration() -> None:
    """Test cost-benefit ratio with zero duration."""
    from benchmarks.ai_transpile.runtime_profiler import compute_cost_benefit_ratio

    # Zero duration should return infinity or raise error
    ratio = compute_cost_benefit_ratio(improvement_pct=10.0, duration_seconds=0.0)
    assert ratio == float("inf") or ratio == 0.0

