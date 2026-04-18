"""Tests for the single-step grid search module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from benchmarks.ai_transpile.chain_executor import ChainResult, ChainStep, StepResult
from benchmarks.ai_transpile.rl_trajectory import (
    CircuitRecord,
    OptimizerRecord,
    TrajectoryDatabase,
)
from benchmarks.ai_transpile.rl_trajectory.single_step_search import (
    FAST_OPTIMIZERS,
    WISQ_BQSKIT_OPTIMIZER,
    WISQ_RULES_OPTIMIZER,
    AsyncSingleStepRunner,
    OptimizersProgressTracker,
    SingleStepConfig,
    SingleStepProgress,
    SingleStepReport,
    SingleStepResult,
)
from benchmarks.ai_transpile.transpilers import CircuitMetrics, TranspiledCircuit
from qiskit import QuantumCircuit

# --- Fixtures ---


@pytest.fixture
def db(tmp_path: Path) -> TrajectoryDatabase:
    """Create a temporary database."""
    db_path = tmp_path / "test_single_step.db"
    return TrajectoryDatabase(db_path)


@pytest.fixture
def sample_circuit_record() -> CircuitRecord:
    """Create a sample circuit record."""
    return CircuitRecord(
        id=1,
        name="test_qft_4",
        category="qft",
        source="local",
        qasm_path=None,
        num_qubits=4,
        initial_depth=20,
        initial_two_qubit_gates=10,
        initial_two_qubit_depth=8,
        initial_total_gates=30,
        gate_density=7.5,
        two_qubit_ratio=0.333,
    )


@pytest.fixture
def sample_quantum_circuit() -> QuantumCircuit:
    """Create a simple test circuit."""
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    return circuit


@pytest.fixture
def sample_input_metrics() -> CircuitMetrics:
    """Create sample input metrics."""
    return CircuitMetrics(
        depth=20,
        two_qubit_gates=10,
        two_qubit_depth=8,
        total_gates=30,
    )


@pytest.fixture
def sample_output_metrics() -> CircuitMetrics:
    """Create sample optimized output metrics."""
    return CircuitMetrics(
        depth=15,
        two_qubit_gates=6,
        two_qubit_depth=5,
        total_gates=20,
    )


# --- Tests for SingleStepConfig ---


def test_single_step_config_defaults() -> None:
    """Test SingleStepConfig default values."""
    config = SingleStepConfig()

    assert config.max_qubits == 20
    assert config.wisq_bqskit_timeout == 300  # 5 minutes
    assert config.max_concurrent_fast == 4
    assert config.max_concurrent_wisq_rules == 2
    assert config.max_concurrent_wisq_bqskit == 1
    assert config.save_artifacts is True


def test_single_step_config_custom() -> None:
    """Test SingleStepConfig with custom values."""
    config = SingleStepConfig(
        categories=["qft", "qaoa"],
        optimizers=["tket", "qiskit_standard"],
        max_qubits=10,
        wisq_bqskit_timeout=180,
        max_concurrent_fast=2,
    )

    assert config.categories == ["qft", "qaoa"]
    assert config.optimizers == ["tket", "qiskit_standard"]
    assert config.max_qubits == 10
    assert config.wisq_bqskit_timeout == 180
    assert config.max_concurrent_fast == 2


# --- Tests for SingleStepResult ---


def test_single_step_result_improvement(
    sample_input_metrics: CircuitMetrics,
    sample_output_metrics: CircuitMetrics,
) -> None:
    """Test SingleStepResult improvement calculation."""
    result = SingleStepResult(
        circuit_id=1,
        optimizer_id=1,
        circuit_name="test",
        optimizer_name="tket",
        input_metrics=sample_input_metrics,
        output_metrics=sample_output_metrics,
        duration_seconds=1.0,
        success=True,
    )

    # (10 - 6) / 10 * 100 = 40%
    assert result.improvement_percentage == pytest.approx(40.0)


def test_single_step_result_improvement_zero_input() -> None:
    """Test improvement when input has zero 2-qubit gates."""
    result = SingleStepResult(
        circuit_id=1,
        optimizer_id=1,
        circuit_name="test",
        optimizer_name="tket",
        input_metrics=CircuitMetrics(depth=10, two_qubit_gates=0, two_qubit_depth=0, total_gates=10),
        output_metrics=CircuitMetrics(depth=5, two_qubit_gates=0, two_qubit_depth=0, total_gates=5),
        duration_seconds=1.0,
        success=True,
    )

    assert result.improvement_percentage == 0.0


# --- Tests for SingleStepProgress ---


def test_single_step_progress_percent_complete() -> None:
    """Test progress percentage calculation."""
    progress = SingleStepProgress(
        total_runs=100,
        completed_runs=30,
        skipped_runs=20,
        failed_runs=5,
        current_circuit="test",
        current_optimizer="tket",
        elapsed_seconds=60.0,
    )

    # (30 + 20) / 100 = 50%
    assert progress.percent_complete == pytest.approx(50.0)


def test_single_step_progress_empty() -> None:
    """Test progress when no runs."""
    progress = SingleStepProgress(
        total_runs=0,
        completed_runs=0,
        skipped_runs=0,
        failed_runs=0,
        current_circuit="",
        current_optimizer="",
        elapsed_seconds=0.0,
    )

    assert progress.percent_complete == 100.0


# --- Tests for Optimizer Categorization ---


def test_fast_optimizers_include_expected() -> None:
    """Test that fast optimizers include tket and qiskit variants."""
    assert "tket" in FAST_OPTIMIZERS
    assert "qiskit_ai" in FAST_OPTIMIZERS
    assert "qiskit_standard" in FAST_OPTIMIZERS


def test_wisq_optimizers_separate() -> None:
    """Test that WISQ optimizers are correctly identified."""
    assert WISQ_RULES_OPTIMIZER == "wisq_rules"
    assert WISQ_BQSKIT_OPTIMIZER == "wisq_bqskit"
    assert WISQ_RULES_OPTIMIZER not in FAST_OPTIMIZERS
    assert WISQ_BQSKIT_OPTIMIZER not in FAST_OPTIMIZERS


# --- Tests for Database Methods ---


def test_run_exists_false(db: TrajectoryDatabase) -> None:
    """Test run_exists returns False for non-existent runs."""
    assert db.run_exists(1, 1) is False


def test_run_exists_true(db: TrajectoryDatabase) -> None:
    """Test run_exists returns True after inserting a run."""
    # Insert a circuit
    circuit = CircuitRecord(
        id=None,
        name="test",
        category="qft",
        source="local",
        qasm_path=None,
        num_qubits=4,
        initial_depth=20,
        initial_two_qubit_gates=10,
        initial_two_qubit_depth=8,
        initial_total_gates=30,
        gate_density=7.5,
        two_qubit_ratio=0.333,
    )
    circuit_id = db.insert_circuit(circuit)

    # Insert an optimizer
    optimizer = OptimizerRecord(
        id=None,
        name="test_opt",
        runner_type="tket",
        options={},
        description="Test optimizer",
    )
    optimizer_id = db.insert_optimizer(optimizer)

    # Insert a run
    db.insert_optimization_run(
        circuit_id=circuit_id,
        optimizer_id=optimizer_id,
        input_depth=20,
        input_two_qubit_gates=10,
        input_two_qubit_depth=8,
        input_total_gates=30,
        output_depth=15,
        output_two_qubit_gates=6,
        output_two_qubit_depth=5,
        output_total_gates=20,
        duration_seconds=1.0,
        success=True,
    )

    assert db.run_exists(circuit_id, optimizer_id) is True
    assert db.run_exists(circuit_id, optimizer_id + 1) is False


def test_count_optimization_runs(db: TrajectoryDatabase) -> None:
    """Test counting optimization runs."""
    assert db.count_optimization_runs() == 0

    # Insert a circuit and optimizer
    circuit = CircuitRecord(
        id=None, name="test", category="qft", source="local", qasm_path=None,
        num_qubits=4, initial_depth=20, initial_two_qubit_gates=10,
        initial_two_qubit_depth=8, initial_total_gates=30, gate_density=7.5, two_qubit_ratio=0.333,
    )
    circuit_id = db.insert_circuit(circuit)

    optimizer = OptimizerRecord(id=None, name="test_opt", runner_type="tket", options={})
    optimizer_id = db.insert_optimizer(optimizer)

    # Insert a run
    db.insert_optimization_run(
        circuit_id=circuit_id, optimizer_id=optimizer_id,
        input_depth=20, input_two_qubit_gates=10, input_two_qubit_depth=8, input_total_gates=30,
        output_depth=15, output_two_qubit_gates=6, output_two_qubit_depth=5, output_total_gates=20,
        duration_seconds=1.0, success=True,
    )

    assert db.count_optimization_runs() == 1


# --- Tests for AsyncSingleStepRunner ---


def test_async_runner_init(tmp_path: Path) -> None:
    """Test AsyncSingleStepRunner initialization."""
    config = SingleStepConfig(
        database_path=tmp_path / "test.db",
        optimizers=["tket", "qiskit_standard"],
    )

    with AsyncSingleStepRunner(config) as runner:
        # Check optimizers were registered
        optimizers = runner.db.list_optimizers()
        assert len(optimizers) >= 2

        # Semaphores should be None initially (created lazily in async context)
        assert runner._sem_fast is None
        assert runner._sem_wisq_rules is None
        assert runner._sem_wisq_bqskit is None


def test_async_runner_invalid_optimizer(tmp_path: Path) -> None:
    """Test AsyncSingleStepRunner with invalid optimizer."""
    config = SingleStepConfig(
        database_path=tmp_path / "test.db",
        optimizers=["invalid_optimizer"],
    )

    with pytest.raises(ValueError, match="Unknown optimizer"):
        AsyncSingleStepRunner(config)


def test_async_runner_get_semaphore(tmp_path: Path) -> None:
    """Test that correct semaphore is returned for each optimizer type."""
    import asyncio
    
    config = SingleStepConfig(
        database_path=tmp_path / "test.db",
        optimizers=["tket", "wisq_rules", "wisq_bqskit"],
    )

    async def _test() -> None:
        with AsyncSingleStepRunner(config) as runner:
            # Create semaphores as done in run_exhaustive_search
            runner._sem_fast = asyncio.Semaphore(runner.config.max_concurrent_fast)
            runner._sem_wisq_rules = asyncio.Semaphore(runner.config.max_concurrent_wisq_rules)
            runner._sem_wisq_bqskit = asyncio.Semaphore(runner.config.max_concurrent_wisq_bqskit)
            
            # Fast optimizer gets fast semaphore
            assert runner._get_semaphore("tket") is runner._sem_fast
            assert runner._get_semaphore("qiskit_ai") is runner._sem_fast
            assert runner._get_semaphore("qiskit_standard") is runner._sem_fast

            # WISQ optimizers get their specific semaphores
            assert runner._get_semaphore("wisq_rules") is runner._sem_wisq_rules
            assert runner._get_semaphore("wisq_bqskit") is runner._sem_wisq_bqskit
    
    asyncio.run(_test())


@patch("benchmarks.ai_transpile.rl_trajectory.single_step_search.execute_chain")
def test_async_runner_run_single_step(
    mock_execute: MagicMock,
    tmp_path: Path,
    sample_quantum_circuit: QuantumCircuit,
    sample_input_metrics: CircuitMetrics,
    sample_output_metrics: CircuitMetrics,
) -> None:
    """Test running a single optimization step."""
    # Setup mock
    mock_step_result = StepResult(
        step=ChainStep(runner_type="tket", name="tket"),
        step_index=0,
        input_metrics=sample_input_metrics,
        output_metrics=sample_output_metrics,
        transpiled=TranspiledCircuit(
            optimizer="tket",
            label="test",
            circuit=sample_quantum_circuit,
            metrics=sample_output_metrics,
        ),
        duration_seconds=1.0,
    )

    mock_chain_result = ChainResult(
        chain_name="tket",
        steps=[ChainStep(runner_type="tket")],
        step_results=[mock_step_result],
        initial_circuit=sample_quantum_circuit,
        initial_metrics=sample_input_metrics,
        final_circuit=sample_quantum_circuit,
        final_metrics=sample_output_metrics,
        total_duration_seconds=1.0,
    )
    mock_execute.return_value = mock_chain_result

    config = SingleStepConfig(
        database_path=tmp_path / "test.db",
        optimizers=["tket"],
        save_artifacts=False,  # Don't save for test
    )

    with AsyncSingleStepRunner(config) as runner:
        # Insert a test circuit
        circuit_record = CircuitRecord(
            id=None,
            name="test_circuit",
            category="qft",
            source="local",
            qasm_path=None,
            num_qubits=4,
            initial_depth=20,
            initial_two_qubit_gates=10,
            initial_two_qubit_depth=8,
            initial_total_gates=30,
            gate_density=7.5,
            two_qubit_ratio=0.333,
        )
        circuit_id = runner.db.insert_circuit(circuit_record)
        circuit_record = runner.db.get_circuit_by_id(circuit_id)
        assert circuit_record is not None

        # Mock circuit loading and analyze
        analyze_patch = "benchmarks.ai_transpile.rl_trajectory.single_step_search.analyze_circuit"
        with patch.object(runner, "_load_circuit", return_value=sample_quantum_circuit):
            with patch(analyze_patch, return_value=sample_input_metrics):
                result = runner._run_single_step_sync(
                    circuit_record,
                    "tket",
                    tmp_path / "output",
                )

        assert result.success is True
        assert result.circuit_name == "test_circuit"
        assert result.optimizer_name == "tket"


def test_async_runner_record_result(
    tmp_path: Path,
    sample_input_metrics: CircuitMetrics,
    sample_output_metrics: CircuitMetrics,
) -> None:
    """Test recording a result to the database."""
    config = SingleStepConfig(
        database_path=tmp_path / "test.db",
        optimizers=["tket"],
    )

    with AsyncSingleStepRunner(config) as runner:
        # Insert circuit
        circuit_record = CircuitRecord(
            id=None, name="test", category="qft", source="local", qasm_path=None,
            num_qubits=4, initial_depth=20, initial_two_qubit_gates=10,
            initial_two_qubit_depth=8, initial_total_gates=30, gate_density=7.5, two_qubit_ratio=0.333,
        )
        circuit_id = runner.db.insert_circuit(circuit_record)

        # Get optimizer
        optimizer = runner.db.get_optimizer_by_name("tket")
        assert optimizer is not None

        # Create result
        result = SingleStepResult(
            circuit_id=circuit_id,
            optimizer_id=optimizer.id or 0,
            circuit_name="test",
            optimizer_name="tket",
            input_metrics=sample_input_metrics,
            output_metrics=sample_output_metrics,
            duration_seconds=1.5,
            success=True,
        )

        # Record it
        runner._record_result(result)

        # Verify
        assert runner.db.count_optimization_runs() == 1
        assert runner.db.run_exists(circuit_id, optimizer.id or 0) is True


def test_async_runner_resume_skips_existing(tmp_path: Path) -> None:
    """Test that resume mode skips existing runs."""
    config = SingleStepConfig(
        database_path=tmp_path / "test.db",
        optimizers=["tket"],
    )

    with AsyncSingleStepRunner(config) as runner:
        # Insert circuit
        circuit_record = CircuitRecord(
            id=None, name="test", category="qft", source="local", qasm_path=None,
            num_qubits=4, initial_depth=20, initial_two_qubit_gates=10,
            initial_two_qubit_depth=8, initial_total_gates=30, gate_density=7.5, two_qubit_ratio=0.333,
        )
        circuit_id = runner.db.insert_circuit(circuit_record)

        # Get optimizer
        optimizer = runner.db.get_optimizer_by_name("tket")
        assert optimizer is not None

        # Insert an existing run
        runner.db.insert_optimization_run(
            circuit_id=circuit_id, optimizer_id=optimizer.id or 0,
            input_depth=20, input_two_qubit_gates=10, input_two_qubit_depth=8, input_total_gates=30,
            output_depth=15, output_two_qubit_gates=6, output_two_qubit_depth=5, output_total_gates=20,
            duration_seconds=1.0, success=True,
        )

        # Run search - should skip the existing run
        report = runner.run_sync(resume=True)

        # Should have 1 skipped, 0 completed (since only 1 circuit and already done)
        assert report.skipped_runs == 1
        assert report.completed_runs == 0


# --- Tests for SingleStepReport ---


def test_single_step_report_fields() -> None:
    """Test SingleStepReport has expected fields."""
    report = SingleStepReport(
        total_circuits=10,
        total_optimizers=5,
        total_runs=50,
        completed_runs=40,
        skipped_runs=5,
        failed_runs=5,
        total_duration_seconds=120.0,
        best_by_optimizer={"tket": {"circuit": "qft_4", "improvement": 30.0}},
        failures=[{"circuit": "test", "optimizer": "wisq", "error": "timeout"}],
    )

    assert report.total_circuits == 10
    assert report.total_optimizers == 5
    assert report.total_runs == 50
    assert report.completed_runs == 40
    assert report.skipped_runs == 5
    assert report.failed_runs == 5
    assert "tket" in report.best_by_optimizer
    assert len(report.failures) == 1


# --- Tests for wisq_bqskit timeout configuration ---


def test_wisq_bqskit_timeout_in_config(tmp_path: Path) -> None:
    """Test that WISQ+BQSKit timeout is properly set."""
    config = SingleStepConfig(
        database_path=tmp_path / "test.db",
        optimizers=["wisq_bqskit"],
        wisq_bqskit_timeout=180,  # 3 minutes
    )

    with AsyncSingleStepRunner(config) as runner:
        optimizer = runner.db.get_optimizer_by_name("wisq_bqskit")
        assert optimizer is not None
        # The timeout should be stored in options
        assert optimizer.options.get("opt_timeout") == 180


def test_wisq_bqskit_default_timeout(tmp_path: Path) -> None:
    """Test that WISQ+BQSKit uses default 5 minute timeout."""
    config = SingleStepConfig(
        database_path=tmp_path / "test.db",
        optimizers=["wisq_bqskit"],
        # Using default wisq_bqskit_timeout=300
    )

    with AsyncSingleStepRunner(config) as runner:
        optimizer = runner.db.get_optimizer_by_name("wisq_bqskit")
        assert optimizer is not None
        assert optimizer.options.get("opt_timeout") == 300


# --- Tests for OptimizersProgressTracker ---


def test_progress_tracker_initialization() -> None:
    """Test OptimizersProgressTracker initialization."""
    optimizer_names = ["tket", "qiskit_ai", "wisq_rules"]
    optimizer_totals = {"tket": 10, "qiskit_ai": 15, "wisq_rules": 8}
    
    tracker = OptimizersProgressTracker(optimizer_names, optimizer_totals)
    
    assert tracker.optimizer_names == optimizer_names
    assert tracker.optimizer_totals == optimizer_totals
    assert all(tracker._completed[name] == 0 for name in optimizer_names)
    assert all(tracker._failed[name] == 0 for name in optimizer_names)
    assert all(tracker._running[name] == 0 for name in optimizer_names)
    assert all(tracker._skipped[name] == 0 for name in optimizer_names)


def test_progress_tracker_task_lifecycle() -> None:
    """Test progress tracker task start/complete/skip."""
    import asyncio
    
    optimizer_names = ["tket"]
    optimizer_totals = {"tket": 5}
    
    async def _test() -> None:
        tracker = OptimizersProgressTracker(optimizer_names, optimizer_totals)
        
        # Start a task
        await tracker.start_task("tket")
        assert tracker._running["tket"] == 1
        assert tracker._total_running == 1
        
        # Complete successfully
        await tracker.complete_task("tket", success=True)
        assert tracker._running["tket"] == 0
        assert tracker._completed["tket"] == 1
        assert tracker._total_completed == 1
        
        # Start and fail a task
        await tracker.start_task("tket")
        await tracker.complete_task("tket", success=False)
        assert tracker._failed["tket"] == 1
        assert tracker._total_failed == 1
        
        # Skip a task
        await tracker.skip_task("tket")
        assert tracker._skipped["tket"] == 1
        assert tracker._total_skipped == 1
    
    asyncio.run(_test())


def test_progress_tracker_context_manager() -> None:
    """Test progress tracker context manager."""
    optimizer_names = ["tket", "qiskit_ai"]
    optimizer_totals = {"tket": 10, "qiskit_ai": 15}
    
    tracker = OptimizersProgressTracker(optimizer_names, optimizer_totals)
    
    # Context manager should start and stop progress display
    with tracker:
        assert tracker._progress is not None
        assert tracker._overall_task is not None
        assert len(tracker._optimizer_tasks) == 2
    
    # After exit, progress should be stopped (but still accessible)
    assert tracker._progress is not None


# --- Tests for async paths ---


def test_run_single_step_async() -> None:
    """Test _run_single_step_async acquires semaphore and calls sync method."""
    import asyncio

    async def _test() -> None:
        config = SingleStepConfig(
            database_path=tmp_path / "test.db",
            optimizers=["tket"],
            save_artifacts=False,
        )

        with AsyncSingleStepRunner(config) as runner:
            # Manually create semaphores (normally done in run_exhaustive_search)
            runner._sem_fast = asyncio.Semaphore(4)
            runner._sem_wisq_rules = asyncio.Semaphore(2)
            runner._sem_wisq_bqskit = asyncio.Semaphore(1)

            circuit_record = CircuitRecord(
                id=None, name="test", category="qft", source="local",
                qasm_path=None, num_qubits=4,
                initial_depth=20, initial_two_qubit_gates=10,
                initial_two_qubit_depth=8, initial_total_gates=30,
                gate_density=7.5, two_qubit_ratio=0.333,
            )
            circuit_id = runner.db.insert_circuit(circuit_record)
            circuit_record = runner.db.get_circuit_by_id(circuit_id)
            assert circuit_record is not None

            mock_result = SingleStepResult(
                circuit_id=circuit_id,
                optimizer_id=1,
                circuit_name="test",
                optimizer_name="tket",
                input_metrics=CircuitMetrics(20, 10, 8, 30),
                output_metrics=CircuitMetrics(15, 6, 5, 20),
                duration_seconds=1.0,
                success=True,
            )

            with patch.object(runner, "_run_single_step_sync", return_value=mock_result):
                result = await runner._run_single_step_async(circuit_record, "tket")

            assert result.success is True
            assert result.circuit_name == "test"

    # Need tmp_path from pytest, create manually
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        asyncio.run(_test())


def test_run_exhaustive_search_empty(tmp_path: Path) -> None:
    """Test run_sync with empty database returns report with 0 runs."""
    config = SingleStepConfig(
        database_path=tmp_path / "test.db",
        optimizers=["tket"],
        save_artifacts=False,
    )

    with AsyncSingleStepRunner(config) as runner:
        report = runner.run_sync(resume=True)

    assert report.total_circuits == 0
    assert report.completed_runs == 0
    assert report.total_runs == 0


@patch("benchmarks.ai_transpile.rl_trajectory.single_step_search.execute_chain")
def test_run_exhaustive_search_with_mocked_sync(
    mock_execute: MagicMock,
    tmp_path: Path,
    sample_quantum_circuit: QuantumCircuit,
    sample_input_metrics: CircuitMetrics,
    sample_output_metrics: CircuitMetrics,
) -> None:
    """Test run_sync with one circuit and mocked sync method."""
    config = SingleStepConfig(
        database_path=tmp_path / "test.db",
        optimizers=["tket"],
        save_artifacts=False,
    )

    with AsyncSingleStepRunner(config) as runner:
        # Insert circuit with a valid path (needed for _load_circuit)
        qasm_content = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\n'
            "h q[0];\ncx q[0], q[1];\ncx q[1], q[2];\ncx q[2], q[3];\n"
        )
        qasm_file = tmp_path / "test.qasm"
        qasm_file.write_text(qasm_content)

        record = CircuitRecord(
            id=None, name="test", category="qft", source="local",
            qasm_path=str(qasm_file), num_qubits=4,
            initial_depth=4, initial_two_qubit_gates=3,
            initial_two_qubit_depth=3, initial_total_gates=4,
            gate_density=1.0, two_qubit_ratio=0.75,
        )
        runner.db.insert_circuit(record)

        # Mock _run_single_step_sync to return a success result
        mock_result = SingleStepResult(
            circuit_id=1,
            optimizer_id=1,
            circuit_name="test",
            optimizer_name="tket",
            input_metrics=sample_input_metrics,
            output_metrics=sample_output_metrics,
            duration_seconds=1.0,
            success=True,
        )

        with patch.object(runner, "_run_single_step_sync", return_value=mock_result):
            report = runner.run_sync(resume=False)

    assert report.total_circuits == 1
    assert report.completed_runs == 1


def test_run_single_step_grid_search_convenience(tmp_path: Path) -> None:
    """Test module-level run_single_step_grid_search with empty db."""
    from benchmarks.ai_transpile.rl_trajectory.single_step_search import (
        run_single_step_grid_search,
    )

    report = run_single_step_grid_search(
        database_path=tmp_path / "test.db",
        optimizers=["tket"],
        max_qubits=4,
        save_artifacts=False,
    )

    assert report.total_circuits == 0
    assert report.total_runs == 0
