"""Tests for the grid search module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from benchmarks.ai_transpile.chain_executor import ChainResult, ChainStep, StepResult
from benchmarks.ai_transpile.rl_trajectory import (
    OPTIMIZER_CONFIGS,
    CircuitRecord,
    GridSearchConfig,
    GridSearchRunner,
    TrajectoryDatabase,
    generate_optimizer_combinations,
)
from benchmarks.ai_transpile.rl_trajectory.state import CATEGORIES, RLState
from benchmarks.ai_transpile.transpilers import CircuitMetrics, TranspiledCircuit
from qiskit import QuantumCircuit

# --- Fixtures ---


@pytest.fixture
def db(tmp_path: Path) -> TrajectoryDatabase:
    """Create a temporary database."""
    db_path = tmp_path / "test_trajectories.db"
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
def sample_initial_metrics() -> CircuitMetrics:
    """Create sample initial metrics."""
    return CircuitMetrics(
        depth=20,
        two_qubit_gates=10,
        two_qubit_depth=8,
        total_gates=30,
    )


@pytest.fixture
def sample_final_metrics() -> CircuitMetrics:
    """Create sample improved metrics."""
    return CircuitMetrics(
        depth=15,
        two_qubit_gates=6,
        two_qubit_depth=5,
        total_gates=20,
    )


# --- Tests for generate_optimizer_combinations ---


def test_generate_single_optimizers() -> None:
    """Test generating single optimizer combinations."""
    optimizers = ["opt1", "opt2"]
    combinations = generate_optimizer_combinations(optimizers, max_length=1)

    assert combinations == [["opt1"], ["opt2"]]


def test_generate_two_step_chains() -> None:
    """Test generating two-step chains."""
    optimizers = ["opt1", "opt2"]
    combinations = generate_optimizer_combinations(optimizers, max_length=2)

    expected = [
        ["opt1"],
        ["opt2"],
        ["opt1", "opt1"],
        ["opt1", "opt2"],
        ["opt2", "opt1"],
        ["opt2", "opt2"],
    ]
    assert combinations == expected


def test_generate_three_step_chains() -> None:
    """Test generating three-step chains."""
    optimizers = ["a", "b"]
    combinations = generate_optimizer_combinations(optimizers, max_length=3)

    # 1-step: 2, 2-step: 4, 3-step: 8 = 14 total
    assert len(combinations) == 14

    # Check some specific combinations
    assert ["a"] in combinations
    assert ["b"] in combinations
    assert ["a", "b"] in combinations
    assert ["a", "b", "a"] in combinations


def test_generate_without_single() -> None:
    """Test generating without single optimizers."""
    optimizers = ["opt1", "opt2"]
    combinations = generate_optimizer_combinations(optimizers, max_length=2, include_single=False)

    # Should only have 2-step combinations
    assert len(combinations) == 4
    assert ["opt1"] not in combinations
    assert ["opt1", "opt2"] in combinations


def test_generate_realistic_count() -> None:
    """Test combination count matches expected."""
    optimizers = list(OPTIMIZER_CONFIGS.keys())  # 5 optimizers

    # Single: 5
    # Two-step: 5^2 = 25
    # Three-step: 5^3 = 125
    # Total: 155
    combinations = generate_optimizer_combinations(optimizers, max_length=3)
    assert len(combinations) == 155


# --- Tests for RLState ---


def test_rl_state_from_metrics(sample_initial_metrics: CircuitMetrics) -> None:
    """Test creating RLState from metrics."""
    state = RLState.from_metrics(
        metrics=sample_initial_metrics,
        num_qubits=4,
        category="qft",
        steps_taken=0,
        time_budget_remaining=300.0,
    )

    assert state.depth == 20
    assert state.two_qubit_gates == 10
    assert state.num_qubits == 4
    assert state.gate_density == pytest.approx(7.5)
    assert state.two_qubit_ratio == pytest.approx(0.333, rel=0.01)
    assert state.steps_taken == 0
    assert state.time_budget_remaining == 300.0


def test_rl_state_to_vector(sample_initial_metrics: CircuitMetrics) -> None:
    """Test converting RLState to numpy vector."""
    state = RLState.from_metrics(
        metrics=sample_initial_metrics,
        num_qubits=4,
        category="qft",
    )

    vector = state.to_vector()

    assert vector.shape == (9 + 3 + len(CATEGORIES),)
    assert vector[0] == 20  # depth
    assert vector[1] == 10  # two_qubit_gates
    assert vector[4] == 4   # num_qubits


def test_rl_state_state_dim() -> None:
    """Test state dimension calculation."""
    dim = RLState.state_dim()
    assert dim == 9 + 3 + len(CATEGORIES)


def test_rl_state_with_updated_metrics(sample_initial_metrics: CircuitMetrics) -> None:
    """Test creating updated state."""
    initial_state = RLState.from_metrics(
        metrics=sample_initial_metrics,
        num_qubits=4,
        category="qft",
        steps_taken=0,
        time_budget_remaining=300.0,
    )

    new_metrics = CircuitMetrics(depth=15, two_qubit_gates=6, two_qubit_depth=5, total_gates=20)
    updated_state = initial_state.with_updated_metrics(new_metrics, time_spent=2.0)

    assert updated_state.depth == 15
    assert updated_state.two_qubit_gates == 6
    assert updated_state.steps_taken == 1
    assert updated_state.time_budget_remaining == 298.0
    assert updated_state.num_qubits == 4  # Unchanged


# --- Tests for GridSearchConfig ---


def test_grid_search_config_defaults() -> None:
    """Test GridSearchConfig default values."""
    config = GridSearchConfig()

    assert config.max_chain_length == 3
    assert config.enable_chain_search is True
    assert config.time_budget == 300.0
    assert config.max_qubits == 20
    assert len(config.optimizers) == len(OPTIMIZER_CONFIGS)


def test_grid_search_config_custom() -> None:
    """Test GridSearchConfig with custom values."""
    config = GridSearchConfig(
        categories=["qft", "qaoa"],
        optimizers=["wisq_rules", "tket"],
        max_chain_length=2,
        max_qubits=10,
    )

    assert config.categories == ["qft", "qaoa"]
    assert config.optimizers == ["wisq_rules", "tket"]
    assert config.max_chain_length == 2
    assert config.max_qubits == 10


# --- Tests for GridSearchRunner ---


def test_grid_search_runner_init(tmp_path: Path) -> None:
    """Test GridSearchRunner initialization."""
    config = GridSearchConfig(
        database_path=tmp_path / "test.db",
        optimizers=["wisq_rules", "tket"],
    )

    runner = GridSearchRunner(config)

    # Check optimizers were registered
    optimizers = runner.db.list_optimizers()
    assert len(optimizers) >= 2
    runner.close()


def test_grid_search_runner_invalid_optimizer(tmp_path: Path) -> None:
    """Test GridSearchRunner with invalid optimizer."""
    config = GridSearchConfig(
        database_path=tmp_path / "test.db",
        optimizers=["invalid_optimizer"],
    )

    with pytest.raises(ValueError, match="Unknown optimizer"):
        GridSearchRunner(config)


@patch("benchmarks.ai_transpile.rl_trajectory.grid_search.execute_chain")
def test_grid_search_run_chain(
    mock_execute: MagicMock,
    tmp_path: Path,
    sample_quantum_circuit: QuantumCircuit,
    sample_initial_metrics: CircuitMetrics,
    sample_final_metrics: CircuitMetrics,
) -> None:
    """Test running a chain."""
    # Setup mock
    mock_step_result = StepResult(
        step=ChainStep(runner_type="qiskit_standard", name="qiskit_standard"),
        step_index=0,
        input_metrics=sample_initial_metrics,
        output_metrics=sample_final_metrics,
        transpiled=TranspiledCircuit(
            optimizer="qiskit_standard",
            label="test",
            circuit=sample_quantum_circuit,
            metrics=sample_final_metrics,
        ),
        duration_seconds=1.0,
    )

    mock_chain_result = ChainResult(
        chain_name="test",
        steps=[ChainStep(runner_type="qiskit_standard")],
        step_results=[mock_step_result],
        initial_circuit=sample_quantum_circuit,
        initial_metrics=sample_initial_metrics,
        final_circuit=sample_quantum_circuit,
        final_metrics=sample_final_metrics,
        total_duration_seconds=1.0,
    )
    mock_execute.return_value = mock_chain_result

    # Create runner
    config = GridSearchConfig(
        database_path=tmp_path / "test.db",
        optimizers=["qiskit_standard"],
    )

    with GridSearchRunner(config) as runner:
        # Insert a test circuit
        circuit_record = CircuitRecord(
            id=None,
            name="test_circuit",
            category="qft",
            source="local",
            qasm_path=None,  # Will use mock
            num_qubits=4,
            initial_depth=20,
            initial_two_qubit_gates=10,
            initial_two_qubit_depth=8,
            initial_total_gates=30,
            gate_density=7.5,
            two_qubit_ratio=0.333,
        )
        circuit_id = runner.db.insert_circuit(circuit_record)

        # Reload to get ID
        circuit_record = runner.db.get_circuit_by_id(circuit_id)
        assert circuit_record is not None

        # Mock the circuit loading
        with patch.object(runner, "_load_circuit", return_value=sample_quantum_circuit):
            result = runner.run_chain(circuit_record, ["qiskit_standard"])

        assert result is not None
        # The chain name comes from the mock, not the optimizer sequence
        assert "qiskit_standard" in result.chain_name or result.chain_name == "test"


def test_grid_search_record_trajectory(
    tmp_path: Path,
    sample_quantum_circuit: QuantumCircuit,
    sample_initial_metrics: CircuitMetrics,
    sample_final_metrics: CircuitMetrics,
) -> None:
    """Test recording a trajectory to the database."""
    config = GridSearchConfig(
        database_path=tmp_path / "test.db",
        optimizers=["qiskit_standard"],
    )

    with GridSearchRunner(config) as runner:
        # Insert circuit
        circuit_record = CircuitRecord(
            id=None, name="test", category="qft", source="local", qasm_path=None,
            num_qubits=4, initial_depth=20, initial_two_qubit_gates=10,
            initial_two_qubit_depth=8, initial_total_gates=30, gate_density=7.5, two_qubit_ratio=0.333,
        )
        circuit_id = runner.db.insert_circuit(circuit_record)
        circuit_record = runner.db.get_circuit_by_id(circuit_id)
        assert circuit_record is not None

        # Create mock chain result
        mock_step_result = StepResult(
            step=ChainStep(runner_type="qiskit_standard", name="qiskit_standard"),
            step_index=0,
            input_metrics=sample_initial_metrics,
            output_metrics=sample_final_metrics,
            transpiled=TranspiledCircuit(
                optimizer="qiskit_standard", label="test",
                circuit=sample_quantum_circuit, metrics=sample_final_metrics,
            ),
            duration_seconds=1.0,
        )

        chain_result = ChainResult(
            chain_name="qiskit_standard",
            steps=[ChainStep(runner_type="qiskit_standard")],
            step_results=[mock_step_result],
            initial_circuit=sample_quantum_circuit,
            initial_metrics=sample_initial_metrics,
            final_circuit=sample_quantum_circuit,
            final_metrics=sample_final_metrics,
            total_duration_seconds=1.0,
        )

        # Record trajectory
        trajectory_id = runner.record_trajectory(
            circuit_record,
            chain_result,
            ["qiskit_standard"],
        )

        assert trajectory_id > 0
        assert runner.db.count_trajectories() == 1
        assert runner.db.count_trajectory_steps() == 1


def test_grid_search_skip_duplicate_trajectory(
    tmp_path: Path,
    sample_quantum_circuit: QuantumCircuit,
    sample_initial_metrics: CircuitMetrics,
    sample_final_metrics: CircuitMetrics,
) -> None:
    """Test that duplicate trajectories are skipped."""
    config = GridSearchConfig(
        database_path=tmp_path / "test.db",
        optimizers=["qiskit_standard"],
    )

    with GridSearchRunner(config) as runner:
        # Insert circuit
        circuit_record = CircuitRecord(
            id=None, name="test", category="qft", source="local", qasm_path=None,
            num_qubits=4, initial_depth=20, initial_two_qubit_gates=10,
            initial_two_qubit_depth=8, initial_total_gates=30, gate_density=7.5, two_qubit_ratio=0.333,
        )
        circuit_id = runner.db.insert_circuit(circuit_record)
        circuit_record = runner.db.get_circuit_by_id(circuit_id)
        assert circuit_record is not None

        # Create chain result
        mock_step_result = StepResult(
            step=ChainStep(runner_type="qiskit_standard", name="qiskit_standard"),
            step_index=0, input_metrics=sample_initial_metrics, output_metrics=sample_final_metrics,
            transpiled=TranspiledCircuit(
                optimizer="qiskit_standard", label="test",
                circuit=sample_quantum_circuit, metrics=sample_final_metrics,
            ),
            duration_seconds=1.0,
        )

        chain_result = ChainResult(
            chain_name="qiskit_standard",
            steps=[ChainStep(runner_type="qiskit_standard")],
            step_results=[mock_step_result],
            initial_circuit=sample_quantum_circuit, initial_metrics=sample_initial_metrics,
            final_circuit=sample_quantum_circuit, final_metrics=sample_final_metrics,
            total_duration_seconds=1.0,
        )

        # First record
        first_id = runner.record_trajectory(circuit_record, chain_result, ["qiskit_standard"])
        assert first_id > 0

        # Second record should be skipped
        second_id = runner.record_trajectory(circuit_record, chain_result, ["qiskit_standard"])
        assert second_id == -1  # Skip indicator

        assert runner.db.count_trajectories() == 1


# --- Tests for OPTIMIZER_CONFIGS ---


def test_optimizer_configs_have_required_fields() -> None:
    """Test that all optimizer configs have required fields."""
    for name, config in OPTIMIZER_CONFIGS.items():
        assert "runner_type" in config, f"Missing runner_type in {name}"
        assert "options" in config, f"Missing options in {name}"


def test_optimizer_configs_valid_runner_types() -> None:
    """Test that optimizer configs use valid runner types."""
    valid_types = {"wisq", "tket", "qiskit_ai", "qiskit_standard", "voqc"}

    for name, config in OPTIMIZER_CONFIGS.items():
        assert config["runner_type"] in valid_types, f"Invalid runner_type in {name}"


def test_optimizer_configs_count() -> None:
    """Test expected number of optimizer configs."""
    # Should have 5 optimizers as per plan (VOQC blocked but may be in config)
    assert len(OPTIMIZER_CONFIGS) >= 4


# --- Tests for GridSearchRunner._load_circuit ---


def test_load_circuit_success(tmp_path: Path) -> None:
    """Test _load_circuit loads a valid QASM file."""
    qasm_content = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n'
    qasm_file = tmp_path / "test.qasm"
    qasm_file.write_text(qasm_content)

    config = GridSearchConfig(
        database_path=tmp_path / "test.db",
        optimizers=["qiskit_standard"],
    )

    with GridSearchRunner(config) as runner:
        record = CircuitRecord(
            id=1, name="test", category="qft", source="local",
            qasm_path=str(qasm_file), num_qubits=2,
            initial_depth=2, initial_two_qubit_gates=1,
            initial_two_qubit_depth=1, initial_total_gates=2,
            gate_density=1.0, two_qubit_ratio=0.5,
        )
        circuit = runner._load_circuit(record)
        assert circuit is not None
        assert circuit.num_qubits == 2


def test_load_circuit_invalid_path(tmp_path: Path) -> None:
    """Test _load_circuit returns None for nonexistent path."""
    config = GridSearchConfig(
        database_path=tmp_path / "test.db",
        optimizers=["qiskit_standard"],
    )

    with GridSearchRunner(config) as runner:
        record = CircuitRecord(
            id=1, name="test", category="qft", source="local",
            qasm_path="/nonexistent/file.qasm", num_qubits=2,
            initial_depth=2, initial_two_qubit_gates=1,
            initial_two_qubit_depth=1, initial_total_gates=2,
            gate_density=1.0, two_qubit_ratio=0.5,
        )
        circuit = runner._load_circuit(record)
        assert circuit is None


def test_load_circuit_none_path(tmp_path: Path) -> None:
    """Test _load_circuit returns None when qasm_path is None."""
    config = GridSearchConfig(
        database_path=tmp_path / "test.db",
        optimizers=["qiskit_standard"],
    )

    with GridSearchRunner(config) as runner:
        record = CircuitRecord(
            id=1, name="test", category="qft", source="local",
            qasm_path=None, num_qubits=2,
            initial_depth=2, initial_two_qubit_gates=1,
            initial_two_qubit_depth=1, initial_total_gates=2,
            gate_density=1.0, two_qubit_ratio=0.5,
        )
        circuit = runner._load_circuit(record)
        assert circuit is None


# --- Tests for GridSearchRunner.run_exhaustive_search ---


def test_run_exhaustive_search_empty_db(tmp_path: Path) -> None:
    """Test run_exhaustive_search with empty database returns empty report."""
    config = GridSearchConfig(
        database_path=tmp_path / "test.db",
        optimizers=["qiskit_standard"],
    )

    with GridSearchRunner(config) as runner:
        report = runner.run_exhaustive_search()
        assert report.total_circuits == 0
        assert report.total_trajectories == 0
        assert report.total_steps == 0


@patch("benchmarks.ai_transpile.rl_trajectory.grid_search.execute_chain")
def test_run_exhaustive_search_with_mocked_chain(
    mock_execute: MagicMock,
    tmp_path: Path,
    sample_quantum_circuit: QuantumCircuit,
    sample_initial_metrics: CircuitMetrics,
    sample_final_metrics: CircuitMetrics,
) -> None:
    """Test run_exhaustive_search with one circuit and mocked chain."""
    from benchmarks.ai_transpile.chain_executor import ChainResult, ChainStep, StepResult
    from benchmarks.ai_transpile.transpilers import TranspiledCircuit

    mock_step_result = StepResult(
        step=ChainStep(runner_type="qiskit_standard", name="qiskit_standard"),
        step_index=0,
        input_metrics=sample_initial_metrics,
        output_metrics=sample_final_metrics,
        transpiled=TranspiledCircuit(
            optimizer="qiskit_standard", label="test",
            circuit=sample_quantum_circuit, metrics=sample_final_metrics,
        ),
        duration_seconds=1.0,
    )
    mock_chain_result = ChainResult(
        chain_name="qiskit_standard",
        steps=[ChainStep(runner_type="qiskit_standard")],
        step_results=[mock_step_result],
        initial_circuit=sample_quantum_circuit,
        initial_metrics=sample_initial_metrics,
        final_circuit=sample_quantum_circuit,
        final_metrics=sample_final_metrics,
        total_duration_seconds=1.0,
    )
    mock_execute.return_value = mock_chain_result

    config = GridSearchConfig(
        database_path=tmp_path / "test.db",
        optimizers=["qiskit_standard"],
        enable_chain_search=False,
    )

    with GridSearchRunner(config) as runner:
        # Insert circuit
        record = CircuitRecord(
            id=None, name="test", category="qft", source="local",
            qasm_path=None, num_qubits=4,
            initial_depth=20, initial_two_qubit_gates=10,
            initial_two_qubit_depth=8, initial_total_gates=30,
            gate_density=7.5, two_qubit_ratio=0.333,
        )
        runner.db.insert_circuit(record)

        with patch.object(runner, "_load_circuit", return_value=sample_quantum_circuit):
            report = runner.run_exhaustive_search(resume=False)

        assert report.total_circuits == 1
        assert report.total_trajectories == 1


@patch("benchmarks.ai_transpile.rl_trajectory.grid_search.execute_chain")
def test_run_exhaustive_search_progress_callback(
    mock_execute: MagicMock,
    tmp_path: Path,
    sample_quantum_circuit: QuantumCircuit,
    sample_initial_metrics: CircuitMetrics,
    sample_final_metrics: CircuitMetrics,
) -> None:
    """Test that progress callback is invoked during exhaustive search."""
    from benchmarks.ai_transpile.chain_executor import ChainResult, ChainStep, StepResult
    from benchmarks.ai_transpile.transpilers import TranspiledCircuit

    mock_step_result = StepResult(
        step=ChainStep(runner_type="qiskit_standard", name="qiskit_standard"),
        step_index=0,
        input_metrics=sample_initial_metrics,
        output_metrics=sample_final_metrics,
        transpiled=TranspiledCircuit(
            optimizer="qiskit_standard", label="test",
            circuit=sample_quantum_circuit, metrics=sample_final_metrics,
        ),
        duration_seconds=1.0,
    )
    mock_chain_result = ChainResult(
        chain_name="qiskit_standard",
        steps=[ChainStep(runner_type="qiskit_standard")],
        step_results=[mock_step_result],
        initial_circuit=sample_quantum_circuit,
        initial_metrics=sample_initial_metrics,
        final_circuit=sample_quantum_circuit,
        final_metrics=sample_final_metrics,
        total_duration_seconds=1.0,
    )
    mock_execute.return_value = mock_chain_result

    callback = MagicMock()
    config = GridSearchConfig(
        database_path=tmp_path / "test.db",
        optimizers=["qiskit_standard"],
        enable_chain_search=False,
    )

    with GridSearchRunner(config, progress_callback=callback) as runner:
        record = CircuitRecord(
            id=None, name="test", category="qft", source="local",
            qasm_path=None, num_qubits=4,
            initial_depth=20, initial_two_qubit_gates=10,
            initial_two_qubit_depth=8, initial_total_gates=30,
            gate_density=7.5, two_qubit_ratio=0.333,
        )
        runner.db.insert_circuit(record)

        with patch.object(runner, "_load_circuit", return_value=sample_quantum_circuit):
            runner.run_exhaustive_search(resume=False)

    assert callback.call_count >= 1


# --- Tests for run_quick_grid_search ---


def test_run_quick_grid_search_empty_db(tmp_path: Path) -> None:
    """Test run_quick_grid_search with empty database."""
    from benchmarks.ai_transpile.rl_trajectory.grid_search import run_quick_grid_search

    report = run_quick_grid_search(
        database_path=tmp_path / "test.db",
        optimizers=["qiskit_standard"],
        max_qubits=4,
    )
    assert report.total_circuits == 0
