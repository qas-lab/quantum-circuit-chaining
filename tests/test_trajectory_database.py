"""Tests for the trajectory database module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from benchmarks.ai_transpile.rl_trajectory import (
    CircuitRecord,
    OptimizerRecord,
    TrajectoryDatabase,
    TrajectoryStepRecord,
)

# --- Fixtures ---


@pytest.fixture
def db(tmp_path: Path) -> TrajectoryDatabase:
    """Create a temporary database."""
    db_path = tmp_path / "test_trajectories.db"
    return TrajectoryDatabase(db_path)


@pytest.fixture
def sample_circuit() -> CircuitRecord:
    """Create a sample circuit record."""
    return CircuitRecord(
        id=None,
        name="test_qft_4",
        category="qft",
        source="local",
        qasm_path="/path/to/qft_4.qasm",
        num_qubits=4,
        initial_depth=20,
        initial_two_qubit_gates=10,
        initial_two_qubit_depth=8,
        initial_total_gates=30,
        gate_density=7.5,
        two_qubit_ratio=0.333,
    )


@pytest.fixture
def sample_optimizer() -> OptimizerRecord:
    """Create a sample optimizer record."""
    return OptimizerRecord(
        id=None,
        name="wisq_rules",
        runner_type="wisq",
        options={"approx_epsilon": 0},
        description="WISQ rules-only",
    )


# --- Tests for Database Initialization ---


def test_database_creation(tmp_path: Path) -> None:
    """Test database is created with correct schema."""
    db_path = tmp_path / "test.db"
    db = TrajectoryDatabase(db_path)

    assert db_path.exists()
    db.close()


def test_database_context_manager(tmp_path: Path) -> None:
    """Test database can be used as context manager."""
    db_path = tmp_path / "test.db"

    with TrajectoryDatabase(db_path):
        assert db_path.exists()


# --- Tests for Circuit CRUD ---


def test_insert_circuit(db: TrajectoryDatabase, sample_circuit: CircuitRecord) -> None:
    """Test inserting a circuit record."""
    circuit_id = db.insert_circuit(sample_circuit)

    assert circuit_id > 0


def test_get_circuit_by_name(db: TrajectoryDatabase, sample_circuit: CircuitRecord) -> None:
    """Test retrieving a circuit by name."""
    db.insert_circuit(sample_circuit)

    retrieved = db.get_circuit_by_name(sample_circuit.name)

    assert retrieved is not None
    assert retrieved.name == sample_circuit.name
    assert retrieved.category == sample_circuit.category
    assert retrieved.num_qubits == sample_circuit.num_qubits


def test_get_circuit_by_id(db: TrajectoryDatabase, sample_circuit: CircuitRecord) -> None:
    """Test retrieving a circuit by ID."""
    circuit_id = db.insert_circuit(sample_circuit)

    retrieved = db.get_circuit_by_id(circuit_id)

    assert retrieved is not None
    assert retrieved.id == circuit_id
    assert retrieved.name == sample_circuit.name


def test_get_nonexistent_circuit(db: TrajectoryDatabase) -> None:
    """Test retrieving a circuit that doesn't exist."""
    result = db.get_circuit_by_name("nonexistent")
    assert result is None


def test_list_circuits(db: TrajectoryDatabase) -> None:
    """Test listing circuits with filters."""
    # Insert multiple circuits
    circuits = [
        CircuitRecord(
            id=None, name="qft_4", category="qft", source="local", qasm_path=None,
            num_qubits=4, initial_depth=20, initial_two_qubit_gates=10,
            initial_two_qubit_depth=8, initial_total_gates=30, gate_density=7.5, two_qubit_ratio=0.333
        ),
        CircuitRecord(
            id=None, name="qaoa_5", category="qaoa", source="benchpress", qasm_path=None,
            num_qubits=5, initial_depth=15, initial_two_qubit_gates=8,
            initial_two_qubit_depth=6, initial_total_gates=25, gate_density=5.0, two_qubit_ratio=0.32
        ),
        CircuitRecord(
            id=None, name="qft_10", category="qft", source="local", qasm_path=None,
            num_qubits=10, initial_depth=50, initial_two_qubit_gates=40,
            initial_two_qubit_depth=30, initial_total_gates=100, gate_density=10.0, two_qubit_ratio=0.4
        ),
    ]

    for c in circuits:
        db.insert_circuit(c)

    # Test no filters
    all_circuits = db.list_circuits()
    assert len(all_circuits) == 3

    # Test category filter
    qft_circuits = db.list_circuits(category="qft")
    assert len(qft_circuits) == 2

    # Test source filter
    local_circuits = db.list_circuits(source="local")
    assert len(local_circuits) == 2

    # Test max_qubits filter
    small_circuits = db.list_circuits(max_qubits=5)
    assert len(small_circuits) == 2


# --- Tests for Optimizer CRUD ---


def test_insert_optimizer(db: TrajectoryDatabase, sample_optimizer: OptimizerRecord) -> None:
    """Test inserting an optimizer record."""
    opt_id = db.insert_optimizer(sample_optimizer)

    assert opt_id > 0


def test_get_optimizer_by_name(db: TrajectoryDatabase, sample_optimizer: OptimizerRecord) -> None:
    """Test retrieving an optimizer by name."""
    db.insert_optimizer(sample_optimizer)

    retrieved = db.get_optimizer_by_name(sample_optimizer.name)

    assert retrieved is not None
    assert retrieved.name == sample_optimizer.name
    assert retrieved.runner_type == sample_optimizer.runner_type
    assert retrieved.options == sample_optimizer.options


def test_get_or_create_optimizer(db: TrajectoryDatabase, sample_optimizer: OptimizerRecord) -> None:
    """Test get_or_create_optimizer returns existing ID."""
    # First insert
    first_id = db.get_or_create_optimizer(sample_optimizer)

    # Second call should return same ID
    second_id = db.get_or_create_optimizer(sample_optimizer)

    assert first_id == second_id


def test_list_optimizers(db: TrajectoryDatabase) -> None:
    """Test listing all optimizers."""
    optimizers = [
        OptimizerRecord(id=None, name="opt1", runner_type="type1", options={}),
        OptimizerRecord(id=None, name="opt2", runner_type="type2", options={"key": "value"}),
    ]

    for opt in optimizers:
        db.insert_optimizer(opt)

    all_opts = db.list_optimizers()
    assert len(all_opts) == 2


# --- Tests for Trajectory CRUD ---


def test_insert_trajectory(
    db: TrajectoryDatabase,
    sample_circuit: CircuitRecord,
) -> None:
    """Test inserting a trajectory record."""
    circuit_id = db.insert_circuit(sample_circuit)

    trajectory_id = db.insert_trajectory(
        circuit_id=circuit_id,
        chain_name="wisq_rules_then_tket",
        num_steps=2,
        initial_depth=20,
        initial_two_qubit_gates=10,
        initial_two_qubit_depth=8,
        initial_total_gates=30,
        final_depth=15,
        final_two_qubit_gates=6,
        final_two_qubit_depth=5,
        final_total_gates=20,
        total_duration_seconds=5.5,
        total_reward=0.15,
        improvement_percentage=40.0,
    )

    assert trajectory_id > 0


def test_trajectory_exists(
    db: TrajectoryDatabase,
    sample_circuit: CircuitRecord,
) -> None:
    """Test checking if trajectory exists."""
    circuit_id = db.insert_circuit(sample_circuit)

    # Before insertion
    assert not db.trajectory_exists(circuit_id, "test_chain")

    # After insertion
    db.insert_trajectory(
        circuit_id=circuit_id,
        chain_name="test_chain",
        num_steps=1,
        initial_depth=20, initial_two_qubit_gates=10, initial_two_qubit_depth=8, initial_total_gates=30,
        final_depth=15, final_two_qubit_gates=6, final_two_qubit_depth=5, final_total_gates=20,
        total_duration_seconds=1.0, total_reward=0.1, improvement_percentage=40.0,
    )

    assert db.trajectory_exists(circuit_id, "test_chain")


# --- Tests for Trajectory Steps ---


def test_insert_trajectory_step(
    db: TrajectoryDatabase,
    sample_circuit: CircuitRecord,
    sample_optimizer: OptimizerRecord,
) -> None:
    """Test inserting a trajectory step."""
    circuit_id = db.insert_circuit(sample_circuit)
    opt_id = db.insert_optimizer(sample_optimizer)
    trajectory_id = db.insert_trajectory(
        circuit_id=circuit_id,
        chain_name="test_chain",
        num_steps=1,
        initial_depth=20, initial_two_qubit_gates=10, initial_two_qubit_depth=8, initial_total_gates=30,
        final_depth=15, final_two_qubit_gates=6, final_two_qubit_depth=5, final_total_gates=20,
        total_duration_seconds=1.0, total_reward=0.1, improvement_percentage=40.0,
    )

    step = TrajectoryStepRecord(
        trajectory_id=trajectory_id,
        step_index=0,
        optimizer_id=opt_id,
        state_depth=20,
        state_two_qubit_gates=10,
        state_two_qubit_depth=8,
        state_total_gates=30,
        state_num_qubits=4,
        state_gate_density=7.5,
        state_two_qubit_ratio=0.333,
        state_steps_taken=0,
        state_time_budget_remaining=300.0,
        state_category=[0.0] * 13,
        next_state_depth=15,
        next_state_two_qubit_gates=6,
        next_state_two_qubit_depth=5,
        next_state_total_gates=20,
        next_state_gate_density=5.0,
        next_state_two_qubit_ratio=0.3,
        next_state_steps_taken=1,
        next_state_time_budget_remaining=299.0,
        reward_improvement_only=0.4,
        reward_efficiency=0.35,
        reward_multi_objective=0.38,
        reward_sparse_final=0.0,
        done=True,
        duration_seconds=1.0,
    )

    step_id = db.insert_trajectory_step(step)
    assert step_id > 0


# --- Tests for RL Training Interface ---


def test_sample_batch_empty(db: TrajectoryDatabase) -> None:
    """Test sampling from empty database."""
    batch = db.sample_batch(batch_size=10)
    assert len(batch) == 0


def test_sample_batch(
    db: TrajectoryDatabase,
    sample_circuit: CircuitRecord,
    sample_optimizer: OptimizerRecord,
) -> None:
    """Test sampling a batch of SARS tuples."""
    # Insert test data
    circuit_id = db.insert_circuit(sample_circuit)
    opt_id = db.insert_optimizer(sample_optimizer)
    trajectory_id = db.insert_trajectory(
        circuit_id=circuit_id,
        chain_name="test_chain",
        num_steps=2,
        initial_depth=20, initial_two_qubit_gates=10, initial_two_qubit_depth=8, initial_total_gates=30,
        final_depth=10, final_two_qubit_gates=4, final_two_qubit_depth=3, final_total_gates=15,
        total_duration_seconds=2.0, total_reward=0.2, improvement_percentage=60.0,
    )

    category_encoding = [0.0] * 13
    category_encoding[0] = 1.0  # qft

    for i in range(2):
        step = TrajectoryStepRecord(
            trajectory_id=trajectory_id,
            step_index=i,
            optimizer_id=opt_id,
            state_depth=20 - i * 5, state_two_qubit_gates=10 - i * 3,
            state_two_qubit_depth=8 - i * 2, state_total_gates=30 - i * 7,
            state_num_qubits=4, state_gate_density=7.5 - i * 1.5, state_two_qubit_ratio=0.333,
            state_steps_taken=i, state_time_budget_remaining=300.0 - i,
            state_category=category_encoding,
            next_state_depth=15 - i * 5, next_state_two_qubit_gates=7 - i * 3,
            next_state_two_qubit_depth=6 - i * 2, next_state_total_gates=23 - i * 8,
            next_state_gate_density=5.75 - i * 2, next_state_two_qubit_ratio=0.304,
            next_state_steps_taken=i + 1, next_state_time_budget_remaining=299.0 - i,
            reward_improvement_only=0.3, reward_efficiency=0.25,
            reward_multi_objective=0.27, reward_sparse_final=0.0 if i == 0 else 0.6,
            done=(i == 1), duration_seconds=1.0,
        )
        db.insert_trajectory_step(step)

    # Sample batch
    batch = db.sample_batch(batch_size=5, seed=42)

    assert len(batch) == 2  # Only 2 steps in database
    for sars in batch:
        assert sars.state.shape == (9 + 13,)  # 9 base features + 13 categories
        assert isinstance(sars.action, (int, np.integer))
        assert isinstance(sars.reward, float)
        assert sars.next_state.shape == sars.state.shape
        assert isinstance(sars.done, bool)


def test_get_sars_tuples_iterator(
    db: TrajectoryDatabase,
    sample_circuit: CircuitRecord,
    sample_optimizer: OptimizerRecord,
) -> None:
    """Test iterating over SARS tuples."""
    # Insert test data
    circuit_id = db.insert_circuit(sample_circuit)
    opt_id = db.insert_optimizer(sample_optimizer)
    trajectory_id = db.insert_trajectory(
        circuit_id=circuit_id, chain_name="test", num_steps=1,
        initial_depth=20, initial_two_qubit_gates=10, initial_two_qubit_depth=8, initial_total_gates=30,
        final_depth=15, final_two_qubit_gates=6, final_two_qubit_depth=5, final_total_gates=20,
        total_duration_seconds=1.0, total_reward=0.1, improvement_percentage=40.0,
    )

    step = TrajectoryStepRecord(
        trajectory_id=trajectory_id, step_index=0, optimizer_id=opt_id,
        state_depth=20, state_two_qubit_gates=10, state_two_qubit_depth=8, state_total_gates=30,
        state_num_qubits=4, state_gate_density=7.5, state_two_qubit_ratio=0.333,
        state_steps_taken=0, state_time_budget_remaining=300.0, state_category=[0.0] * 13,
        next_state_depth=15, next_state_two_qubit_gates=6, next_state_two_qubit_depth=5, next_state_total_gates=20,
        next_state_gate_density=5.0, next_state_two_qubit_ratio=0.3,
        next_state_steps_taken=1, next_state_time_budget_remaining=299.0,
        reward_improvement_only=0.4, reward_efficiency=0.35, reward_multi_objective=0.38, reward_sparse_final=0.4,
        done=True, duration_seconds=1.0,
    )
    db.insert_trajectory_step(step)

    # Iterate
    tuples = list(db.get_sars_tuples(reward_type="reward_efficiency"))

    assert len(tuples) == 1
    assert tuples[0].reward == pytest.approx(0.35)


def test_export_to_d4rl_format(
    db: TrajectoryDatabase,
    sample_circuit: CircuitRecord,
    sample_optimizer: OptimizerRecord,
) -> None:
    """Test exporting to D4RL format."""
    # Insert test data
    circuit_id = db.insert_circuit(sample_circuit)
    opt_id = db.insert_optimizer(sample_optimizer)
    trajectory_id = db.insert_trajectory(
        circuit_id=circuit_id, chain_name="test", num_steps=1,
        initial_depth=20, initial_two_qubit_gates=10, initial_two_qubit_depth=8, initial_total_gates=30,
        final_depth=15, final_two_qubit_gates=6, final_two_qubit_depth=5, final_total_gates=20,
        total_duration_seconds=1.0, total_reward=0.1, improvement_percentage=40.0,
    )

    step = TrajectoryStepRecord(
        trajectory_id=trajectory_id, step_index=0, optimizer_id=opt_id,
        state_depth=20, state_two_qubit_gates=10, state_two_qubit_depth=8, state_total_gates=30,
        state_num_qubits=4, state_gate_density=7.5, state_two_qubit_ratio=0.333,
        state_steps_taken=0, state_time_budget_remaining=300.0, state_category=[0.0] * 13,
        next_state_depth=15, next_state_two_qubit_gates=6, next_state_two_qubit_depth=5, next_state_total_gates=20,
        next_state_gate_density=5.0, next_state_two_qubit_ratio=0.3,
        next_state_steps_taken=1, next_state_time_budget_remaining=299.0,
        reward_improvement_only=0.4, reward_efficiency=0.35, reward_multi_objective=0.38, reward_sparse_final=0.4,
        done=True, duration_seconds=1.0,
    )
    db.insert_trajectory_step(step)

    # Export
    data = db.export_to_d4rl_format()

    assert "observations" in data
    assert "actions" in data
    assert "rewards" in data
    assert "next_observations" in data
    assert "terminals" in data

    assert data["observations"].shape == (1, 22)  # 9 + 13
    assert data["actions"].shape == (1,)
    assert data["rewards"].shape == (1,)
    assert data["next_observations"].shape == (1, 22)
    assert data["terminals"].shape == (1,)


def test_export_empty_database(db: TrajectoryDatabase) -> None:
    """Test exporting empty database."""
    data = db.export_to_d4rl_format()

    assert len(data["observations"]) == 0
    assert len(data["actions"]) == 0


# --- Tests for Statistics ---


def test_get_statistics_empty(db: TrajectoryDatabase) -> None:
    """Test statistics on empty database."""
    stats = db.get_statistics()

    assert stats["num_circuits"] == 0
    assert stats["num_optimizers"] == 0
    assert stats["num_trajectories"] == 0
    assert stats["num_trajectory_steps"] == 0


def test_get_statistics(
    db: TrajectoryDatabase,
    sample_circuit: CircuitRecord,
    sample_optimizer: OptimizerRecord,
) -> None:
    """Test statistics with data."""
    circuit_id = db.insert_circuit(sample_circuit)
    db.insert_optimizer(sample_optimizer)
    db.insert_trajectory(
        circuit_id=circuit_id, chain_name="test", num_steps=2,
        initial_depth=20, initial_two_qubit_gates=10, initial_two_qubit_depth=8, initial_total_gates=30,
        final_depth=15, final_two_qubit_gates=6, final_two_qubit_depth=5, final_total_gates=20,
        total_duration_seconds=2.0, total_reward=0.2, improvement_percentage=40.0,
    )

    stats = db.get_statistics()

    assert stats["num_circuits"] == 1
    assert stats["num_optimizers"] == 1
    assert stats["num_trajectories"] == 1
    assert stats["circuits_by_category"]["qft"] == 1
    assert stats["avg_improvement_percentage"] == pytest.approx(40.0)


def test_count_methods(
    db: TrajectoryDatabase,
    sample_circuit: CircuitRecord,
    sample_optimizer: OptimizerRecord,
) -> None:
    """Test count methods."""
    assert db.count_trajectories() == 0
    assert db.count_trajectory_steps() == 0

    circuit_id = db.insert_circuit(sample_circuit)
    opt_id = db.insert_optimizer(sample_optimizer)
    trajectory_id = db.insert_trajectory(
        circuit_id=circuit_id, chain_name="test", num_steps=1,
        initial_depth=20, initial_two_qubit_gates=10, initial_two_qubit_depth=8, initial_total_gates=30,
        final_depth=15, final_two_qubit_gates=6, final_two_qubit_depth=5, final_total_gates=20,
        total_duration_seconds=1.0, total_reward=0.1, improvement_percentage=40.0,
    )

    assert db.count_trajectories() == 1

    step = TrajectoryStepRecord(
        trajectory_id=trajectory_id, step_index=0, optimizer_id=opt_id,
        state_depth=20, state_two_qubit_gates=10, state_two_qubit_depth=8, state_total_gates=30,
        state_num_qubits=4, state_gate_density=7.5, state_two_qubit_ratio=0.333,
        state_steps_taken=0, state_time_budget_remaining=300.0, state_category=[0.0] * 13,
        next_state_depth=15, next_state_two_qubit_gates=6, next_state_two_qubit_depth=5, next_state_total_gates=20,
        next_state_gate_density=5.0, next_state_two_qubit_ratio=0.3,
        next_state_steps_taken=1, next_state_time_budget_remaining=299.0,
        reward_improvement_only=0.4, reward_efficiency=0.35, reward_multi_objective=0.38, reward_sparse_final=0.4,
        done=True, duration_seconds=1.0,
    )
    db.insert_trajectory_step(step)

    assert db.count_trajectory_steps() == 1
