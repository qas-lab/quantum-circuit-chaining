"""Tests for the abstract runner interface module."""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.ai_transpile.runner_interface import (
    CircuitRunner,
    FileBasedRunner,
    InMemoryRunner,
    RunnerConfig,
)
from benchmarks.ai_transpile.transpilers import TranspiledCircuit
from qiskit import QuantumCircuit

# --- Test Fixtures ---


@pytest.fixture
def runner_config(tmp_path: Path) -> RunnerConfig:
    """Create a runner config with a temporary output directory."""
    return RunnerConfig(output_dir=tmp_path, job_info="test_job")


# --- Tests for RunnerConfig ---


def test_runner_config_defaults() -> None:
    """Test RunnerConfig default values."""
    config = RunnerConfig()
    assert config.output_dir == Path("reports/runner_output")
    assert config.job_info == "runner"


def test_runner_config_custom_values(tmp_path: Path) -> None:
    """Test RunnerConfig with custom values."""
    config = RunnerConfig(output_dir=tmp_path, job_info="custom_job")
    assert config.output_dir == tmp_path
    assert config.job_info == "custom_job"


def test_runner_config_output_file_for(runner_config: RunnerConfig) -> None:
    """Test output_file_for generates correct paths."""
    circuit_path = Path("/some/path/my_circuit.qasm")
    output_path = runner_config.output_file_for(circuit_path)

    assert output_path.name == "my_circuit_test_job.qasm"
    assert output_path.parent == runner_config.output_dir


# --- Tests for CircuitRunner (abstract) ---


class ConcreteRunner(CircuitRunner):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, available: bool = True) -> None:
        self._available = available

    @property
    def name(self) -> str:
        return "concrete_test_runner"

    def run(
        self,
        circuit: QuantumCircuit | Path,
        config: RunnerConfig | None = None,
    ) -> list[TranspiledCircuit]:
        # Return empty list for testing
        return []

    def is_available(self) -> bool:
        return self._available


def test_circuit_runner_name() -> None:
    """Test that name property works."""
    runner = ConcreteRunner()
    assert runner.name == "concrete_test_runner"


def test_circuit_runner_is_available() -> None:
    """Test is_available method."""
    available_runner = ConcreteRunner(available=True)
    assert available_runner.is_available() is True

    unavailable_runner = ConcreteRunner(available=False)
    assert unavailable_runner.is_available() is False


def test_circuit_runner_get_availability_error_when_available() -> None:
    """Test get_availability_error returns None when available."""
    runner = ConcreteRunner(available=True)
    assert runner.get_availability_error() is None


def test_circuit_runner_get_availability_error_when_unavailable() -> None:
    """Test get_availability_error returns message when unavailable."""
    runner = ConcreteRunner(available=False)
    error = runner.get_availability_error()
    assert error is not None
    assert "concrete_test_runner" in error
    assert "not available" in error


def test_circuit_runner_analyze_and_create_result(sample_circuit: QuantumCircuit, tmp_path: Path) -> None:
    """Test _analyze_and_create_result helper method."""
    artifact_path = tmp_path / "test.qasm"
    artifact_path.write_text("dummy")

    result = CircuitRunner._analyze_and_create_result(
        circuit=sample_circuit,
        optimizer="test_optimizer",
        label="test_label",
        artifact_path=artifact_path,
        metadata={"custom_key": "custom_value"},
    )

    assert result.optimizer == "test_optimizer"
    assert result.label == "test_label"
    assert result.circuit is sample_circuit
    assert result.artifact_path == artifact_path
    assert result.metadata["custom_key"] == "custom_value"
    # Verify metrics are computed
    assert result.metrics.depth >= 0
    assert result.metrics.total_gates >= 0


def test_circuit_runner_analyze_and_create_result_no_metadata(sample_circuit: QuantumCircuit) -> None:
    """Test _analyze_and_create_result with no metadata."""
    result = CircuitRunner._analyze_and_create_result(
        circuit=sample_circuit,
        optimizer="test",
        label="test",
    )

    assert result.metadata == {}
    assert result.artifact_path is None


# --- Tests for FileBasedRunner ---


class ConcreteFileRunner(FileBasedRunner):
    """Concrete file-based runner for testing."""

    def __init__(self) -> None:
        self.run_from_path_called = False
        self.last_path: Path | None = None

    @property
    def name(self) -> str:
        return "file_runner"

    def is_available(self) -> bool:
        return True

    def _run_from_path(
        self,
        circuit_path: Path,
        config: RunnerConfig | None = None,
    ) -> list[TranspiledCircuit]:
        self.run_from_path_called = True
        self.last_path = circuit_path
        # Return empty list for testing
        return []


def test_file_based_runner_run_with_path(tmp_path: Path) -> None:
    """Test FileBasedRunner.run with a Path input."""
    circuit_path = tmp_path / "test.qasm"
    circuit_path.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n')

    runner = ConcreteFileRunner()
    runner.run(circuit_path)

    assert runner.run_from_path_called
    assert runner.last_path == circuit_path


def test_file_based_runner_run_with_circuit(sample_circuit: QuantumCircuit) -> None:
    """Test FileBasedRunner.run with a QuantumCircuit input (creates temp file)."""
    runner = ConcreteFileRunner()
    runner.run(sample_circuit)

    assert runner.run_from_path_called
    assert runner.last_path is not None
    # Temp file should have been created with .qasm extension
    assert runner.last_path.suffix == ".qasm"


# --- Tests for InMemoryRunner ---


class ConcreteInMemoryRunner(InMemoryRunner):
    """Concrete in-memory runner for testing."""

    def __init__(self) -> None:
        self.run_from_circuit_called = False
        self.last_circuit: QuantumCircuit | None = None

    @property
    def name(self) -> str:
        return "memory_runner"

    def is_available(self) -> bool:
        return True

    def _run_from_circuit(
        self,
        circuit: QuantumCircuit,
        config: RunnerConfig | None = None,
    ) -> list[TranspiledCircuit]:
        self.run_from_circuit_called = True
        self.last_circuit = circuit
        return []


def test_in_memory_runner_run_with_circuit(sample_circuit: QuantumCircuit) -> None:
    """Test InMemoryRunner.run with a QuantumCircuit input."""
    runner = ConcreteInMemoryRunner()
    runner.run(sample_circuit)

    assert runner.run_from_circuit_called
    assert runner.last_circuit is sample_circuit


def test_in_memory_runner_run_with_path(tmp_path: Path, sample_qasm_content: str) -> None:
    """Test InMemoryRunner.run with a Path input (loads circuit)."""
    circuit_path = tmp_path / "test.qasm"
    circuit_path.write_text(sample_qasm_content)

    runner = ConcreteInMemoryRunner()
    runner.run(circuit_path)

    assert runner.run_from_circuit_called
    assert runner.last_circuit is not None
    assert runner.last_circuit.num_qubits == 2


# --- Edge Case Tests ---


def test_file_based_runner_temp_file_cleanup(sample_circuit: QuantumCircuit) -> None:
    """Test that temp files are cleaned up after FileBasedRunner.run."""
    runner = ConcreteFileRunner()
    runner.run(sample_circuit)

    # The temp file should have been deleted
    assert runner.last_path is not None
    assert not runner.last_path.exists()

