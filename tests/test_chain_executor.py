"""Tests for chain executor module."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from benchmarks.ai_transpile.chain_executor import (
    ChainResult,
    ChainStep,
    StepResult,
    create_chain_from_config,
    execute_chain,
)
from benchmarks.ai_transpile.transpilers import CircuitMetrics, TranspiledCircuit
from qiskit import QuantumCircuit

# --- Fixtures ---


@pytest.fixture
def simple_circuit() -> QuantumCircuit:
    """Create a simple test circuit."""
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    return circuit


@pytest.fixture
def sample_metrics() -> CircuitMetrics:
    """Create sample circuit metrics."""
    return CircuitMetrics(
        depth=10,
        two_qubit_gates=5,
        two_qubit_depth=5,
        total_gates=20,
    )


@pytest.fixture
def improved_metrics() -> CircuitMetrics:
    """Create improved circuit metrics (after optimization)."""
    return CircuitMetrics(
        depth=8,
        two_qubit_gates=3,
        two_qubit_depth=3,
        total_gates=15,
    )


@pytest.fixture
def mock_transpiled_circuit(
    simple_circuit: QuantumCircuit, improved_metrics: CircuitMetrics
) -> TranspiledCircuit:
    """Create a mock transpiled circuit result."""
    return TranspiledCircuit(
        optimizer="mock",
        label="mock_result",
        circuit=simple_circuit,
        metrics=improved_metrics,
        artifact_path=None,
        metadata={},
    )


# --- Tests for ChainStep ---


def test_chain_step_creation() -> None:
    """Test ChainStep dataclass creation."""
    step = ChainStep(runner_type="wisq", options={"approx_epsilon": 0})

    assert step.runner_type == "wisq"
    assert step.options["approx_epsilon"] == 0
    assert step.name is None
    assert step.step_name == "wisq"


def test_chain_step_with_name() -> None:
    """Test ChainStep with custom name."""
    step = ChainStep(
        runner_type="wisq",
        options={"approx_epsilon": 0},
        name="wisq_rules_only",
    )

    assert step.runner_type == "wisq"
    assert step.name == "wisq_rules_only"
    assert step.step_name == "wisq_rules_only"


def test_chain_step_default_options() -> None:
    """Test ChainStep with default empty options."""
    step = ChainStep(runner_type="tket")

    assert step.runner_type == "tket"
    assert step.options == {}


# --- Tests for StepResult ---


def test_step_result_creation(
    sample_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
    mock_transpiled_circuit: TranspiledCircuit,
) -> None:
    """Test StepResult dataclass creation."""
    step = ChainStep(runner_type="tket")
    result = StepResult(
        step=step,
        step_index=0,
        input_metrics=sample_metrics,
        output_metrics=improved_metrics,
        transpiled=mock_transpiled_circuit,
        duration_seconds=1.5,
        artifact_path=Path("/tmp/test.qasm"),
    )

    assert result.step.runner_type == "tket"
    assert result.step_index == 0
    assert result.input_metrics.two_qubit_gates == 5
    assert result.output_metrics.two_qubit_gates == 3
    assert result.duration_seconds == 1.5
    assert result.artifact_path == Path("/tmp/test.qasm")


# --- Tests for ChainResult ---


def test_chain_result_creation(
    simple_circuit: QuantumCircuit,
    sample_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
    mock_transpiled_circuit: TranspiledCircuit,
) -> None:
    """Test ChainResult dataclass creation."""
    steps = [ChainStep(runner_type="tket")]
    step_results = [
        StepResult(
            step=steps[0],
            step_index=0,
            input_metrics=sample_metrics,
            output_metrics=improved_metrics,
            transpiled=mock_transpiled_circuit,
            duration_seconds=1.5,
        )
    ]

    result = ChainResult(
        chain_name="test_chain",
        steps=steps,
        step_results=step_results,
        initial_circuit=simple_circuit,
        initial_metrics=sample_metrics,
        final_circuit=simple_circuit,
        final_metrics=improved_metrics,
        total_duration_seconds=1.5,
    )

    assert result.chain_name == "test_chain"
    assert len(result.steps) == 1
    assert len(result.step_results) == 1
    assert result.total_duration_seconds == 1.5


def test_chain_result_intermediate_circuits(
    simple_circuit: QuantumCircuit,
    sample_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
    mock_transpiled_circuit: TranspiledCircuit,
) -> None:
    """Test ChainResult.intermediate_circuits property."""
    steps = [ChainStep(runner_type="tket"), ChainStep(runner_type="wisq")]
    step_results = [
        StepResult(
            step=steps[0],
            step_index=0,
            input_metrics=sample_metrics,
            output_metrics=improved_metrics,
            transpiled=mock_transpiled_circuit,
            duration_seconds=1.0,
        ),
        StepResult(
            step=steps[1],
            step_index=1,
            input_metrics=improved_metrics,
            output_metrics=improved_metrics,
            transpiled=mock_transpiled_circuit,
            duration_seconds=0.5,
        ),
    ]

    result = ChainResult(
        chain_name="test_chain",
        steps=steps,
        step_results=step_results,
        initial_circuit=simple_circuit,
        initial_metrics=sample_metrics,
        final_circuit=simple_circuit,
        final_metrics=improved_metrics,
        total_duration_seconds=1.5,
    )

    intermediates = result.intermediate_circuits
    assert len(intermediates) == 2
    assert all(isinstance(c, QuantumCircuit) for c in intermediates)


def test_chain_result_intermediate_metrics(
    simple_circuit: QuantumCircuit,
    sample_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
    mock_transpiled_circuit: TranspiledCircuit,
) -> None:
    """Test ChainResult.intermediate_metrics property."""
    steps = [ChainStep(runner_type="tket")]
    step_results = [
        StepResult(
            step=steps[0],
            step_index=0,
            input_metrics=sample_metrics,
            output_metrics=improved_metrics,
            transpiled=mock_transpiled_circuit,
            duration_seconds=1.0,
        )
    ]

    result = ChainResult(
        chain_name="test_chain",
        steps=steps,
        step_results=step_results,
        initial_circuit=simple_circuit,
        initial_metrics=sample_metrics,
        final_circuit=simple_circuit,
        final_metrics=improved_metrics,
        total_duration_seconds=1.0,
    )

    metrics = result.intermediate_metrics
    assert len(metrics) == 1
    assert metrics[0].two_qubit_gates == 3


def test_chain_result_improvement_percentage(
    simple_circuit: QuantumCircuit,
    sample_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
    mock_transpiled_circuit: TranspiledCircuit,
) -> None:
    """Test ChainResult.improvement_percentage method."""
    steps = [ChainStep(runner_type="tket")]
    step_results = [
        StepResult(
            step=steps[0],
            step_index=0,
            input_metrics=sample_metrics,
            output_metrics=improved_metrics,
            transpiled=mock_transpiled_circuit,
            duration_seconds=1.0,
        )
    ]

    result = ChainResult(
        chain_name="test_chain",
        steps=steps,
        step_results=step_results,
        initial_circuit=simple_circuit,
        initial_metrics=sample_metrics,
        final_circuit=simple_circuit,
        final_metrics=improved_metrics,
        total_duration_seconds=1.0,
    )

    # Initial: 5 two_qubit_gates, Final: 3 two_qubit_gates
    # Improvement: (5-3)/5 * 100 = 40%
    improvement = result.improvement_percentage("two_qubit_gates")
    assert improvement == pytest.approx(40.0)


def test_chain_result_improvement_percentage_zero_initial(
    simple_circuit: QuantumCircuit,
    mock_transpiled_circuit: TranspiledCircuit,
) -> None:
    """Test improvement_percentage when initial metric is zero."""
    zero_metrics = CircuitMetrics(depth=0, two_qubit_gates=0, two_qubit_depth=0, total_gates=0)
    steps = [ChainStep(runner_type="tket")]
    step_results = [
        StepResult(
            step=steps[0],
            step_index=0,
            input_metrics=zero_metrics,
            output_metrics=zero_metrics,
            transpiled=mock_transpiled_circuit,
            duration_seconds=1.0,
        )
    ]

    result = ChainResult(
        chain_name="test_chain",
        steps=steps,
        step_results=step_results,
        initial_circuit=simple_circuit,
        initial_metrics=zero_metrics,
        final_circuit=simple_circuit,
        final_metrics=zero_metrics,
        total_duration_seconds=1.0,
    )

    improvement = result.improvement_percentage("two_qubit_gates")
    assert improvement == 0.0


def test_chain_result_to_dict(
    simple_circuit: QuantumCircuit,
    sample_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
    mock_transpiled_circuit: TranspiledCircuit,
    tmp_path: Path,
) -> None:
    """Test ChainResult.to_dict serialization."""
    steps = [ChainStep(runner_type="tket", options={"gate_set": "IBMN"}, name="tket_step")]
    artifact_path = tmp_path / "test.qasm"
    step_results = [
        StepResult(
            step=steps[0],
            step_index=0,
            input_metrics=sample_metrics,
            output_metrics=improved_metrics,
            transpiled=mock_transpiled_circuit,
            duration_seconds=1.5,
            artifact_path=artifact_path,
        )
    ]

    result = ChainResult(
        chain_name="test_chain",
        steps=steps,
        step_results=step_results,
        initial_circuit=simple_circuit,
        initial_metrics=sample_metrics,
        final_circuit=simple_circuit,
        final_metrics=improved_metrics,
        total_duration_seconds=1.5,
        metadata={"key": "value"},
    )

    result_dict = result.to_dict()

    assert result_dict["chain_name"] == "test_chain"
    assert len(result_dict["steps"]) == 1
    assert result_dict["steps"][0]["runner_type"] == "tket"
    assert result_dict["steps"][0]["options"]["gate_set"] == "IBMN"
    assert result_dict["initial_metrics"]["two_qubit_gates"] == 5
    assert result_dict["final_metrics"]["two_qubit_gates"] == 3
    assert result_dict["step_results"][0]["artifact_path"] == str(artifact_path)
    assert result_dict["metadata"]["key"] == "value"


# --- Tests for create_chain_from_config ---


def test_create_chain_from_config_basic() -> None:
    """Test creating chain steps from configuration."""
    config: list[dict[str, Any]] = [
        {"type": "wisq", "approx_epsilon": 0},
        {"type": "tket", "gate_set": "IBMN"},
    ]

    steps = create_chain_from_config(config)

    assert len(steps) == 2
    assert steps[0].runner_type == "wisq"
    assert steps[0].options["approx_epsilon"] == 0
    assert steps[1].runner_type == "tket"
    assert steps[1].options["gate_set"] == "IBMN"


def test_create_chain_from_config_with_names() -> None:
    """Test creating chain steps with custom names."""
    config: list[dict[str, Any]] = [
        {"type": "wisq", "name": "wisq_rules", "approx_epsilon": 0},
    ]

    steps = create_chain_from_config(config)

    assert len(steps) == 1
    assert steps[0].runner_type == "wisq"
    assert steps[0].name == "wisq_rules"
    assert steps[0].step_name == "wisq_rules"


def test_create_chain_from_config_missing_type() -> None:
    """Test that missing type raises ValueError."""
    config: list[dict[str, Any]] = [
        {"name": "invalid_step", "option": "value"},
    ]

    with pytest.raises(ValueError, match="must have a 'type' field"):
        create_chain_from_config(config)


def test_create_chain_from_config_empty() -> None:
    """Test creating chain from empty config."""
    config: list[dict[str, Any]] = []
    steps = create_chain_from_config(config)
    assert steps == []


# --- Tests for execute_chain ---


def test_execute_chain_empty_steps(simple_circuit: QuantumCircuit) -> None:
    """Test that empty steps raises ValueError."""
    with pytest.raises(ValueError, match="at least one step"):
        execute_chain(simple_circuit, steps=[])


@patch("benchmarks.ai_transpile.chain_executor._execute_qiskit_standard_step")
def test_execute_chain_single_step(
    mock_qiskit_standard: MagicMock,
    simple_circuit: QuantumCircuit,
    improved_metrics: CircuitMetrics,
    tmp_path: Path,
) -> None:
    """Test executing a single-step chain with mocked optimizer."""
    # Setup mock
    mock_result = TranspiledCircuit(
        optimizer="qiskit_standard",
        label="qiskit_opt_level_3",
        circuit=simple_circuit,
        metrics=improved_metrics,
        metadata={"optimization_level": 3},
    )
    mock_qiskit_standard.return_value = mock_result

    steps = [ChainStep(runner_type="qiskit_standard", options={"optimization_levels": [3]})]

    result = execute_chain(
        simple_circuit,
        steps=steps,
        chain_name="single_step_test",
        output_dir=tmp_path,
    )

    assert result.chain_name == "single_step_test"
    assert len(result.step_results) == 1
    assert result.final_metrics == improved_metrics
    mock_qiskit_standard.assert_called_once()


@patch("benchmarks.ai_transpile.chain_executor._execute_qiskit_standard_step")
@patch("benchmarks.ai_transpile.chain_executor._execute_qiskit_ai_step")
def test_execute_chain_two_steps(
    mock_qiskit_ai: MagicMock,
    mock_qiskit_standard: MagicMock,
    simple_circuit: QuantumCircuit,
    sample_metrics: CircuitMetrics,
    improved_metrics: CircuitMetrics,
    tmp_path: Path,
) -> None:
    """Test executing a two-step chain with mocked optimizers."""
    # Intermediate result (after first step)
    intermediate_metrics = CircuitMetrics(
        depth=9,
        two_qubit_gates=4,
        two_qubit_depth=4,
        total_gates=18,
    )
    intermediate_result = TranspiledCircuit(
        optimizer="qiskit_standard",
        label="qiskit_opt_level_3",
        circuit=simple_circuit,
        metrics=intermediate_metrics,
    )

    # Final result (after second step)
    final_result = TranspiledCircuit(
        optimizer="qiskit_ai",
        label="ai_level_3_iter_1",
        circuit=simple_circuit,
        metrics=improved_metrics,
    )

    mock_qiskit_standard.return_value = intermediate_result
    mock_qiskit_ai.return_value = final_result

    steps = [
        ChainStep(runner_type="qiskit_standard", options={"optimization_levels": [3]}),
        ChainStep(runner_type="qiskit_ai", options={"optimization_levels": [3]}),
    ]

    result = execute_chain(
        simple_circuit,
        steps=steps,
        chain_name="two_step_test",
        output_dir=tmp_path,
    )

    assert result.chain_name == "two_step_test"
    assert len(result.step_results) == 2

    # Check first step
    assert result.step_results[0].step.runner_type == "qiskit_standard"
    assert result.step_results[0].output_metrics.two_qubit_gates == 4

    # Check second step
    assert result.step_results[1].step.runner_type == "qiskit_ai"
    assert result.step_results[1].input_metrics.two_qubit_gates == 4
    assert result.step_results[1].output_metrics.two_qubit_gates == 3

    # Check final metrics
    assert result.final_metrics.two_qubit_gates == 3


@patch("benchmarks.ai_transpile.chain_executor._execute_qiskit_standard_step")
def test_execute_chain_saves_intermediates(
    mock_qiskit_standard: MagicMock,
    simple_circuit: QuantumCircuit,
    improved_metrics: CircuitMetrics,
    tmp_path: Path,
) -> None:
    """Test that intermediate QASM files are saved."""
    mock_result = TranspiledCircuit(
        optimizer="qiskit_standard",
        label="qiskit_opt_level_3",
        circuit=simple_circuit,
        metrics=improved_metrics,
    )
    mock_qiskit_standard.return_value = mock_result

    steps = [ChainStep(runner_type="qiskit_standard")]

    result = execute_chain(
        simple_circuit,
        steps=steps,
        chain_name="save_test",
        output_dir=tmp_path,
        save_intermediates=True,
    )

    assert result.step_results[0].artifact_path is not None
    assert result.step_results[0].artifact_path.exists()


@patch("benchmarks.ai_transpile.chain_executor._execute_qiskit_standard_step")
def test_execute_chain_no_save_intermediates(
    mock_qiskit_standard: MagicMock,
    simple_circuit: QuantumCircuit,
    improved_metrics: CircuitMetrics,
    tmp_path: Path,
) -> None:
    """Test that intermediates are not saved when disabled."""
    mock_result = TranspiledCircuit(
        optimizer="qiskit_standard",
        label="qiskit_opt_level_3",
        circuit=simple_circuit,
        metrics=improved_metrics,
    )
    mock_qiskit_standard.return_value = mock_result

    steps = [ChainStep(runner_type="qiskit_standard")]

    result = execute_chain(
        simple_circuit,
        steps=steps,
        chain_name="no_save_test",
        output_dir=tmp_path,
        save_intermediates=False,
    )

    assert result.step_results[0].artifact_path is None


def test_execute_chain_unknown_runner(
    simple_circuit: QuantumCircuit,
    tmp_path: Path,
) -> None:
    """Test that unknown runner type raises ValueError."""
    steps = [ChainStep(runner_type="unknown_optimizer")]

    with pytest.raises(ValueError, match="Unknown runner type"):
        execute_chain(simple_circuit, steps=steps, output_dir=tmp_path)


@patch("benchmarks.ai_transpile.chain_executor._execute_qiskit_standard_step")
def test_execute_chain_from_path(
    mock_qiskit_standard: MagicMock,
    simple_circuit: QuantumCircuit,
    improved_metrics: CircuitMetrics,
    tmp_path: Path,
) -> None:
    """Test executing chain from a QASM file path."""
    # Save circuit to file
    from qiskit import qasm2

    qasm_path = tmp_path / "input.qasm"
    qasm_path.write_text(qasm2.dumps(simple_circuit))

    mock_result = TranspiledCircuit(
        optimizer="qiskit_standard",
        label="qiskit_opt_level_3",
        circuit=simple_circuit,
        metrics=improved_metrics,
    )
    mock_qiskit_standard.return_value = mock_result

    steps = [ChainStep(runner_type="qiskit_standard")]

    result = execute_chain(
        qasm_path,
        steps=steps,
        chain_name="path_test",
        output_dir=tmp_path / "output",
    )

    assert result.chain_name == "path_test"
    assert len(result.step_results) == 1


@patch("benchmarks.ai_transpile.chain_executor._execute_qiskit_standard_step")
def test_execute_chain_duration_tracking(
    mock_qiskit_standard: MagicMock,
    simple_circuit: QuantumCircuit,
    improved_metrics: CircuitMetrics,
    tmp_path: Path,
) -> None:
    """Test that durations are tracked correctly."""
    import time

    def slow_mock(circuit: QuantumCircuit, options: Any) -> TranspiledCircuit:
        time.sleep(0.1)  # Small delay to ensure measurable duration
        return TranspiledCircuit(
            optimizer="qiskit_standard",
            label="qiskit_opt_level_3",
            circuit=circuit,
            metrics=improved_metrics,
        )

    mock_qiskit_standard.side_effect = slow_mock

    steps = [ChainStep(runner_type="qiskit_standard")]

    result = execute_chain(
        simple_circuit,
        steps=steps,
        chain_name="duration_test",
        output_dir=tmp_path,
    )

    # Check that duration is positive
    assert result.total_duration_seconds > 0
    assert result.step_results[0].duration_seconds > 0


# --- Integration Tests (using real in-memory optimizers) ---


class TestChainExecutorIntegration:
    """Integration tests that use real (in-memory) optimizers."""

    def test_qiskit_standard_chain(self, simple_circuit: QuantumCircuit, tmp_path: Path) -> None:
        """Test chain with real Qiskit standard transpiler."""
        steps = [
            ChainStep(
                runner_type="qiskit_standard",
                options={"optimization_levels": [1]},
                name="qiskit_opt1",
            ),
            ChainStep(
                runner_type="qiskit_standard",
                options={"optimization_levels": [3]},
                name="qiskit_opt3",
            ),
        ]

        result = execute_chain(
            simple_circuit,
            steps=steps,
            chain_name="qiskit_standard_chain",
            output_dir=tmp_path,
        )

        # Basic sanity checks
        assert result.chain_name == "qiskit_standard_chain"
        assert len(result.step_results) == 2
        assert result.final_circuit is not None
        assert result.final_metrics.total_gates > 0

        # Check that intermediate files were saved
        for sr in result.step_results:
            assert sr.artifact_path is not None
            assert sr.artifact_path.exists()

    def test_qiskit_standard_then_ai_chain(
        self, simple_circuit: QuantumCircuit, tmp_path: Path
    ) -> None:
        """Test chain with Qiskit standard then AI transpiler."""
        steps = [
            ChainStep(
                runner_type="qiskit_standard",
                options={"optimization_levels": [1]},
            ),
            ChainStep(
                runner_type="qiskit_ai",
                options={"optimization_levels": [1], "iterations_per_level": 1},
            ),
        ]

        result = execute_chain(
            simple_circuit,
            steps=steps,
            chain_name="standard_then_ai",
            output_dir=tmp_path,
        )

        assert len(result.step_results) == 2
        assert result.step_results[0].step.runner_type == "qiskit_standard"
        assert result.step_results[1].step.runner_type == "qiskit_ai"

    def test_chain_metrics_progression(
        self, simple_circuit: QuantumCircuit, tmp_path: Path
    ) -> None:
        """Test that metrics are tracked correctly through chain."""
        steps = [
            ChainStep(
                runner_type="qiskit_standard",
                options={"optimization_levels": [3]},
            ),
        ]

        result = execute_chain(
            simple_circuit,
            steps=steps,
            chain_name="metrics_test",
            output_dir=tmp_path,
        )

        # Initial metrics should reflect the original circuit
        assert result.initial_metrics.total_gates == simple_circuit.size()

        # Step result input should match initial
        assert result.step_results[0].input_metrics == result.initial_metrics

        # Final metrics should match last step output
        assert result.final_metrics == result.step_results[-1].output_metrics

    def test_chain_result_serialization(
        self, simple_circuit: QuantumCircuit, tmp_path: Path
    ) -> None:
        """Test that chain result can be serialized to JSON."""
        import json

        steps = [
            ChainStep(
                runner_type="qiskit_standard",
                options={"optimization_levels": [1]},
            ),
        ]

        result = execute_chain(
            simple_circuit,
            steps=steps,
            chain_name="serialization_test",
            output_dir=tmp_path,
        )

        # Should be JSON-serializable
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)

        assert parsed["chain_name"] == "serialization_test"
        assert len(parsed["step_results"]) == 1




