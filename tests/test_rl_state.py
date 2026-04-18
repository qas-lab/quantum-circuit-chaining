"""Tests for the RL state representation module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from benchmarks.ai_transpile.rl_trajectory.state import (
    CATEGORIES,
    CATEGORY_TO_INDEX,
    RLState,
    compute_circuit_features,
    get_category_encoding,
    normalize_state,
)
from benchmarks.ai_transpile.transpilers import CircuitMetrics
from qiskit import QuantumCircuit

# --- Fixtures ---


@pytest.fixture
def sample_metrics() -> CircuitMetrics:
    """Create sample circuit metrics."""
    return CircuitMetrics(
        depth=20,
        two_qubit_gates=10,
        two_qubit_depth=8,
        total_gates=30,
    )


@pytest.fixture
def zero_metrics() -> CircuitMetrics:
    """Create metrics with zero values for edge cases."""
    return CircuitMetrics(
        depth=0,
        two_qubit_gates=0,
        two_qubit_depth=0,
        total_gates=0,
    )


@pytest.fixture
def sample_circuit_2q() -> QuantumCircuit:
    """A simple 2-qubit circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def zero_qubit_circuit() -> QuantumCircuit:
    """A circuit with 0 qubits."""
    return QuantumCircuit(0)


# --- Tests for get_category_encoding ---


def test_get_category_encoding_known_category() -> None:
    """Test encoding for a known category returns correct one-hot vector."""
    encoding = get_category_encoding("qft")
    assert len(encoding) == len(CATEGORIES)
    assert encoding[CATEGORY_TO_INDEX["qft"]] == 1.0
    assert sum(encoding) == 1.0


def test_get_category_encoding_unknown_category() -> None:
    """Test encoding for an unknown category maps to 'unknown'."""
    encoding = get_category_encoding("nonexistent_category")
    assert len(encoding) == len(CATEGORIES)
    assert encoding[CATEGORY_TO_INDEX["unknown"]] == 1.0
    assert sum(encoding) == 1.0


def test_get_category_encoding_length() -> None:
    """Test that encoding length matches number of categories (14)."""
    encoding = get_category_encoding("qaoa")
    assert len(encoding) == 14


def test_get_category_encoding_all_categories() -> None:
    """Test that all known categories produce valid encodings."""
    for category in CATEGORIES:
        encoding = get_category_encoding(category)
        assert sum(encoding) == 1.0
        assert encoding[CATEGORY_TO_INDEX[category]] == 1.0


# --- Tests for RLState.from_circuit ---


@patch("benchmarks.ai_transpile.rl_trajectory.state.analyze_circuit")
def test_from_circuit_auto_computes_metrics(
    mock_analyze: object,
    sample_circuit_2q: QuantumCircuit,
    sample_metrics: CircuitMetrics,
) -> None:
    """Test that from_circuit computes metrics when not provided."""
    mock_analyze.return_value = sample_metrics  # type: ignore[union-attr]

    state = RLState.from_circuit(sample_circuit_2q)

    mock_analyze.assert_called_once_with(sample_circuit_2q)  # type: ignore[union-attr]
    assert state.depth == 20
    assert state.two_qubit_gates == 10
    assert state.total_gates == 30


def test_from_circuit_with_pre_computed_metrics(
    sample_circuit_2q: QuantumCircuit,
    sample_metrics: CircuitMetrics,
) -> None:
    """Test that from_circuit uses provided metrics."""
    state = RLState.from_circuit(sample_circuit_2q, metrics=sample_metrics)
    assert state.depth == 20
    assert state.total_gates == 30


def test_from_circuit_gate_density_zero_qubits(
    zero_qubit_circuit: QuantumCircuit,
    zero_metrics: CircuitMetrics,
) -> None:
    """Test gate_density is 0 when circuit has 0 qubits."""
    state = RLState.from_circuit(zero_qubit_circuit, metrics=zero_metrics)
    assert state.gate_density == 0.0


def test_from_circuit_two_qubit_ratio_zero_total_gates(
    sample_circuit_2q: QuantumCircuit,
    zero_metrics: CircuitMetrics,
) -> None:
    """Test two_qubit_ratio is 0 when total_gates is 0."""
    state = RLState.from_circuit(sample_circuit_2q, metrics=zero_metrics)
    assert state.two_qubit_ratio == 0.0


def test_from_circuit_category_encoding_propagation(
    sample_circuit_2q: QuantumCircuit,
    sample_metrics: CircuitMetrics,
) -> None:
    """Test that category encoding is properly set."""
    state = RLState.from_circuit(
        sample_circuit_2q, metrics=sample_metrics, category="qaoa"
    )
    expected = get_category_encoding("qaoa")
    assert state.category_encoding == expected


def test_from_circuit_gate_density_computed(
    sample_circuit_2q: QuantumCircuit,
    sample_metrics: CircuitMetrics,
) -> None:
    """Test gate_density is correctly computed."""
    state = RLState.from_circuit(sample_circuit_2q, metrics=sample_metrics)
    assert state.gate_density == pytest.approx(30.0 / 2.0)


def test_from_circuit_two_qubit_ratio_computed(
    sample_circuit_2q: QuantumCircuit,
    sample_metrics: CircuitMetrics,
) -> None:
    """Test two_qubit_ratio is correctly computed."""
    state = RLState.from_circuit(sample_circuit_2q, metrics=sample_metrics)
    assert state.two_qubit_ratio == pytest.approx(10.0 / 30.0)


# --- Tests for RLState.to_vector ---


def test_to_vector_dtype(sample_metrics: CircuitMetrics) -> None:
    """Test that to_vector returns float32 dtype."""
    state = RLState.from_metrics(metrics=sample_metrics, num_qubits=4)
    vector = state.to_vector()
    assert vector.dtype == np.float32


def test_to_vector_shape(sample_metrics: CircuitMetrics) -> None:
    """Test that to_vector returns shape (26,)."""
    state = RLState.from_metrics(metrics=sample_metrics, num_qubits=4)
    vector = state.to_vector()
    assert vector.shape == (26,)


def test_to_vector_correct_value_ordering(sample_metrics: CircuitMetrics) -> None:
    """Test that to_vector has correct value ordering: 12 features + 14 category."""
    state = RLState.from_metrics(
        metrics=sample_metrics,
        num_qubits=4,
        category="qft",
        steps_taken=2,
        time_budget_remaining=250.0,
    )
    vector = state.to_vector()

    assert vector[0] == pytest.approx(20.0)    # depth
    assert vector[1] == pytest.approx(10.0)    # two_qubit_gates
    assert vector[2] == pytest.approx(8.0)     # two_qubit_depth
    assert vector[3] == pytest.approx(30.0)    # total_gates
    assert vector[4] == pytest.approx(4.0)     # num_qubits
    assert vector[5] == pytest.approx(7.5)     # gate_density
    assert vector[6] == pytest.approx(10 / 30) # two_qubit_ratio
    assert vector[7] == pytest.approx(2.0)     # steps_taken
    assert vector[8] == pytest.approx(250.0)   # time_budget_remaining


def test_to_vector_category_encoding_positions(sample_metrics: CircuitMetrics) -> None:
    """Test that category encoding occupies positions [12:26]."""
    state = RLState.from_metrics(
        metrics=sample_metrics, num_qubits=4, category="qft"
    )
    vector = state.to_vector()
    category_part = vector[12:26]
    expected = np.array(get_category_encoding("qft"), dtype=np.float32)
    np.testing.assert_array_almost_equal(category_part, expected)


# --- Tests for compute_circuit_features ---


@patch("benchmarks.ai_transpile.rl_trajectory.state.analyze_circuit")
def test_compute_circuit_features_without_metrics(
    mock_analyze: object,
    sample_circuit_2q: QuantumCircuit,
    sample_metrics: CircuitMetrics,
) -> None:
    """Test compute_circuit_features auto-computes metrics."""
    mock_analyze.return_value = sample_metrics  # type: ignore[union-attr]

    features = compute_circuit_features(sample_circuit_2q)

    mock_analyze.assert_called_once_with(sample_circuit_2q)  # type: ignore[union-attr]
    assert "depth" in features
    assert "two_qubit_gates" in features
    assert len(features) == 7


def test_compute_circuit_features_with_metrics(
    sample_circuit_2q: QuantumCircuit,
    sample_metrics: CircuitMetrics,
) -> None:
    """Test compute_circuit_features uses provided metrics."""
    features = compute_circuit_features(sample_circuit_2q, metrics=sample_metrics)
    assert features["depth"] == 20.0
    assert features["two_qubit_gates"] == 10.0
    assert features["total_gates"] == 30.0
    assert features["num_qubits"] == 2.0
    assert features["gate_density"] == pytest.approx(15.0)
    assert features["two_qubit_ratio"] == pytest.approx(10 / 30)


def test_compute_circuit_features_zero_qubits(
    zero_qubit_circuit: QuantumCircuit,
    zero_metrics: CircuitMetrics,
) -> None:
    """Test gate_density is 0 with 0 qubits."""
    features = compute_circuit_features(zero_qubit_circuit, metrics=zero_metrics)
    assert features["gate_density"] == 0.0


def test_compute_circuit_features_zero_total_gates(
    sample_circuit_2q: QuantumCircuit,
    zero_metrics: CircuitMetrics,
) -> None:
    """Test two_qubit_ratio is 0 with 0 total gates."""
    features = compute_circuit_features(sample_circuit_2q, metrics=zero_metrics)
    assert features["two_qubit_ratio"] == 0.0


def test_compute_circuit_features_returns_7_keys(
    sample_circuit_2q: QuantumCircuit,
    sample_metrics: CircuitMetrics,
) -> None:
    """Test that exactly 7 keys are returned."""
    features = compute_circuit_features(sample_circuit_2q, metrics=sample_metrics)
    expected_keys = {
        "depth", "two_qubit_gates", "two_qubit_depth", "total_gates",
        "num_qubits", "gate_density", "two_qubit_ratio",
    }
    assert set(features.keys()) == expected_keys


# --- Tests for normalize_state ---


def test_normalize_state_default_params() -> None:
    """Test normalize_state with default (None) means and stds."""
    state = np.array(
        [50.0, 20.0, 15.0, 100.0, 10.0, 10.0, 0.2, 1.5, 150.0, 2.0, 0.5, 0.5]
        + [0.1] * len(CATEGORIES),
        dtype=np.float32,
    )
    result = normalize_state(state)
    # With default means matching the input, output should be ~0
    np.testing.assert_array_almost_equal(result, np.zeros_like(result), decimal=5)


def test_normalize_state_custom_params() -> None:
    """Test normalize_state with custom means and stds."""
    state = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    means = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    stds = np.array([5.0, 5.0, 5.0], dtype=np.float32)

    result = normalize_state(state, means=means, stds=stds)
    expected = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_normalize_state_division_by_zero_protection() -> None:
    """Test that stds=0 doesn't cause division by zero."""
    state = np.array([10.0, 20.0], dtype=np.float32)
    means = np.array([5.0, 10.0], dtype=np.float32)
    stds = np.array([0.0, 0.0], dtype=np.float32)

    result = normalize_state(state, means=means, stds=stds)
    # stds should be clamped to 1e-8, so result = (state - means) / 1e-8
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(5.0 / 1e-8)


def test_normalize_state_output_dtype() -> None:
    """Test that normalize_state preserves float dtype."""
    state = np.array(
        [50.0, 20.0, 15.0, 100.0, 10.0, 10.0, 0.2, 1.5, 150.0, 2.0, 0.5, 0.5]
        + [0.1] * len(CATEGORIES),
        dtype=np.float32,
    )
    result = normalize_state(state)
    assert result.dtype == np.float32
