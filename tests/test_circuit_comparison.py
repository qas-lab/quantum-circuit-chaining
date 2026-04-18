from unittest.mock import MagicMock, patch

import pytest
from benchmarks.ai_transpile.circuit_comparison import (
    compare_against_baseline,
    compare_circuits,
    compare_circuits_operator,
    compare_circuits_qcec,
    compare_circuits_statevector,
)
from qiskit import QuantumCircuit

# Note: circuit_h, circuit_h_h, circuit_i, circuit_x fixtures are now in conftest.py


# --- Tests for compare_circuits_qcec ---


def test_compare_circuits_qcec_success(circuit_h: QuantumCircuit) -> None:
    # Mock qcec
    mock_result = MagicMock()
    mock_result.considered_equivalent = True
    mock_result.equivalence = "equivalent"

    with patch("benchmarks.ai_transpile.circuit_comparison.qcec") as mock_qcec:
        mock_qcec.verify.return_value = mock_result

        result = compare_circuits_qcec(circuit_h, circuit_h)
        assert result.equivalent
        assert result.method == "qcec"
        assert result.details is not None
        assert result.details["equivalence"] == "equivalent"


def test_compare_circuits_qcec_not_available(circuit_h: QuantumCircuit) -> None:
    with patch("benchmarks.ai_transpile.circuit_comparison.qcec", None):
        result = compare_circuits_qcec(circuit_h, circuit_h)
        assert not result.equivalent
        assert result.method == "failed"
        assert result.error is not None
        assert "not available" in result.error


def test_compare_circuits_qcec_failure(circuit_h: QuantumCircuit) -> None:
    with patch("benchmarks.ai_transpile.circuit_comparison.qcec") as mock_qcec:
        mock_qcec.verify.side_effect = Exception("QCEC error")

        result = compare_circuits_qcec(circuit_h, circuit_h)
        assert not result.equivalent
        assert result.method == "failed"
        assert result.error is not None
        assert "QCEC check failed" in result.error


# --- Tests for compare_circuits_operator ---


def test_compare_circuits_operator_equivalent(circuit_h_h: QuantumCircuit, circuit_i: QuantumCircuit) -> None:
    result = compare_circuits_operator(circuit_h_h, circuit_i)
    assert result.equivalent
    assert result.method == "operator"
    assert result.fidelity is not None
    assert result.fidelity > 0.99


def test_compare_circuits_operator_not_equivalent(circuit_i: QuantumCircuit, circuit_x: QuantumCircuit) -> None:
    result = compare_circuits_operator(circuit_i, circuit_x)
    assert not result.equivalent
    assert result.method == "operator"
    assert result.fidelity is not None
    assert result.fidelity < 0.01


def test_compare_circuits_operator_qubit_mismatch(circuit_i: QuantumCircuit) -> None:
    qc2 = QuantumCircuit(2)
    result = compare_circuits_operator(circuit_i, qc2)
    assert not result.equivalent
    assert result.error is not None
    assert "Different qubit counts" in result.error


def test_compare_circuits_operator_failure(circuit_i: QuantumCircuit) -> None:
    with patch("benchmarks.ai_transpile.circuit_comparison.Operator", side_effect=Exception("Op error")):
        result = compare_circuits_operator(circuit_i, circuit_i)
        assert not result.equivalent
        assert result.error is not None
        assert "Operator comparison failed" in result.error


# --- Tests for compare_circuits_statevector ---


def test_compare_circuits_statevector_equivalent(circuit_h_h: QuantumCircuit, circuit_i: QuantumCircuit) -> None:
    result = compare_circuits_statevector(circuit_h_h, circuit_i)
    assert result.equivalent
    assert result.method == "statevector"
    assert result.fidelity is not None
    assert result.fidelity > 0.99


def test_compare_circuits_statevector_not_equivalent(circuit_i: QuantumCircuit, circuit_x: QuantumCircuit) -> None:
    result = compare_circuits_statevector(circuit_i, circuit_x)
    assert not result.equivalent
    assert result.method == "statevector"
    assert result.fidelity is not None
    assert result.fidelity < 0.01


def test_compare_circuits_statevector_qubit_mismatch(circuit_i: QuantumCircuit) -> None:
    qc2 = QuantumCircuit(2)
    result = compare_circuits_statevector(circuit_i, qc2)
    assert not result.equivalent
    assert result.error is not None
    assert "Different qubit counts" in result.error


def test_compare_circuits_statevector_failure(circuit_i: QuantumCircuit) -> None:
    with patch("qiskit.quantum_info.Statevector.from_instruction", side_effect=Exception("SV error")):
        result = compare_circuits_statevector(circuit_i, circuit_i)
        assert not result.equivalent
        assert result.error is not None
        assert "Statevector comparison failed" in result.error


# --- Tests for compare_circuits (auto) ---


def test_compare_circuits_auto_qcec(circuit_h: QuantumCircuit) -> None:
    # Should use QCEC if available
    with patch("benchmarks.ai_transpile.circuit_comparison.qcec") as mock_qcec:
        mock_qcec.verify.return_value.considered_equivalent = True
        result = compare_circuits(circuit_h, circuit_h, method="auto")
        assert result.method == "qcec"


def test_compare_circuits_auto_fallback_operator(circuit_h: QuantumCircuit) -> None:
    # Should fallback to operator if QCEC missing
    with patch("benchmarks.ai_transpile.circuit_comparison.qcec", None):
        result = compare_circuits(circuit_h, circuit_h, method="auto")
        assert result.method == "operator"


def test_compare_circuits_auto_fallback_statevector(circuit_h: QuantumCircuit) -> None:
    # Fallback to statevector if operator fails (e.g. too many qubits, but here we simulate by max_qubits)
    with patch("benchmarks.ai_transpile.circuit_comparison.qcec", None):
        result = compare_circuits(circuit_h, circuit_h, method="auto", max_qubits_for_operator=0)
        assert result.method == "statevector"


def test_compare_circuits_auto_failure(circuit_h: QuantumCircuit) -> None:
    # Fail if all fail or limits exceeded
    with patch("benchmarks.ai_transpile.circuit_comparison.qcec", None):
        result = compare_circuits(
            circuit_h, circuit_h, method="auto", max_qubits_for_operator=0, max_qubits_for_statevector=0
        )
        assert result.method == "failed"


def test_compare_circuits_explicit_method(circuit_h: QuantumCircuit) -> None:
    result = compare_circuits(circuit_h, circuit_h, method="statevector")
    assert result.method == "statevector"

    with patch("benchmarks.ai_transpile.circuit_comparison.compare_circuits_qcec") as mock_qcec:
        compare_circuits(circuit_h, circuit_h, method="qcec")
        mock_qcec.assert_called_once()

    with patch("benchmarks.ai_transpile.circuit_comparison.compare_circuits_operator") as mock_op:
        compare_circuits(circuit_h, circuit_h, method="operator")
        mock_op.assert_called_once()

    with pytest.raises(ValueError):
        compare_circuits(circuit_h, circuit_h, method="unknown")  # type: ignore


# --- Tests for compare_against_baseline ---


def test_compare_against_baseline(circuit_h_h: QuantumCircuit, circuit_i: QuantumCircuit) -> None:
    result = compare_against_baseline(circuit_i, circuit_h_h)
    assert result["equivalent"]
    assert result["method"] in ["qcec", "operator", "statevector"]
    assert result["baseline_qubits"] == 1
    assert result["optimized_qubits"] == 1
