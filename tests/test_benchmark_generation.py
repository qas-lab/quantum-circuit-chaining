"""Tests for benchmark circuit generation."""

from __future__ import annotations

import json
from pathlib import Path

from qiskit import QuantumCircuit


def test_generate_qft_circuit() -> None:
    """Test QFT circuit generation with correct properties."""
    from scripts.generate_benchmark_circuits import generate_qft_circuit

    num_qubits = 8
    circuit = generate_qft_circuit(num_qubits)

    assert circuit.num_qubits == num_qubits
    assert circuit.num_clbits == 0  # No classical bits for pure QFT
    assert circuit.size() > 0  # Has gates


def test_generate_qft_various_sizes() -> None:
    """Test QFT generation for different qubit counts."""
    from scripts.generate_benchmark_circuits import generate_qft_circuit

    for num_qubits in [8, 10, 12, 16]:
        circuit = generate_qft_circuit(num_qubits)
        assert circuit.num_qubits == num_qubits
        # QFT should have O(n^2) gates
        assert circuit.size() > num_qubits


def test_generate_real_amplitudes() -> None:
    """Test RealAmplitudes ansatz generation."""
    from scripts.generate_benchmark_circuits import generate_real_amplitudes

    num_qubits = 12
    reps = 3
    circuit = generate_real_amplitudes(num_qubits, reps=reps)

    assert circuit.num_qubits == num_qubits
    assert circuit.size() > 0


def test_generate_efficient_su2() -> None:
    """Test EfficientSU2 ansatz generation."""
    from scripts.generate_benchmark_circuits import generate_efficient_su2

    num_qubits = 8
    reps = 2
    circuit = generate_efficient_su2(num_qubits, reps=reps)

    assert circuit.num_qubits == num_qubits
    assert circuit.size() > 0


def test_bind_parameters_to_zero() -> None:
    """Test parameter binding utility."""
    from scripts.generate_benchmark_circuits import bind_parameters_to_zero, generate_real_amplitudes

    num_qubits = 8
    circuit = generate_real_amplitudes(num_qubits, reps=2)

    # Should have parameters before binding
    assert len(circuit.parameters) > 0

    bound = bind_parameters_to_zero(circuit)

    # Should have no parameters after binding
    assert len(bound.parameters) == 0
    assert bound.num_qubits == circuit.num_qubits


def test_circuit_to_qasm_valid(tmp_path: Path) -> None:
    """Test that generated circuits can be exported to valid QASM."""
    from scripts.generate_benchmark_circuits import generate_qft_circuit, save_benchmark_circuit

    circuit = generate_qft_circuit(8)
    output_file = tmp_path / "test.qasm"
    save_benchmark_circuit(circuit, output_file)

    # Should be able to reload the QASM using the standard Qiskit method
    reloaded = QuantumCircuit.from_qasm_file(str(output_file))
    assert reloaded.num_qubits == circuit.num_qubits


def test_generate_benchmark_metadata() -> None:
    """Test metadata generation for circuits."""
    from scripts.generate_benchmark_circuits import generate_circuit_metadata

    circuit = QuantumCircuit(8)
    circuit.h(0)
    circuit.cx(0, 1)

    metadata = generate_circuit_metadata(
        circuit=circuit,
        name="test_circuit",
        description="Test description",
        tags=["test", "demo"],
    )

    assert metadata["name"] == "test_circuit"
    assert metadata["description"] == "Test description"
    assert metadata["tags"] == ["test", "demo"]
    assert metadata["num_qubits"] == 8
    assert "metrics" in metadata
    metrics = metadata["metrics"]
    assert isinstance(metrics, dict)
    assert "depth" in metrics
    assert "two_qubit_gates" in metrics


def test_save_benchmark_circuit(tmp_path: Path) -> None:
    """Test saving circuit to QASM file."""
    from scripts.generate_benchmark_circuits import generate_qft_circuit, save_benchmark_circuit

    circuit = generate_qft_circuit(8)
    output_file = tmp_path / "test_qft.qasm"

    save_benchmark_circuit(circuit, output_file)

    assert output_file.exists()
    # Should be valid QASM
    loaded = QuantumCircuit.from_qasm_file(str(output_file))
    assert loaded.num_qubits == circuit.num_qubits


def test_update_metadata_file(tmp_path: Path) -> None:
    """Test updating metadata.json with new circuits."""
    from scripts.generate_benchmark_circuits import update_metadata_file

    metadata_file = tmp_path / "metadata.json"

    # Create initial metadata
    initial_data = {"circuits": [{"name": "existing", "file": "existing.qasm"}]}
    metadata_file.write_text(json.dumps(initial_data, indent=2))

    # Add new circuit
    new_circuit_metadata = {
        "name": "new_circuit",
        "description": "New test circuit",
        "file": "qasm/new_circuit.qasm",
        "num_qubits": 8,
        "tags": ["test"],
        "metrics": {"depth": 10, "two_qubit_gates": 5, "two_qubit_depth": 5, "total_gates": 15},
    }

    update_metadata_file(metadata_file, new_circuit_metadata)

    # Verify the file was updated
    updated_data = json.loads(metadata_file.read_text())
    assert len(updated_data["circuits"]) == 2
    assert updated_data["circuits"][1]["name"] == "new_circuit"


def test_update_metadata_file_no_duplicates(tmp_path: Path) -> None:
    """Test that updating metadata doesn't create duplicates."""
    from scripts.generate_benchmark_circuits import update_metadata_file

    metadata_file = tmp_path / "metadata.json"

    # Create initial metadata
    initial_data = {"circuits": [{"name": "test_circuit", "file": "test.qasm"}]}
    metadata_file.write_text(json.dumps(initial_data, indent=2))

    # Try to add the same circuit again
    duplicate_metadata = {
        "name": "test_circuit",
        "description": "Updated description",
        "file": "qasm/test_circuit.qasm",
        "num_qubits": 8,
        "tags": ["test"],
        "metrics": {"depth": 10, "two_qubit_gates": 5, "two_qubit_depth": 5, "total_gates": 15},
    }

    update_metadata_file(metadata_file, duplicate_metadata)

    # Should still only have one circuit (replaced, not duplicated)
    updated_data = json.loads(metadata_file.read_text())
    assert len(updated_data["circuits"]) == 1
    assert updated_data["circuits"][0]["description"] == "Updated description"

