from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from qiskit import QuantumCircuit

if TYPE_CHECKING:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --- Common Circuit Fixtures ---


@pytest.fixture
def sample_circuit() -> QuantumCircuit:
    """A simple 2-qubit circuit with H and CX gates."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def circuit_h() -> QuantumCircuit:
    """A 1-qubit circuit with just a Hadamard gate."""
    qc = QuantumCircuit(1)
    qc.h(0)
    return qc


@pytest.fixture
def circuit_h_h() -> QuantumCircuit:
    """A 1-qubit circuit with two Hadamard gates (equivalent to identity)."""
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.h(0)
    return qc


@pytest.fixture
def circuit_i() -> QuantumCircuit:
    """A 1-qubit circuit with just an identity gate."""
    qc = QuantumCircuit(1)
    qc.id(0)
    return qc


@pytest.fixture
def circuit_x() -> QuantumCircuit:
    """A 1-qubit circuit with just an X (NOT) gate."""
    qc = QuantumCircuit(1)
    qc.x(0)
    return qc


# --- Test Data Fixtures ---


@pytest.fixture
def mock_metadata_json() -> str:
    """Sample metadata.json content for testing circuit loading."""
    return """
    {
        "circuits": [
            {
                "name": "test_c",
                "description": "desc",
                "tags": ["tag"],
                "num_qubits": 2,
                "metrics": {
                    "depth": 2,
                    "two_qubit_gates": 1,
                    "two_qubit_depth": 1,
                    "total_gates": 2
                },
                "file": "test.qasm"
            }
        ]
    }
    """


@pytest.fixture
def sample_qasm_content() -> str:
    """Sample QASM 2.0 content for a 2-qubit circuit."""
    return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n'


@pytest.fixture
def empty_qasm_content() -> str:
    """Sample QASM 2.0 content for an empty 2-qubit circuit."""
    return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'


# --- File System Fixtures ---


@pytest.fixture
def tmp_circuit_file(tmp_path: Path, sample_qasm_content: str) -> Path:
    """Create a temporary QASM file with sample circuit content."""
    circuit_file = tmp_path / "test_circuit.qasm"
    circuit_file.write_text(sample_qasm_content)
    return circuit_file


@pytest.fixture
def tmp_empty_circuit_file(tmp_path: Path, empty_qasm_content: str) -> Path:
    """Create a temporary QASM file with empty circuit content."""
    circuit_file = tmp_path / "empty_circuit.qasm"
    circuit_file.write_text(empty_qasm_content)
    return circuit_file


@pytest.fixture
def tmp_metadata_dir(tmp_path: Path, mock_metadata_json: str, empty_qasm_content: str) -> Path:
    """Create a temporary directory with metadata.json and a test circuit file."""
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(mock_metadata_json)
    qasm_file = tmp_path / "test.qasm"
    qasm_file.write_text(empty_qasm_content)
    return tmp_path
