from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from benchmarks.ai_transpile.transpilers import (
    BenchmarkCircuit,
    QiskitAIRunnerConfig,
    QiskitStandardConfig,
    TKETConfig,
    VOQCConfig,
    WisqConfig,
    analyze_circuit,
    get_benchmark_circuit,
    load_benchmark_circuits,
    run_tket,
    run_voqc,
    run_wisq_opt,
    transpile_with_qiskit_ai,
    transpile_with_qiskit_standard,
)
from qiskit import QuantumCircuit

# Note: sample_circuit, mock_metadata_json fixtures are now in conftest.py


# --- Tests for Circuit Loading ---


def test_load_benchmark_circuits(mock_metadata_json: str, tmp_path: Path) -> None:
    # Setup mock file system
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(mock_metadata_json)

    circuits = load_benchmark_circuits(root=tmp_path)
    assert len(circuits) == 1
    assert "test_c" in circuits
    c = circuits["test_c"]
    assert c.name == "test_c"
    assert c.num_qubits == 2
    assert c.metrics.depth == 2
    assert c.qasm_path == tmp_path / "test.qasm"


def test_get_benchmark_circuit(mock_metadata_json: str, tmp_path: Path) -> None:
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(mock_metadata_json)

    c = get_benchmark_circuit("test_c", root=tmp_path)
    assert c.name == "test_c"

    with pytest.raises(KeyError):
        get_benchmark_circuit("unknown", root=tmp_path)


def test_benchmark_circuit_load(tmp_path: Path) -> None:
    qasm_file = tmp_path / "test.qasm"
    qasm_file.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n')

    c = BenchmarkCircuit(name="test", description="", tags=(), num_qubits=2, metrics=MagicMock(), qasm_path=qasm_file)
    qc = c.load()
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 2


# --- Tests for Analysis ---


def test_analyze_circuit(sample_circuit: QuantumCircuit) -> None:
    metrics = analyze_circuit(sample_circuit)
    assert metrics.depth == 2
    assert metrics.two_qubit_gates == 1
    assert metrics.total_gates == 2


# --- Tests for Qiskit AI Runner ---


def test_transpile_with_qiskit_ai(sample_circuit: QuantumCircuit) -> None:
    # Mock PassManager to avoid actual transpilation
    with patch("benchmarks.ai_transpile.transpilers.PassManager") as MockPM:
        mock_pm_instance = MockPM.return_value
        mock_pm_instance.run.side_effect = lambda c: c  # Return circuit as is

        config = QiskitAIRunnerConfig(optimization_levels=(1,), iterations_per_level=1)

        results = transpile_with_qiskit_ai(sample_circuit, config)

        # Should return 1 baseline + 1 optimized variant
        assert len(results) == 2
        assert results[0].label == "sabre_routed"
        assert results[1].label == "ai_level_1_iter_1"


# --- Tests for Qiskit Standard Runner ---


def test_transpile_with_qiskit_standard(sample_circuit: QuantumCircuit) -> None:
    """Test standard Qiskit transpiler at different optimization levels."""
    config = QiskitStandardConfig(optimization_levels=(1, 2))
    results = transpile_with_qiskit_standard(sample_circuit, config)

    # Should return one result per optimization level
    assert len(results) == 2
    assert results[0].optimizer == "qiskit_standard"
    assert results[0].label == "qiskit_opt_level_1"
    assert results[1].label == "qiskit_opt_level_2"

    # Verify metadata contains duration
    for result in results:
        assert "duration_seconds" in result.metadata
        assert result.metadata["variant"] == "standard_transpiler"


def test_transpile_with_qiskit_standard_default_config(sample_circuit: QuantumCircuit) -> None:
    """Test standard Qiskit transpiler with default configuration."""
    results = transpile_with_qiskit_standard(sample_circuit)

    # Default config has optimization_levels (1, 2, 3)
    assert len(results) == 3
    assert all(r.optimizer == "qiskit_standard" for r in results)


def test_transpile_with_qiskit_standard_metrics(sample_circuit: QuantumCircuit) -> None:
    """Test that metrics are properly computed for standard transpiler."""
    config = QiskitStandardConfig(optimization_levels=(1,))
    results = transpile_with_qiskit_standard(sample_circuit, config)

    assert len(results) == 1
    result = results[0]

    # Verify metrics are populated
    assert result.metrics.depth >= 0
    assert result.metrics.total_gates >= 0
    assert result.metrics.two_qubit_gates >= 0


# --- Tests for WISQ Runner ---


def test_run_wisq_opt_success(tmp_path: Path) -> None:
    # Mock wisq.optimize
    with patch("benchmarks.ai_transpile.transpilers.wisq_optimize") as mock_opt:
        config = WisqConfig(output_dir=tmp_path, advanced_args={"k": "v"})
        circuit_path = tmp_path / "c.qasm"

        # Create dummy output
        output_path = config.output_file_for(circuit_path)
        output_path.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n')

        result = run_wisq_opt(circuit_path, config)

        assert result.optimizer == "wisq"
        mock_opt.assert_called_once()
        # Check advanced args passed
        args = mock_opt.call_args[1]["advanced_args"]
        assert args["k"] == "v"


def test_run_wisq_opt_missing() -> None:
    # Simulate wisq not installed
    with patch("benchmarks.ai_transpile.transpilers.wisq_optimize", None):
        with pytest.raises(ImportError):
            run_wisq_opt(Path("c.qasm"))


# --- Tests for TKET Runner ---


def test_run_tket_success(tmp_path: Path) -> None:
    """Test TKET runner with mocked subprocess."""
    import json

    # Create input circuit file
    circuit_path = tmp_path / "input.qasm"
    circuit_path.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n')

    # Mock the TKET environment check and script execution
    mock_result_data = {
        "qasm": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n',
        "duration": 0.5,
    }
    mock_result = MagicMock()
    mock_result.stdout = json.dumps(mock_result_data)

    with (
        patch("benchmarks.ai_transpile.transpilers.PYTKET_AVAILABLE", True),
        patch("benchmarks.tket_runner.run_tket_script", return_value=mock_result),
    ):
        config = TKETConfig(output_dir=tmp_path)
        result = run_tket(circuit_path, config)

        assert result.optimizer == "tket"
        assert result.label == "tket_ibmn"
        assert result.metadata["gate_set"] == "IBMN"
        assert result.metadata["duration_seconds"] == 0.5
        assert result.artifact_path is not None
        assert result.artifact_path.exists()


def test_run_tket_not_available() -> None:
    """Test TKET runner when environment is not available."""
    with patch("benchmarks.ai_transpile.transpilers.PYTKET_AVAILABLE", False):
        with pytest.raises(ImportError, match="TKET environment not found"):
            run_tket(Path("circuit.qasm"))


def test_run_tket_custom_gate_set(tmp_path: Path) -> None:
    """Test TKET runner with custom gate set."""
    import json

    circuit_path = tmp_path / "input.qasm"
    circuit_path.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\n')

    mock_result_data = {
        "qasm": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\n',
        "duration": 0.3,
    }
    mock_result = MagicMock()
    mock_result.stdout = json.dumps(mock_result_data)

    with (
        patch("benchmarks.ai_transpile.transpilers.PYTKET_AVAILABLE", True),
        patch("benchmarks.tket_runner.run_tket_script", return_value=mock_result),
    ):
        config = TKETConfig(output_dir=tmp_path, gate_set="nam")
        result = run_tket(circuit_path, config)

        assert result.label == "tket_nam"
        assert result.metadata["gate_set"] == "nam"


def test_build_tket_optimization_script_valid_syntax() -> None:
    """Test that _build_tket_optimization_script generates valid Python syntax."""
    import ast

    from benchmarks.ai_transpile.transpilers import _build_tket_optimization_script

    # Test with typical QASM content
    qasm_content = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n'
    gate_names = ["cx", "rz", "sx", "x"]

    script = _build_tket_optimization_script(qasm_content, gate_names)

    # Verify the script is valid Python syntax by parsing it
    try:
        ast.parse(script)
    except SyntaxError as e:
        pytest.fail(f"Generated script has invalid syntax: {e}\n\nScript:\n{script}")


def test_build_tket_optimization_script_with_special_chars() -> None:
    """Test script generation with QASM containing special characters."""
    import ast

    from benchmarks.ai_transpile.transpilers import _build_tket_optimization_script

    # QASM with various special characters that might cause escaping issues
    qasm_content = '''OPENQASM 2.0;
include "qelib1.inc";
// Comment with 'single' and "double" quotes
qreg q[2];
h q[0];
'''
    gate_names = ["cx", "h"]

    script = _build_tket_optimization_script(qasm_content, gate_names)

    # Verify the script is valid Python syntax
    try:
        ast.parse(script)
    except SyntaxError as e:
        pytest.fail(f"Generated script has invalid syntax: {e}\n\nScript:\n{script}")


def test_build_tket_optimization_script_escaping_preserves_content() -> None:
    """Test that the escaping correctly preserves QASM content including edge cases.

    This verifies that:
    - Triple double quotes are correctly escaped and restored
    - Backslashes are correctly escaped and restored
    - Combinations of both work correctly

    Note: In Python triple-quoted strings, backslash-quote is a valid escape
    sequence that produces a single quote character.
    """
    import ast

    from benchmarks.ai_transpile.transpilers import _build_tket_optimization_script

    test_cases = [
        'OPENQASM 2.0; // normal content',
        'OPENQASM 2.0; // with """triple quotes"""',
        r'OPENQASM 2.0; // with \backslash',
        r'OPENQASM 2.0; // with \"""both\"""',
        'OPENQASM 2.0;\n// multiline\nqreg q[2];',
    ]

    for original in test_cases:
        script = _build_tket_optimization_script(original, ["cx"])

        # Verify script is valid Python
        try:
            tree = ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"Invalid syntax for input {repr(original)}: {e}")

        # Find the qasm_str assignment and extract the string value from the AST
        qasm_str_value = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'qasm_str':
                        if isinstance(node.value, ast.Constant):
                            qasm_str_value = node.value.value
                        break

        assert qasm_str_value is not None, f"Could not find qasm_str in AST for input {repr(original)}"
        assert qasm_str_value == original, f"Escaping failed for: {repr(original)}\nGot: {repr(qasm_str_value)}"


# --- Tests for VOQC Runner ---


def test_run_voqc_success(tmp_path: Path) -> None:
    """Test VOQC runner with mocked pass manager."""
    # Create input circuit file
    circuit_path = tmp_path / "input.qasm"
    circuit_path.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n')

    # Create a mock circuit to return
    mock_circuit = QuantumCircuit(2)
    mock_circuit.h(0)
    mock_circuit.cx(0, 1)

    # Mock the pass manager
    mock_pm = MagicMock()
    mock_pm.run.return_value = mock_circuit

    with (
        patch("benchmarks.ai_transpile.transpilers.PYVOQC_AVAILABLE", True),
        patch("benchmarks.ai_transpile.transpilers.voqc_pass_manager", return_value=mock_pm),
    ):
        config = VOQCConfig(output_dir=tmp_path)
        result = run_voqc(circuit_path, config)

        assert result.optimizer == "voqc"
        assert result.label == "voqc_nam"
        assert result.metadata["optimization_method"] == "nam"
        assert "duration_seconds" in result.metadata
        assert result.artifact_path is not None
        mock_pm.run.assert_called_once()


def test_run_voqc_not_available() -> None:
    """Test VOQC runner when pyvoqc is not installed."""
    with patch("benchmarks.ai_transpile.transpilers.PYVOQC_AVAILABLE", False):
        with pytest.raises(ImportError, match="pyvoqc is not available"):
            run_voqc(Path("circuit.qasm"))


def test_run_voqc_with_error_message(tmp_path: Path) -> None:
    """Test VOQC runner includes import error message when available."""
    with (
        patch("benchmarks.ai_transpile.transpilers.PYVOQC_AVAILABLE", False),
        patch("benchmarks.ai_transpile.transpilers.PYVOQC_ERROR", "OCaml library not found"),
    ):
        with pytest.raises(ImportError, match="OCaml library not found"):
            run_voqc(tmp_path / "circuit.qasm")


def test_run_voqc_custom_optimization_method(tmp_path: Path) -> None:
    """Test VOQC runner with custom optimization method."""
    circuit_path = tmp_path / "input.qasm"
    circuit_path.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\n')

    mock_circuit = QuantumCircuit(2)
    mock_circuit.h(0)

    mock_pm = MagicMock()
    mock_pm.run.return_value = mock_circuit

    with (
        patch("benchmarks.ai_transpile.transpilers.PYVOQC_AVAILABLE", True),
        patch("benchmarks.ai_transpile.transpilers.voqc_pass_manager", return_value=mock_pm) as mock_vpm,
    ):
        config = VOQCConfig(output_dir=tmp_path, optimization_method="ibm")
        result = run_voqc(circuit_path, config)

        assert result.label == "voqc_ibm"
        assert result.metadata["optimization_method"] == "ibm"
        # Verify the correct post_opts were passed
        mock_vpm.assert_called_once()
        call_kwargs = mock_vpm.call_args[1]
        assert call_kwargs["post_opts"] == ["optimize_nam", "optimize_ibm"]
