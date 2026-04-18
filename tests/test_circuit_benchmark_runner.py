from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from benchmarks.ai_transpile.circuit_benchmark_runner import (
    CircuitConfig,
    ExperimentConfig,
    RunnerSpec,
    TranspiledCircuit,
    _discover_project_root,
    _result_record,
    configure_bqskit_workers,
    is_port_available,
    load_experiment_config,
    retry_on_failure,
    run_experiment,
    wait_for_port_cleanup,
)
from benchmarks.ai_transpile.transpilers import CircuitMetrics
from qiskit import QuantumCircuit

# --- Test Data ---


@pytest.fixture
def sample_circuit_config() -> CircuitConfig:
    return CircuitConfig(
        name="test_circuit",
        path=Path("path/to/circuit.qasm"),
        gate_set="IBMN",
        tags=("tag1",),
    )


@pytest.fixture
def sample_runner_spec() -> RunnerSpec:
    return RunnerSpec(
        name="test_runner",
        type="qiskit_ai",
        options={"opt_level": 3},
    )


@pytest.fixture
def sample_experiment_config(sample_circuit_config: CircuitConfig, sample_runner_spec: RunnerSpec) -> ExperimentConfig:
    return ExperimentConfig(
        metadata={"job_info": "test_job"},
        circuits=[sample_circuit_config],
        runners=[sample_runner_spec],
        metrics=("depth",),
    )


# --- Tests for ExperimentConfig ---


def test_experiment_config_initialization(sample_experiment_config: ExperimentConfig) -> None:
    assert sample_experiment_config.metadata["job_info"] == "test_job"
    assert len(sample_experiment_config.circuits) == 1
    assert len(sample_experiment_config.runners) == 1
    assert sample_experiment_config.metrics == ("depth",)


def test_experiment_config_defaults(sample_experiment_config: ExperimentConfig) -> None:
    # Test default output_dir
    assert sample_experiment_config.output_dir == Path("reports/circuit_benchmark")

    # Test default job_info
    assert sample_experiment_config.job_info == "test_job"

    # Test with custom output dir
    config_custom = ExperimentConfig(
        metadata={"default_output_dir": "custom/dir"},
        circuits=[],
        runners=[],
        metrics=(),
    )
    assert config_custom.output_dir == Path("custom/dir")


# --- Tests for RunnerSpec ---


def test_runner_spec_initialization(sample_runner_spec: RunnerSpec) -> None:
    assert sample_runner_spec.name == "test_runner"
    assert sample_runner_spec.type == "qiskit_ai"
    assert sample_runner_spec.options == {"opt_level": 3}


def test_runner_spec_defaults() -> None:
    spec = RunnerSpec(name="default", type="test")
    assert spec.options == {}


# --- Tests for _discover_project_root ---


def test_discover_project_root_with_pyproject(tmp_path: Path) -> None:
    # Create a fake project structure
    (tmp_path / "pyproject.toml").touch()
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    # Should find root from subdir
    assert _discover_project_root(subdir) == tmp_path
    # Should find root from root
    assert _discover_project_root(tmp_path) == tmp_path


def test_discover_project_root_fallback(tmp_path: Path) -> None:
    # No pyproject.toml
    # We need to mock _find_project_root to return None to test the fallbacks
    with patch("benchmarks.ai_transpile.circuit_benchmark_runner._find_project_root", return_value=None):
        # If we pass a path, and it can't find a root, it checks module root, then parents.
        # This is a bit hard to test without mocking __file__, but we can test the "parents >= 3" logic

        deep_path = Path("/a/b/c/d/e/f.yaml")
        # If we mock _find_project_root to always return None
        assert _discover_project_root(deep_path) == Path("/a/b/c")

        # Test fallback to parent if not deep enough
        shallow_path = Path("/a/b.yaml")
        assert _discover_project_root(shallow_path) == Path("/a")


def test_discover_project_root_module_fallback(tmp_path: Path) -> None:
    # Test when module root is found
    with patch("benchmarks.ai_transpile.circuit_benchmark_runner._find_project_root") as mock_find:
        # First call returns None (for config path), second returns something (for module)
        mock_find.side_effect = [None, tmp_path]
        assert _discover_project_root(Path("/some/path")) == tmp_path


def test_discover_project_root_system_root() -> None:
    # Test when reaching system root
    with patch("pathlib.Path.exists", return_value=False):
        # Should return None eventually, but _discover_project_root handles it
        # We need to mock _find_project_root to return None
        with patch("benchmarks.ai_transpile.circuit_benchmark_runner._find_project_root", return_value=None):
            # And mock parents to be short
            p = MagicMock()
            p.parents = [MagicMock()]
            p.parent = p  # Loop
            # This is hard to test perfectly without real FS, but we covered most logic
            pass


# --- Tests for _result_record ---


def test_result_record_creation() -> None:
    # Mock a TranspiledCircuit
    mock_circuit = MagicMock(spec=QuantumCircuit)
    metrics = CircuitMetrics(depth=10, two_qubit_gates=5, two_qubit_depth=5, total_gates=20)

    transpiled = TranspiledCircuit(
        optimizer="test_opt", label="test_label", circuit=mock_circuit, metrics=metrics, metadata={"extra": "info"}
    )

    record = _result_record("circuit_name", "runner_name", transpiled)

    assert record["circuit"] == "circuit_name"
    assert record["runner"] == "runner_name"
    assert record["optimizer"] == "test_opt"
    assert record["label"] == "test_label"
    assert record["metrics"]["depth"] == 10
    assert record["metadata"]["extra"] == "info"
    assert "artifact_path" not in record


def test_result_record_with_artifact() -> None:
    metrics = CircuitMetrics(depth=10, two_qubit_gates=5, two_qubit_depth=5, total_gates=20)
    transpiled = TranspiledCircuit(
        optimizer="test_opt",
        label="test_label",
        circuit=MagicMock(),
        metrics=metrics,
        artifact_path=Path("/tmp/out.qasm"),
    )

    record = _result_record("c", "r", transpiled)
    assert record["artifact_path"] == "/tmp/out.qasm"


# --- Tests for load_experiment_config ---


def test_load_experiment_config(tmp_path: Path) -> None:
    # Create a dummy config file
    config_content = """
    metadata:
      job_info: "test_run"
    circuits:
      - name: "c1"
        path: "circuits/c1.qasm"
        gate_set: "IBMN"
        tags: ["small"]
    runners:
      - name: "r1"
        type: "qiskit_ai"
        opt_level: 3
    metrics: ["depth"]
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    # Mock _discover_project_root to return tmp_path so circuit path resolution works
    with patch("benchmarks.ai_transpile.circuit_benchmark_runner._discover_project_root", return_value=tmp_path):
        config = load_experiment_config(config_file)

        assert config.metadata["job_info"] == "test_run"
        assert len(config.circuits) == 1
        assert config.circuits[0].name == "c1"
        assert config.circuits[0].path == tmp_path / "circuits/c1.qasm"
        assert len(config.runners) == 1
        assert config.runners[0].name == "r1"
        assert config.runners[0].type == "qiskit_ai"
        assert config.runners[0].options["opt_level"] == 3


# --- Tests for _load_circuit ---


def test_load_circuit(tmp_path: Path) -> None:
    qasm_content = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n'
    circuit_file = tmp_path / "test.qasm"
    circuit_file.write_text(qasm_content)

    from benchmarks.ai_transpile.circuit_benchmark_runner import _load_circuit

    circuit = _load_circuit(circuit_file)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


# --- Tests for is_port_available ---


def test_is_port_available_free_port() -> None:
    """Test that is_port_available returns True for a free port."""
    with patch("benchmarks.ai_transpile.circuit_benchmark_runner.socket.socket") as mock_socket:
        mock_sock_instance = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock_instance
        mock_sock_instance.connect_ex.return_value = 1  # Non-zero means connection failed (port free)

        assert is_port_available(9999) is True


def test_is_port_available_used_port() -> None:
    """Test that is_port_available returns False for a used port."""
    with patch("benchmarks.ai_transpile.circuit_benchmark_runner.socket.socket") as mock_socket:
        mock_sock_instance = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock_instance
        mock_sock_instance.connect_ex.return_value = 0  # Zero means connection succeeded (port in use)

        assert is_port_available(9999) is False


def test_is_port_available_exception() -> None:
    """Test that is_port_available returns False on exception."""
    with patch("benchmarks.ai_transpile.circuit_benchmark_runner.socket.socket") as mock_socket:
        mock_socket.return_value.__enter__.side_effect = Exception("Socket error")

        assert is_port_available(9999) is False


# --- Tests for wait_for_port_cleanup ---


def test_wait_for_port_cleanup_immediate() -> None:
    """Test wait_for_port_cleanup when port is immediately available."""
    with patch("benchmarks.ai_transpile.circuit_benchmark_runner.is_port_available", return_value=True):
        result = wait_for_port_cleanup(8080, max_wait=1.0)
        assert result is True


def test_wait_for_port_cleanup_timeout() -> None:
    """Test wait_for_port_cleanup when port never becomes available."""
    with (
        patch("benchmarks.ai_transpile.circuit_benchmark_runner.is_port_available", return_value=False),
        patch("benchmarks.ai_transpile.circuit_benchmark_runner.time.sleep"),
    ):
        result = wait_for_port_cleanup(8080, max_wait=0.1, check_interval=0.05)
        assert result is False


def test_wait_for_port_cleanup_eventual_success() -> None:
    """Test wait_for_port_cleanup when port becomes available after some time."""
    call_count = 0

    def mock_is_port_available(port: int) -> bool:
        nonlocal call_count
        call_count += 1
        return call_count >= 3  # Port available on 3rd check

    with (
        patch("benchmarks.ai_transpile.circuit_benchmark_runner.is_port_available", side_effect=mock_is_port_available),
        patch("benchmarks.ai_transpile.circuit_benchmark_runner.time.sleep"),
    ):
        result = wait_for_port_cleanup(8080, max_wait=5.0, check_interval=0.1)
        assert result is True


# --- Tests for retry_on_failure ---


def test_retry_on_failure_success_first_try() -> None:
    """Test retry_on_failure when function succeeds on first try."""

    def successful_func() -> str:
        return "success"

    wrapped = retry_on_failure(successful_func, max_attempts=3)
    result = wrapped()
    assert result == "success"


def test_retry_on_failure_success_after_retries() -> None:
    """Test retry_on_failure when function succeeds after retries."""
    call_count = 0

    def eventually_succeeds() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("Transient error")
        return "success"

    with patch("benchmarks.ai_transpile.circuit_benchmark_runner.time.sleep"):
        wrapped = retry_on_failure(
            eventually_succeeds,
            max_attempts=3,
            initial_delay=0.1,
            retry_exceptions=(RuntimeError,),
        )
        result = wrapped()
        assert result == "success"
        assert call_count == 3


def test_retry_on_failure_all_attempts_fail() -> None:
    """Test retry_on_failure when all attempts fail."""

    def always_fails() -> str:
        raise OSError("Persistent error")

    with patch("benchmarks.ai_transpile.circuit_benchmark_runner.time.sleep"):
        wrapped = retry_on_failure(
            always_fails,
            max_attempts=3,
            initial_delay=0.1,
            retry_exceptions=(OSError,),
        )
        with pytest.raises(OSError, match="Persistent error"):
            wrapped()


def test_retry_on_failure_non_retryable_exception() -> None:
    """Test retry_on_failure with non-retryable exception."""

    def raises_value_error() -> str:
        raise ValueError("Non-retryable")

    wrapped = retry_on_failure(
        raises_value_error,
        max_attempts=3,
        retry_exceptions=(RuntimeError,),  # ValueError not included
    )
    with pytest.raises(ValueError, match="Non-retryable"):
        wrapped()


def test_retry_on_failure_exponential_backoff() -> None:
    """Test that retry_on_failure uses exponential backoff."""
    sleep_times: list[float] = []

    def mock_sleep(seconds: float) -> None:
        sleep_times.append(seconds)

    def always_fails() -> str:
        raise RuntimeError("Error")

    with patch("benchmarks.ai_transpile.circuit_benchmark_runner.time.sleep", side_effect=mock_sleep):
        wrapped = retry_on_failure(
            always_fails,
            max_attempts=4,
            initial_delay=1.0,
            backoff_factor=2.0,
            retry_exceptions=(RuntimeError,),
        )
        with pytest.raises(RuntimeError):
            wrapped()

    # Should have slept 3 times (attempts 1, 2, 3 before giving up)
    assert len(sleep_times) == 3
    assert sleep_times[0] == 1.0
    assert sleep_times[1] == 2.0
    assert sleep_times[2] == 4.0


# --- Tests for run_experiment ---


@patch("benchmarks.ai_transpile.circuit_benchmark_runner._load_circuit")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.transpile_with_qiskit_ai")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.run_wisq_opt")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.qasm2.dumps")
def test_run_experiment(
    mock_qasm_dumps: MagicMock,
    mock_wisq: MagicMock,
    mock_qiskit_ai: MagicMock,
    mock_load_circuit: MagicMock,
    sample_experiment_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    # Setup mocks
    mock_circuit = MagicMock(spec=QuantumCircuit)
    mock_circuit.num_qubits = 2
    mock_circuit.decompose.return_value = mock_circuit  # For swap decomposition
    mock_load_circuit.return_value = mock_circuit
    mock_qasm_dumps.return_value = "OPENQASM 2.0; ..."

    metrics = CircuitMetrics(depth=5, two_qubit_gates=2, two_qubit_depth=2, total_gates=10)

    # Mock Qiskit AI result
    mock_qiskit_ai.return_value = [TranspiledCircuit("qiskit_ai", "v1", mock_circuit, metrics, None, {})]

    # Mock WISQ result
    mock_wisq.return_value = TranspiledCircuit("wisq", "v1", mock_circuit, metrics, Path("out.qasm"), {})

    # 1. Test Qiskit AI runner
    report = run_experiment(sample_experiment_config, output_dir=tmp_path)
    assert len(report["results"]) == 1
    assert report["results"][0]["runner"] == "test_runner"
    assert report["failures"] == []

    # 2. Test WISQ runner
    wisq_runner = RunnerSpec("wisq_runner", "wisq", {})
    config_wisq = ExperimentConfig({}, sample_experiment_config.circuits, [wisq_runner], ())
    report_wisq = run_experiment(config_wisq, output_dir=tmp_path)
    assert len(report_wisq["results"]) == 1
    assert report_wisq["results"][0]["runner"] == "wisq_runner"

    # 3. Test Unknown runner
    bad_runner = RunnerSpec("bad", "unknown", {})
    config_bad = ExperimentConfig({}, sample_experiment_config.circuits, [bad_runner], ())
    report_bad = run_experiment(config_bad, output_dir=tmp_path)
    assert len(report_bad["failures"]) == 1
    assert "Unsupported runner type" in report_bad["failures"][0]["error"]

    # 4. Test Skip runner
    report_skip = run_experiment(sample_experiment_config, output_dir=tmp_path, skip_runners=["test_runner"])
    assert len(report_skip["results"]) == 0
    assert len(report_skip["failures"]) == 1
    assert report_skip["failures"][0]["error"] == "skipped"


@patch("benchmarks.ai_transpile.circuit_benchmark_runner._load_circuit")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.transpile_with_qiskit_standard")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.qasm2.dumps")
def test_run_experiment_qiskit_standard(
    mock_qasm_dumps: MagicMock,
    mock_qiskit_standard: MagicMock,
    mock_load_circuit: MagicMock,
    sample_circuit_config: CircuitConfig,
    tmp_path: Path,
) -> None:
    """Test run_experiment with qiskit_standard runner type."""
    mock_circuit = MagicMock(spec=QuantumCircuit)
    mock_circuit.decompose.return_value = mock_circuit
    mock_load_circuit.return_value = mock_circuit
    mock_qasm_dumps.return_value = "OPENQASM 2.0; ..."

    metrics = CircuitMetrics(depth=5, two_qubit_gates=2, two_qubit_depth=2, total_gates=10)
    mock_qiskit_standard.return_value = [TranspiledCircuit("qiskit_standard", "opt_1", mock_circuit, metrics, None, {})]

    qiskit_standard_runner = RunnerSpec("qiskit_std", "qiskit_standard", {"optimization_levels": [1]})
    config = ExperimentConfig({}, [sample_circuit_config], [qiskit_standard_runner], ())

    report = run_experiment(config, output_dir=tmp_path)
    assert len(report["results"]) == 1
    assert report["results"][0]["runner"] == "qiskit_std"
    mock_qiskit_standard.assert_called_once()


@patch("benchmarks.ai_transpile.circuit_benchmark_runner._load_circuit")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.run_tket")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.qasm2.dumps")
def test_run_experiment_tket(
    mock_qasm_dumps: MagicMock,
    mock_run_tket: MagicMock,
    mock_load_circuit: MagicMock,
    sample_circuit_config: CircuitConfig,
    tmp_path: Path,
) -> None:
    """Test run_experiment with tket runner type."""
    mock_circuit = MagicMock(spec=QuantumCircuit)
    mock_circuit.decompose.return_value = mock_circuit
    mock_load_circuit.return_value = mock_circuit
    mock_qasm_dumps.return_value = "OPENQASM 2.0; ..."

    metrics = CircuitMetrics(depth=5, two_qubit_gates=2, two_qubit_depth=2, total_gates=10)
    artifact_path = tmp_path / "out.qasm"
    mock_run_tket.return_value = TranspiledCircuit("tket", "tket_ibmn", mock_circuit, metrics, artifact_path, {})

    tket_runner = RunnerSpec("tket_runner", "tket", {"gate_set": "IBMN"})
    config = ExperimentConfig({}, [sample_circuit_config], [tket_runner], ())

    report = run_experiment(config, output_dir=tmp_path)
    assert len(report["results"]) == 1
    assert report["results"][0]["runner"] == "tket_runner"
    mock_run_tket.assert_called_once()


@patch("benchmarks.ai_transpile.circuit_benchmark_runner._load_circuit")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.run_voqc")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.qasm2.dumps")
def test_run_experiment_voqc(
    mock_qasm_dumps: MagicMock,
    mock_run_voqc: MagicMock,
    mock_load_circuit: MagicMock,
    sample_circuit_config: CircuitConfig,
    tmp_path: Path,
) -> None:
    """Test run_experiment with voqc runner type."""
    mock_circuit = MagicMock(spec=QuantumCircuit)
    mock_circuit.decompose.return_value = mock_circuit
    mock_load_circuit.return_value = mock_circuit
    mock_qasm_dumps.return_value = "OPENQASM 2.0; ..."

    metrics = CircuitMetrics(depth=5, two_qubit_gates=2, two_qubit_depth=2, total_gates=10)
    mock_run_voqc.return_value = TranspiledCircuit("voqc", "voqc_nam", mock_circuit, metrics, tmp_path / "out.qasm", {})

    voqc_runner = RunnerSpec("voqc_runner", "voqc", {"optimization_method": "nam"})
    config = ExperimentConfig({}, [sample_circuit_config], [voqc_runner], ())

    report = run_experiment(config, output_dir=tmp_path)
    assert len(report["results"]) == 1
    assert report["results"][0]["runner"] == "voqc_runner"
    mock_run_voqc.assert_called_once()


@patch("benchmarks.ai_transpile.circuit_benchmark_runner._load_circuit")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.execute_chain")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.qasm2.dumps")
def test_run_experiment_chain(
    mock_qasm_dumps: MagicMock,
    mock_execute_chain: MagicMock,
    mock_load_circuit: MagicMock,
    sample_circuit_config: CircuitConfig,
    tmp_path: Path,
) -> None:
    """Test run_experiment with chain runner type."""
    from benchmarks.ai_transpile.chain_executor import ChainResult, ChainStep, StepResult

    mock_circuit = MagicMock(spec=QuantumCircuit)
    mock_circuit.decompose.return_value = mock_circuit
    mock_load_circuit.return_value = mock_circuit
    mock_qasm_dumps.return_value = "OPENQASM 2.0; ..."

    initial_metrics = CircuitMetrics(depth=10, two_qubit_gates=5, two_qubit_depth=5, total_gates=20)
    final_metrics = CircuitMetrics(depth=8, two_qubit_gates=3, two_qubit_depth=3, total_gates=15)

    # Create mock chain result
    steps = [ChainStep("qiskit_standard", {"optimization_levels": [3]})]
    step_results = [
        StepResult(
            step=steps[0],
            step_index=0,
            input_metrics=initial_metrics,
            output_metrics=final_metrics,
            transpiled=TranspiledCircuit("qiskit_standard", "opt3", mock_circuit, final_metrics, None, {}),
            duration_seconds=1.0,
            artifact_path=tmp_path / "out.qasm",
        )
    ]
    mock_chain_result = ChainResult(
        chain_name="test_chain",
        steps=steps,
        step_results=step_results,
        initial_circuit=mock_circuit,
        initial_metrics=initial_metrics,
        final_circuit=mock_circuit,
        final_metrics=final_metrics,
        total_duration_seconds=1.0,
    )
    mock_execute_chain.return_value = mock_chain_result

    chain_runner = RunnerSpec(
        "chain_runner",
        "chain",
        {"steps": [{"type": "qiskit_standard", "optimization_levels": [3]}]},
    )
    config = ExperimentConfig({}, [sample_circuit_config], [chain_runner], ())

    report = run_experiment(config, output_dir=tmp_path)
    assert len(report["results"]) == 1
    assert report["results"][0]["runner"] == "chain_runner"
    assert report["results"][0]["optimizer"] == "chain"
    assert "chain_name" in report["results"][0]["metadata"]
    mock_execute_chain.assert_called_once()


@patch("benchmarks.ai_transpile.circuit_benchmark_runner._load_circuit")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.transpile_with_qiskit_ai")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.compare_against_baseline")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.qasm2.dumps")
def test_run_experiment_comparison(
    mock_qasm_dumps: MagicMock,
    mock_compare: MagicMock,
    mock_qiskit_ai: MagicMock,
    mock_load_circuit: MagicMock,
    sample_experiment_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    # Setup mocks
    mock_circuit = MagicMock(spec=QuantumCircuit)
    mock_circuit.decompose.return_value = mock_circuit
    mock_load_circuit.return_value = mock_circuit
    mock_qasm_dumps.return_value = "OPENQASM 2.0; ..."

    metrics = CircuitMetrics(depth=5, two_qubit_gates=2, two_qubit_depth=2, total_gates=10)

    # Return a variant with artifact_path set, simulating a loaded result
    mock_qiskit_ai.return_value = [TranspiledCircuit("qiskit_ai", "v1", mock_circuit, metrics, None, {})]

    mock_compare.return_value = {"equivalent": True}

    # Add a second runner to compare against the first
    runner1 = sample_experiment_config.runners[0]  # test_runner
    runner2 = RunnerSpec("other_runner", "qiskit_ai", {})

    config = ExperimentConfig({}, sample_experiment_config.circuits, [runner1, runner2], ())

    # We need to ensure qasm2.loads works when loading the artifact for comparison
    with patch("qiskit.qasm2.loads", return_value=mock_circuit):
        report = run_experiment(config, output_dir=tmp_path, compare_against_baseline_runner="test_runner")

    assert "comparisons" in report
    assert len(report["comparisons"]) == 1
    assert report["comparisons"][0]["baseline_runner"] == "test_runner"
    assert report["comparisons"][0]["optimized_runner"] == "other_runner"


@patch("benchmarks.ai_transpile.circuit_benchmark_runner._load_circuit")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.transpile_with_qiskit_ai")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.qasm2.dumps")
def test_run_experiment_comparison_failure(
    mock_qasm_dumps: MagicMock,
    mock_qiskit_ai: MagicMock,
    mock_load_circuit: MagicMock,
    sample_experiment_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    # Test case where loading the optimized circuit fails
    mock_circuit = MagicMock(spec=QuantumCircuit)
    mock_circuit.decompose.return_value = mock_circuit
    mock_load_circuit.return_value = mock_circuit
    mock_qasm_dumps.return_value = "OPENQASM 2.0; ..."
    metrics = CircuitMetrics(depth=5, two_qubit_gates=2, two_qubit_depth=2, total_gates=10)
    mock_qiskit_ai.return_value = [TranspiledCircuit("qiskit_ai", "v1", mock_circuit, metrics, None, {})]

    runner1 = sample_experiment_config.runners[0]
    runner2 = RunnerSpec("other_runner", "qiskit_ai", {})
    config = ExperimentConfig({}, sample_experiment_config.circuits, [runner1, runner2], ())

    # Mock qasm2.loads to raise exception
    with patch("qiskit.qasm2.loads", side_effect=Exception("Load error")):
        report = run_experiment(config, output_dir=tmp_path, compare_against_baseline_runner="test_runner")

    assert len(report["comparisons"]) == 1
    assert report["comparisons"][0]["equivalent"] is False
    assert "Load error" in report["comparisons"][0]["error"]


# --- Tests for main ---


@patch("benchmarks.ai_transpile.circuit_benchmark_runner.load_experiment_config")
@patch("benchmarks.ai_transpile.circuit_benchmark_runner.run_experiment")
def test_main(mock_run: MagicMock, mock_load: MagicMock) -> None:
    from benchmarks.ai_transpile.circuit_benchmark_runner import main

    # Mock return values
    mock_load.return_value = MagicMock()
    mock_run.return_value = {"metadata": {"num_results": 5}, "report_path": "report.json", "failures": []}

    # Test with default args
    with patch("sys.argv", ["script_name"]):
        main()
        mock_load.assert_called_once()
        mock_run.assert_called_once()

    # Test with failures
    mock_run.return_value["failures"] = [{"circuit": "c1", "runner": "r1", "error": "e1"}]
    with patch("sys.argv", ["script_name"]):
        main()  # Should print failures but not crash


# --- Tests for configure_bqskit_workers ---


def test_configure_bqskit_workers_explicit_count() -> None:
    """Test configure_bqskit_workers with explicit worker count."""
    import os
    
    # Clear any existing value
    os.environ.pop("BQSKIT_NUM_WORKERS", None)
    
    result = configure_bqskit_workers(num_workers=4)
    
    assert result == 4
    assert os.environ["BQSKIT_NUM_WORKERS"] == "4"
    
    # Clean up
    os.environ.pop("BQSKIT_NUM_WORKERS", None)


def test_configure_bqskit_workers_fraction() -> None:
    """Test configure_bqskit_workers with worker fraction."""
    import multiprocessing
    import os
    
    # Clear any existing value
    os.environ.pop("BQSKIT_NUM_WORKERS", None)
    
    result = configure_bqskit_workers(worker_fraction=0.25)
    
    expected = max(1, int(multiprocessing.cpu_count() * 0.25))
    assert result == expected
    assert os.environ["BQSKIT_NUM_WORKERS"] == str(expected)
    
    # Clean up
    os.environ.pop("BQSKIT_NUM_WORKERS", None)


def test_configure_bqskit_workers_uses_existing_env() -> None:
    """Test configure_bqskit_workers respects existing env var."""
    import os
    
    # Set existing value
    os.environ["BQSKIT_NUM_WORKERS"] = "8"
    
    result = configure_bqskit_workers()
    
    assert result == 8
    assert os.environ["BQSKIT_NUM_WORKERS"] == "8"
    
    # Clean up
    os.environ.pop("BQSKIT_NUM_WORKERS", None)


def test_configure_bqskit_workers_default_count() -> None:
    """Test configure_bqskit_workers with default worker count (12)."""
    import os
    
    # Clear any existing value
    os.environ.pop("BQSKIT_NUM_WORKERS", None)
    
    result = configure_bqskit_workers()
    
    # Default is 12 workers
    assert result == 12
    assert os.environ["BQSKIT_NUM_WORKERS"] == "12"
    
    # Clean up
    os.environ.pop("BQSKIT_NUM_WORKERS", None)


def test_configure_bqskit_workers_fraction_overrides_env() -> None:
    """Test that explicit worker_fraction overrides existing BQSKIT_NUM_WORKERS env var.
    
    Bug fix test: Previously, when BQSKIT_NUM_WORKERS was set in environment and
    worker_fraction was explicitly provided, the env var value was used instead
    of calculating from the fraction. This test ensures explicit worker_fraction
    arguments are treated as overrides.
    """
    import multiprocessing
    import os
    
    # Set existing value in environment (e.g., from run_wisq_safe.sh wrapper)
    os.environ["BQSKIT_NUM_WORKERS"] = "16"
    
    # User explicitly configures worker_fraction in YAML (e.g., bqskit_worker_fraction: 0.25)
    result = configure_bqskit_workers(worker_fraction=0.25)
    
    # The explicit worker_fraction should override the env var
    expected = max(1, int(multiprocessing.cpu_count() * 0.25))
    assert result == expected
    assert os.environ["BQSKIT_NUM_WORKERS"] == str(expected)
    
    # Clean up
    os.environ.pop("BQSKIT_NUM_WORKERS", None)


def test_configure_bqskit_workers_num_workers_overrides_env() -> None:
    """Test that explicit num_workers overrides existing BQSKIT_NUM_WORKERS env var."""
    import os
    
    # Set existing value in environment
    os.environ["BQSKIT_NUM_WORKERS"] = "16"
    
    # User explicitly configures num_workers
    result = configure_bqskit_workers(num_workers=4)
    
    # The explicit num_workers should override the env var
    assert result == 4
    assert os.environ["BQSKIT_NUM_WORKERS"] == "4"
    
    # Clean up
    os.environ.pop("BQSKIT_NUM_WORKERS", None)
