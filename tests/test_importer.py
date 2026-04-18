"""Tests for the circuit importer module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from benchmarks.ai_transpile.rl_trajectory.database import (
    TrajectoryDatabase,
)
from benchmarks.ai_transpile.rl_trajectory.importer import (
    BenchpressImporter,
    LocalCircuitImporter,
    _discover_qasm_files,
    _get_circuit_num_qubits,
    _infer_category_from_path,
    import_from_metadata_json,
)

# --- Fixtures ---


@pytest.fixture
def sample_qasm_content() -> str:
    """Valid QASM 2.0 content for a 2-qubit circuit."""
    return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n'


@pytest.fixture
def db(tmp_path: Path) -> TrajectoryDatabase:
    """Create a temporary database."""
    return TrajectoryDatabase(tmp_path / "test.db")


# --- Tests for _discover_qasm_files ---


def test_discover_qasm_files_recursive(tmp_path: Path, sample_qasm_content: str) -> None:
    """Test recursive discovery finds nested QASM files."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.qasm").write_text(sample_qasm_content)
    (sub / "b.qasm").write_text(sample_qasm_content)

    files = _discover_qasm_files(tmp_path, recursive=True)
    names = {f.name for f in files}
    assert "a.qasm" in names
    assert "b.qasm" in names


def test_discover_qasm_files_non_recursive(tmp_path: Path, sample_qasm_content: str) -> None:
    """Test non-recursive discovery skips nested files."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.qasm").write_text(sample_qasm_content)
    (sub / "b.qasm").write_text(sample_qasm_content)

    files = _discover_qasm_files(tmp_path, recursive=False)
    names = {f.name for f in files}
    assert "a.qasm" in names
    assert "b.qasm" not in names


def test_discover_qasm_files_empty_dir(tmp_path: Path) -> None:
    """Test discovery in empty directory returns empty list."""
    files = _discover_qasm_files(tmp_path)
    assert files == []


def test_discover_qasm_files_nonexistent_dir() -> None:
    """Test discovery in nonexistent directory returns empty list."""
    files = _discover_qasm_files(Path("/nonexistent/dir"))
    assert files == []


# --- Tests for _get_circuit_num_qubits ---


def test_get_circuit_num_qubits_valid(tmp_path: Path, sample_qasm_content: str) -> None:
    """Test valid QASM returns correct qubit count."""
    qasm_file = tmp_path / "test.qasm"
    qasm_file.write_text(sample_qasm_content)
    result = _get_circuit_num_qubits(qasm_file)
    assert result == 2


def test_get_circuit_num_qubits_invalid(tmp_path: Path) -> None:
    """Test invalid QASM content returns None."""
    qasm_file = tmp_path / "bad.qasm"
    qasm_file.write_text("not valid qasm")
    result = _get_circuit_num_qubits(qasm_file)
    assert result is None


def test_get_circuit_num_qubits_missing_file() -> None:
    """Test missing file returns None."""
    result = _get_circuit_num_qubits(Path("/nonexistent/file.qasm"))
    assert result is None


# --- Tests for _infer_category_from_path ---


def test_infer_category_qft_in_path(tmp_path: Path) -> None:
    """Test 'qft' in path returns 'qft' category."""
    qasm_path = tmp_path / "qft_circuits" / "test.qasm"
    result = _infer_category_from_path(qasm_path, tmp_path)
    assert result == "qft"


def test_infer_category_qaoa_in_path(tmp_path: Path) -> None:
    """Test 'qaoa' in path returns 'qaoa' category."""
    qasm_path = tmp_path / "qaoa_bench" / "test.qasm"
    result = _infer_category_from_path(qasm_path, tmp_path)
    assert result == "qaoa"


def test_infer_category_fallback_to_dir_name(tmp_path: Path) -> None:
    """Test fallback to first directory name when no known category matches."""
    qasm_path = tmp_path / "MyCustomDir" / "test.qasm"
    result = _infer_category_from_path(qasm_path, tmp_path)
    assert result == "mycustomdir"


def test_infer_category_file_at_root(tmp_path: Path) -> None:
    """Test file at base returns 'unknown'."""
    qasm_path = tmp_path / "test.qasm"
    result = _infer_category_from_path(qasm_path, tmp_path)
    assert result == "unknown"


def test_infer_category_unrelated_base() -> None:
    """Test unrelated base path returns 'unknown'."""
    result = _infer_category_from_path(Path("/a/b/c.qasm"), Path("/x/y"))
    assert result == "unknown"


# --- Tests for LocalCircuitImporter ---


def test_local_importer_discover_circuits(tmp_path: Path, sample_qasm_content: str) -> None:
    """Test LocalCircuitImporter.discover_circuits finds QASM files."""
    (tmp_path / "circ1.qasm").write_text(sample_qasm_content)
    (tmp_path / "circ2.qasm").write_text(sample_qasm_content)

    importer = LocalCircuitImporter(tmp_path)
    circuits = importer.discover_circuits()

    assert len(circuits) == 2
    names = {c.name for c in circuits}
    assert "local_circ1" in names
    assert "local_circ2" in names


def test_local_importer_max_qubits_filter(tmp_path: Path, sample_qasm_content: str) -> None:
    """Test max_qubits filters out circuits with too many qubits."""
    (tmp_path / "test.qasm").write_text(sample_qasm_content)

    importer = LocalCircuitImporter(tmp_path)
    # Circuit has 2 qubits, so max_qubits=1 should filter it out
    circuits = importer.discover_circuits(max_qubits=1)
    assert len(circuits) == 0

    # max_qubits=2 should include it
    circuits = importer.discover_circuits(max_qubits=2)
    assert len(circuits) == 1


def test_local_importer_import_to_database(
    tmp_path: Path, sample_qasm_content: str, db: TrajectoryDatabase
) -> None:
    """Test LocalCircuitImporter.import_to_database inserts records."""
    (tmp_path / "test.qasm").write_text(sample_qasm_content)

    importer = LocalCircuitImporter(tmp_path)
    count = importer.import_to_database(db)

    assert count == 1
    circuits = db.list_circuits()
    assert len(circuits) == 1
    assert circuits[0].name == "local_test"


def test_local_importer_skip_existing(
    tmp_path: Path, sample_qasm_content: str, db: TrajectoryDatabase
) -> None:
    """Test skip_existing prevents duplicate imports."""
    (tmp_path / "test.qasm").write_text(sample_qasm_content)

    importer = LocalCircuitImporter(tmp_path)
    first_count = importer.import_to_database(db, skip_existing=True)
    second_count = importer.import_to_database(db, skip_existing=True)

    assert first_count == 1
    assert second_count == 0
    assert len(db.list_circuits()) == 1


# --- Tests for import_from_metadata_json ---


def test_import_from_metadata_json(tmp_metadata_dir: Path, db: TrajectoryDatabase) -> None:
    """Test importing circuits from metadata.json."""
    metadata_path = tmp_metadata_dir / "metadata.json"
    count = import_from_metadata_json(db, metadata_path)

    assert count == 1
    circuits = db.list_circuits()
    assert len(circuits) == 1
    assert circuits[0].name == "test_c"


def test_import_from_metadata_json_skip_existing(
    tmp_metadata_dir: Path, db: TrajectoryDatabase
) -> None:
    """Test skip_existing prevents duplicate imports from metadata."""
    metadata_path = tmp_metadata_dir / "metadata.json"
    first_count = import_from_metadata_json(db, metadata_path, skip_existing=True)
    second_count = import_from_metadata_json(db, metadata_path, skip_existing=True)

    assert first_count == 1
    assert second_count == 0


# --- Tests for BenchpressImporter ---


@patch("benchmarks.ai_transpile.rl_trajectory.importer.subprocess.run")
def test_benchpress_clone_or_update_clones(mock_run: MagicMock, tmp_path: Path) -> None:
    """Test clone_or_update_repo calls git clone when repo doesn't exist."""
    importer = BenchpressImporter(cache_dir=tmp_path / "cache")
    mock_run.return_value = MagicMock(returncode=0)

    importer.clone_or_update_repo()

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "git"
    assert args[1] == "clone"


@patch("benchmarks.ai_transpile.rl_trajectory.importer.subprocess.run")
def test_benchpress_clone_or_update_pulls(mock_run: MagicMock, tmp_path: Path) -> None:
    """Test clone_or_update_repo calls git pull when repo exists and force_update=True."""
    importer = BenchpressImporter(cache_dir=tmp_path / "cache")
    # Create the repo directory to simulate it already existing
    importer.repo_path.mkdir(parents=True)
    mock_run.return_value = MagicMock(returncode=0)

    importer.clone_or_update_repo(force_update=True)

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "git"
    assert args[1] == "pull"


@patch("benchmarks.ai_transpile.rl_trajectory.importer.subprocess.run")
def test_benchpress_clone_or_update_skips_when_exists(
    mock_run: MagicMock, tmp_path: Path
) -> None:
    """Test clone_or_update_repo skips when repo exists and not forced."""
    importer = BenchpressImporter(cache_dir=tmp_path / "cache")
    importer.repo_path.mkdir(parents=True)

    importer.clone_or_update_repo(force_update=False)

    mock_run.assert_not_called()


def test_benchpress_discover_circuits_with_mock_repo(
    tmp_path: Path, sample_qasm_content: str
) -> None:
    """Test discover_circuits finds files in mock repo structure."""
    cache_dir = tmp_path / "cache"
    importer = BenchpressImporter(cache_dir=cache_dir)

    # Create mock repo structure
    repo = importer.repo_path
    circuits_dir = repo / "benchpress" / "circuits" / "qft"
    circuits_dir.mkdir(parents=True)
    (circuits_dir / "qft_4.qasm").write_text(sample_qasm_content)

    circuits = importer.discover_circuits(max_qubits=20)
    assert len(circuits) >= 1
    assert any("qft" in c.name for c in circuits)
