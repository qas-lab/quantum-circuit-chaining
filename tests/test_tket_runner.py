"""Tests for the TKET environment runner module."""

import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from benchmarks.tket_runner import (
    REPO_ROOT,
    TKET_PYTHON,
    TKETEnvironmentError,
    get_tket_python_path,
    print_environment_info,
    run_tket_function,
    run_tket_script,
    verify_tket_environment,
)

# --- Tests for verify_tket_environment ---


def test_verify_tket_environment_success() -> None:
    """Test that verify_tket_environment passes when the environment exists."""
    with patch("benchmarks.tket_runner.TKET_PYTHON") as mock_path:
        mock_path.exists.return_value = True
        # Should not raise
        verify_tket_environment()


def test_verify_tket_environment_missing() -> None:
    """Test that verify_tket_environment raises when the environment is missing."""
    with patch("benchmarks.tket_runner.TKET_PYTHON") as mock_path:
        mock_path.exists.return_value = False
        with pytest.raises(TKETEnvironmentError, match="TKET environment not found"):
            verify_tket_environment()


# --- Tests for run_tket_script ---


def test_run_tket_script_success() -> None:
    """Test running a simple script in the TKET environment."""
    mock_result = MagicMock(spec=subprocess.CompletedProcess)
    mock_result.stdout = "hello\n"
    mock_result.stderr = ""
    mock_result.returncode = 0

    with (
        patch("benchmarks.tket_runner.TKET_PYTHON") as mock_tket_python,
        patch("benchmarks.tket_runner.subprocess.run", return_value=mock_result) as mock_run,
    ):
        mock_tket_python.exists.return_value = True
        mock_tket_python.__str__ = lambda x: "/fake/path/python"

        result = run_tket_script("print('hello')")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["check"] is True
        assert result.stdout == "hello\n"


def test_run_tket_script_capture_output() -> None:
    """Test that capture_output parameter is passed correctly."""
    mock_result = MagicMock(spec=subprocess.CompletedProcess)

    with (
        patch("benchmarks.tket_runner.TKET_PYTHON") as mock_tket_python,
        patch("benchmarks.tket_runner.subprocess.run", return_value=mock_result) as mock_run,
    ):
        mock_tket_python.exists.return_value = True
        mock_tket_python.__str__ = lambda x: "/fake/path/python"

        run_tket_script("print(1)", capture_output=True)
        assert mock_run.call_args[1]["capture_output"] is True

        run_tket_script("print(2)", capture_output=False)
        assert mock_run.call_args[1]["capture_output"] is False


def test_run_tket_script_environment_missing() -> None:
    """Test that run_tket_script raises when environment is missing."""
    with patch("benchmarks.tket_runner.TKET_PYTHON") as mock_tket_python:
        mock_tket_python.exists.return_value = False
        with pytest.raises(TKETEnvironmentError):
            run_tket_script("print('hello')")


def test_run_tket_script_failure() -> None:
    """Test that subprocess errors are propagated."""
    with (
        patch("benchmarks.tket_runner.TKET_PYTHON") as mock_tket_python,
        patch(
            "benchmarks.tket_runner.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "python", stderr="syntax error"),
        ),
    ):
        mock_tket_python.exists.return_value = True
        mock_tket_python.__str__ = lambda x: "/fake/path/python"

        with pytest.raises(subprocess.CalledProcessError):
            run_tket_script("invalid python $$")


# --- Tests for run_tket_function ---


def test_run_tket_function_success() -> None:
    """Test calling a function in the TKET environment."""
    mock_result = MagicMock(spec=subprocess.CompletedProcess)
    mock_result.stdout = '{"result": 42}\n'

    with (
        patch("benchmarks.tket_runner.TKET_PYTHON") as mock_tket_python,
        patch("benchmarks.tket_runner.subprocess.run", return_value=mock_result),
    ):
        mock_tket_python.exists.return_value = True
        mock_tket_python.__str__ = lambda x: "/fake/path/python"

        result = run_tket_function("math", "sqrt", 16)

        assert result == {"result": 42}


def test_run_tket_function_with_kwargs(tmp_path: Path) -> None:
    """Test calling a function with keyword arguments writes args to temp file."""
    mock_result = MagicMock(spec=subprocess.CompletedProcess)
    mock_result.stdout = '"test_value"\n'

    written_content: list[str] = []

    original_ntf = tempfile.NamedTemporaryFile

    def mock_ntf(**kwargs: Any) -> Any:
        # Redirect temp file to tmp_path so we can inspect it
        kwargs["dir"] = str(tmp_path)
        f = original_ntf(**kwargs)
        original_write = f.write

        def capture_write(data: str) -> int:
            written_content.append(data)
            return original_write(data)

        f.write = capture_write  # type: ignore[method-assign]
        return f

    with (
        patch("benchmarks.tket_runner.TKET_PYTHON") as mock_tket_python,
        patch("benchmarks.tket_runner.run_tket_script", return_value=mock_result),
        patch("benchmarks.tket_runner.tempfile.NamedTemporaryFile", side_effect=mock_ntf),
    ):
        mock_tket_python.exists.return_value = True

        run_tket_function("module.path", "func_name", "arg1", kwarg1="val1")

        # Verify kwargs were written to the temp file
        assert len(written_content) >= 1
        import json

        args_data = json.loads(written_content[0])
        assert args_data["kwargs"] == {"kwarg1": "val1"}
        assert args_data["args"] == ["arg1"]


def test_run_tket_function_environment_missing() -> None:
    """Test that run_tket_function raises when environment is missing."""
    with patch("benchmarks.tket_runner.TKET_PYTHON") as mock_tket_python:
        mock_tket_python.exists.return_value = False
        with pytest.raises(TKETEnvironmentError):
            run_tket_function("module", "func")


# --- Tests for get_tket_python_path ---


def test_get_tket_python_path_success() -> None:
    """Test getting the Python path when environment exists."""
    with patch("benchmarks.tket_runner.TKET_PYTHON") as mock_tket_python:
        mock_tket_python.exists.return_value = True
        path = get_tket_python_path()
        # Should return the mocked TKET_PYTHON
        assert path is mock_tket_python


def test_get_tket_python_path_missing() -> None:
    """Test that get_tket_python_path raises when environment is missing."""
    with patch("benchmarks.tket_runner.TKET_PYTHON") as mock_tket_python:
        mock_tket_python.exists.return_value = False
        with pytest.raises(TKETEnvironmentError):
            get_tket_python_path()


# --- Tests for print_environment_info ---


def test_print_environment_info_both_available(capsys: pytest.CaptureFixture[str]) -> None:
    """Test printing environment info when both environments are available."""
    with (
        patch("benchmarks.tket_runner.TKET_PYTHON") as mock_tket_python,
        patch("benchmarks.tket_runner.subprocess.run") as mock_run,
    ):
        mock_tket_python.exists.return_value = True
        mock_tket_python.__str__ = lambda x: "/fake/path/python"
        mock_run.return_value = MagicMock(returncode=0)

        print_environment_info()

        captured = capsys.readouterr()
        assert "Main Environment" in captured.out
        assert "TKET Environment" in captured.out


def test_print_environment_info_tket_missing(capsys: pytest.CaptureFixture[str]) -> None:
    """Test printing environment info when TKET environment is missing."""
    with (
        patch("benchmarks.tket_runner.TKET_PYTHON") as mock_tket_python,
        patch("benchmarks.tket_runner.subprocess.run") as mock_run,
    ):
        mock_tket_python.exists.return_value = False
        mock_run.return_value = MagicMock(returncode=0)

        print_environment_info()

        captured = capsys.readouterr()
        assert "Main Environment" in captured.out
        assert "TKET Environment" in captured.out
        assert "Error:" in captured.out


# --- Tests for module constants ---


def test_repo_root_exists() -> None:
    """Test that REPO_ROOT points to a valid directory."""
    assert REPO_ROOT.exists()
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_tket_python_path_structure() -> None:
    """Test that TKET_PYTHON has expected path structure."""
    assert TKET_PYTHON.name == "python"
    assert ".venv-tket" in str(TKET_PYTHON)
    assert "bin" in str(TKET_PYTHON)
