"""Tests for rl_training dataset loading, normalization, and splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from benchmarks.ai_transpile.rl_training.config import TrainingConfig
from benchmarks.ai_transpile.rl_training.dataset import (
    OfflineRLDataset,
    make_dataloader,
    split_dataset,
)
from benchmarks.ai_transpile.rl_training.normalization import (
    NormalizationStats,
    compute_normalization_stats,
)
from benchmarks.ai_transpile.rl_trajectory.state import get_category_encoding

# --- Normalization Tests ---


class TestNormalizationStats:
    def test_compute_stats(self) -> None:
        """Compute mean/std from data."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        stats = compute_normalization_stats(data)
        np.testing.assert_allclose(stats.means, [3.0, 4.0], atol=1e-5)
        assert stats.count == 3

    def test_normalize(self) -> None:
        """Z-score normalization."""
        stats = NormalizationStats(
            means=np.array([0.0, 10.0], dtype=np.float32),
            stds=np.array([1.0, 5.0], dtype=np.float32),
            count=100,
        )
        x = np.array([1.0, 15.0], dtype=np.float32)
        result = stats.normalize(x)
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-5)

    def test_normalize_zero_std(self) -> None:
        """Zero std should not cause division by zero."""
        stats = NormalizationStats(
            means=np.array([5.0], dtype=np.float32),
            stds=np.array([0.0], dtype=np.float32),
            count=10,
        )
        result = stats.normalize(np.array([5.0], dtype=np.float32))
        assert np.isfinite(result[0])

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Save and load normalization stats."""
        stats = NormalizationStats(
            means=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            stds=np.array([0.5, 1.0, 1.5], dtype=np.float32),
            count=50,
        )
        path = tmp_path / "norm.json"
        stats.save(path)
        loaded = NormalizationStats.load(path)
        np.testing.assert_allclose(loaded.means, stats.means)
        np.testing.assert_allclose(loaded.stds, stats.stds)
        assert loaded.count == 50

    def test_compute_stats_invalid_input(self) -> None:
        """Empty or 1D array should raise ValueError."""
        with pytest.raises(ValueError):
            compute_normalization_stats(np.array([], dtype=np.float32).reshape(0, 3))
        with pytest.raises(ValueError):
            compute_normalization_stats(np.array([1.0, 2.0], dtype=np.float32))


# --- Dataset Tests ---


def _create_test_db(tmp_path: Path, n_circuits: int = 3, n_optimizers: int = 2) -> Path:
    """Create a minimal test database with trajectory steps."""
    import sqlite3

    from benchmarks.ai_transpile.rl_trajectory.database import SCHEMA_SQL

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)

    # Insert circuits
    for i in range(n_circuits):
        conn.execute(
            """INSERT INTO circuits (name, category, source, num_qubits,
            initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
            initial_total_gates, gate_density, two_qubit_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"circuit_{i}", "qft", "test", 4, 10, 5, 3, 20, 5.0, 0.25),
        )

    # Insert optimizers
    for i in range(n_optimizers):
        conn.execute(
            """INSERT INTO optimizers (name, runner_type, options_json)
            VALUES (?, ?, ?)""",
            (f"opt_{i}", "test", "{}"),
        )

    # Insert trajectories and steps
    category_encoding = [0.0] * 13
    category_encoding[0] = 1.0  # qft
    cat_json = json.dumps(category_encoding)

    step_id = 0
    for circuit_id in range(1, n_circuits + 1):
        for opt_id in range(1, n_optimizers + 1):
            traj_id = conn.execute(
                """INSERT INTO trajectories (circuit_id, chain_name, num_steps,
                initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
                initial_total_gates, final_depth, final_two_qubit_gates,
                final_two_qubit_depth, final_total_gates, total_duration_seconds,
                total_reward, improvement_percentage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (circuit_id, f"single_{opt_id}", 1, 10, 5, 3, 20, 8, 3, 2, 15, 1.5, 0.4, 40.0),
            ).lastrowid

            conn.execute(
                """INSERT INTO trajectory_steps (
                trajectory_id, step_index, optimizer_id,
                state_depth, state_two_qubit_gates, state_two_qubit_depth,
                state_total_gates, state_num_qubits, state_gate_density,
                state_two_qubit_ratio, state_steps_taken,
                state_time_budget_remaining, state_category_json,
                next_state_depth, next_state_two_qubit_gates,
                next_state_two_qubit_depth, next_state_total_gates,
                next_state_gate_density, next_state_two_qubit_ratio,
                next_state_steps_taken, next_state_time_budget_remaining,
                reward_improvement_only, reward_efficiency,
                reward_multi_objective, reward_sparse_final,
                done, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    traj_id, 0, opt_id,
                    10, 5, 3, 20, 4, 5.0, 0.25, 0, 300.0, cat_json,
                    8, 3, 2, 15, 3.75, 0.2, 1, 298.5,
                    0.4, 0.24, 0.35, 0.4,
                    1, 1.5,
                ),
            )
            step_id += 1

    conn.commit()
    conn.close()
    return db_path


def _create_legacy_category_db(tmp_path: Path, category: str) -> Path:
    """Create a DB row with legacy-shape category JSON and a chosen text label."""
    import sqlite3

    from benchmarks.ai_transpile.rl_trajectory.database import SCHEMA_SQL

    db_path = tmp_path / f"legacy_{category}.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)

    circuit_id = conn.execute(
        """INSERT INTO circuits (
        name, category, source, num_qubits,
        initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
        initial_total_gates, gate_density, two_qubit_ratio
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (f"legacy_{category}", category, "test", 4, 10, 5, 3, 20, 5.0, 0.25),
    ).lastrowid
    optimizer_id = conn.execute(
        "INSERT INTO optimizers (name, runner_type, options_json) VALUES (?, ?, ?)",
        ("opt_0", "test", "{}"),
    ).lastrowid
    trajectory_id = conn.execute(
        """INSERT INTO trajectories (
        circuit_id, chain_name, num_steps,
        initial_depth, initial_two_qubit_gates, initial_two_qubit_depth,
        initial_total_gates, final_depth, final_two_qubit_gates,
        final_two_qubit_depth, final_total_gates, total_duration_seconds,
        total_reward, improvement_percentage
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (circuit_id, "single_legacy", 1, 10, 5, 3, 20, 8, 3, 2, 15, 1.5, 0.4, 40.0),
    ).lastrowid

    old_category_index = 11 if category == "local" else 12
    legacy_encoding = [0.0] * 13
    legacy_encoding[old_category_index] = 1.0
    cat_json = json.dumps(legacy_encoding)

    conn.execute(
        """INSERT INTO trajectory_steps (
        trajectory_id, step_index, optimizer_id,
        state_depth, state_two_qubit_gates, state_two_qubit_depth,
        state_total_gates, state_num_qubits, state_gate_density,
        state_two_qubit_ratio, state_steps_taken,
        state_time_budget_remaining, state_category_json,
        next_state_depth, next_state_two_qubit_gates,
        next_state_two_qubit_depth, next_state_total_gates,
        next_state_gate_density, next_state_two_qubit_ratio,
        next_state_steps_taken, next_state_time_budget_remaining,
        reward_improvement_only, reward_efficiency,
        reward_multi_objective, reward_sparse_final,
        done, duration_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trajectory_id, 0, optimizer_id,
            10, 5, 3, 20, 4, 5.0, 0.25, 0, 300.0, cat_json,
            8, 3, 2, 15, 3.75, 0.2, 1, 298.5,
            0.4, 0.24, 0.35, 0.4,
            1, 1.5,
        ),
    )
    conn.commit()
    conn.close()
    return db_path


class TestOfflineRLDataset:
    def test_load_from_database(self, tmp_path: Path) -> None:
        """Load dataset from test database."""
        db_path = _create_test_db(tmp_path)
        config = TrainingConfig(database_path=str(db_path))
        dataset = OfflineRLDataset.from_database(db_path, config)

        assert len(dataset) == 6  # 3 circuits * 2 optimizers
        assert dataset.observations.shape == (6, 26)
        assert dataset.actions.shape == (6,)
        assert dataset.rewards.shape == (6,)

    def test_action_remapping(self, tmp_path: Path) -> None:
        """DB optimizer IDs (1-indexed) should be remapped to 0-indexed."""
        db_path = _create_test_db(tmp_path)
        config = TrainingConfig(database_path=str(db_path))
        dataset = OfflineRLDataset.from_database(db_path, config)

        # Actions should be 0-indexed
        assert dataset.actions.min().item() == 0
        assert dataset.actions.max().item() == 1  # 2 optimizers: 0 and 1

    def test_action_names(self, tmp_path: Path) -> None:
        """Action names should match optimizer names from DB."""
        db_path = _create_test_db(tmp_path)
        config = TrainingConfig(database_path=str(db_path))
        dataset = OfflineRLDataset.from_database(db_path, config)

        assert dataset.action_names == ["opt_0", "opt_1"]

    def test_normalization_applied(self, tmp_path: Path) -> None:
        """Observations should be normalized (roughly zero mean)."""
        db_path = _create_test_db(tmp_path)
        config = TrainingConfig(database_path=str(db_path))
        dataset = OfflineRLDataset.from_database(db_path, config)

        # With all identical states, normalized values should be ~0
        mean = dataset.observations.mean(dim=0)
        assert mean.abs().max().item() < 1.0  # Loosely normalized

    def test_getitem(self, tmp_path: Path) -> None:
        """__getitem__ should return dict with correct keys."""
        db_path = _create_test_db(tmp_path)
        config = TrainingConfig(database_path=str(db_path))
        dataset = OfflineRLDataset.from_database(db_path, config)

        item = dataset[0]
        assert set(item.keys()) == {"observations", "actions", "rewards", "next_observations", "terminals"}
        assert item["observations"].shape == (26,)
        assert item["actions"].dim() == 0  # scalar

    def test_custom_norm_stats(self, tmp_path: Path) -> None:
        """Pre-computed normalization stats should be used."""
        db_path = _create_test_db(tmp_path)
        config = TrainingConfig(database_path=str(db_path))

        custom_stats = NormalizationStats(
            means=np.zeros(26, dtype=np.float32),
            stds=np.ones(26, dtype=np.float32),
            count=100,
        )
        dataset = OfflineRLDataset.from_database(db_path, config, norm_stats=custom_stats)
        assert dataset.norm_stats.count == 100

    def test_reward_type_selection(self, tmp_path: Path) -> None:
        """Different reward types should give different values."""
        db_path = _create_test_db(tmp_path)

        config_imp = TrainingConfig(reward_type="reward_improvement_only")
        ds_imp = OfflineRLDataset.from_database(db_path, config_imp)

        config_eff = TrainingConfig(reward_type="reward_efficiency")
        ds_eff = OfflineRLDataset.from_database(db_path, config_eff)

        # Different reward types should give different values
        assert not (ds_imp.rewards == ds_eff.rewards).all()

    @pytest.mark.parametrize("category", ["local", "unknown"])
    def test_legacy_category_encodings_preserve_semantics(
        self,
        tmp_path: Path,
        category: str,
    ) -> None:
        """Old 13-dim encodings should map to the correct post-guoq category."""
        db_path = _create_legacy_category_db(tmp_path, category)
        config = TrainingConfig(database_path=str(db_path))
        dataset = OfflineRLDataset.from_database(
            db_path,
            config,
            norm_stats=NormalizationStats(
                means=np.zeros(26, dtype=np.float32),
                stds=np.ones(26, dtype=np.float32),
                count=1,
            ),
        )

        expected = np.array(get_category_encoding(category), dtype=np.float32)
        np.testing.assert_array_equal(dataset.observations[0].numpy()[12:], expected)

    def test_unknown_fallback_for_unrecognized_circuit_category(self, tmp_path: Path) -> None:
        """Unrecognized textual categories should fall back to the unknown bucket."""
        db_path = _create_legacy_category_db(tmp_path, "no_such_category")
        config = TrainingConfig(database_path=str(db_path))
        dataset = OfflineRLDataset.from_database(
            db_path,
            config,
            norm_stats=NormalizationStats(
                means=np.zeros(26, dtype=np.float32),
                stds=np.ones(26, dtype=np.float32),
                count=1,
            ),
        )

        expected = np.array(get_category_encoding("unknown"), dtype=np.float32)
        np.testing.assert_array_equal(dataset.observations[0].numpy()[12:], expected)


class TestDatasetSplitting:
    def test_split_by_circuit(self, tmp_path: Path) -> None:
        """Circuit-based splitting should partition data."""
        db_path = _create_test_db(tmp_path, n_circuits=6, n_optimizers=2)
        config = TrainingConfig(
            val_fraction=0.15,
            test_fraction=0.15,
            split_by_circuit=True,
            seed=42,
        )
        dataset = OfflineRLDataset.from_database(db_path, config)
        train_ds, val_ds, test_ds = split_dataset(dataset, db_path, config)

        # All samples should be accounted for
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == len(dataset)

        # Each split should have at least 1 sample
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

    def test_random_split(self, tmp_path: Path) -> None:
        """Random split should also partition data."""
        db_path = _create_test_db(tmp_path, n_circuits=4, n_optimizers=2)
        config = TrainingConfig(
            val_fraction=0.25,
            test_fraction=0.25,
            split_by_circuit=False,
            seed=42,
        )
        dataset = OfflineRLDataset.from_database(db_path, config)
        train_ds, val_ds, test_ds = split_dataset(dataset, db_path, config)

        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == len(dataset)


class TestDataLoader:
    def test_make_dataloader(self, tmp_path: Path) -> None:
        """DataLoader should yield batches with correct structure."""
        db_path = _create_test_db(tmp_path)
        config = TrainingConfig(database_path=str(db_path))
        dataset = OfflineRLDataset.from_database(db_path, config)
        loader = make_dataloader(dataset, batch_size=4)

        batch = next(iter(loader))
        assert "observations" in batch
        assert batch["observations"].shape[1] == 26
        assert batch["actions"].dtype in (torch.int64, torch.long)


# --- Config Tests ---


class TestTrainingConfig:
    def test_defaults(self) -> None:
        """Default config should have sensible values."""
        config = TrainingConfig()
        assert config.algorithm == "bc"
        assert config.state_dim == 26
        assert config.action_dim == 5

    def test_yaml_roundtrip(self, tmp_path: Path) -> None:
        """Save and load config from YAML."""
        config = TrainingConfig(algorithm="cql", learning_rate=0.001, seed=123)
        path = tmp_path / "config.yaml"
        config.to_yaml(path)

        loaded = TrainingConfig.from_yaml(path)
        assert loaded.algorithm == "cql"
        assert loaded.learning_rate == 0.001
        assert loaded.seed == 123

    def test_from_dict_ignores_unknown(self) -> None:
        """from_dict should ignore unknown keys."""
        config = TrainingConfig.from_dict({"algorithm": "iql", "unknown_key": 42})
        assert config.algorithm == "iql"

    def test_resolve_device(self) -> None:
        """Device resolution should return valid string."""
        config = TrainingConfig(device="cpu")
        assert config.resolve_device() == "cpu"

        config_auto = TrainingConfig(device="auto")
        device = config_auto.resolve_device()
        assert device in ("cpu", "cuda")
