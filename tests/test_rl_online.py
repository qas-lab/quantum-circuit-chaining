"""Tests for conservative online rollout helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from benchmarks.ai_transpile.rl_training.dataset import (
    OfflineRLDataset,
    concat_datasets,
    filter_dataset_by_circuit_kind,
)
from benchmarks.ai_transpile.rl_training.normalization import NormalizationStats
from benchmarks.ai_transpile.rl_training.online import record_rollout, rollout_policy, summarize_rollouts
from benchmarks.ai_transpile.rl_trajectory.database import CircuitRecord, TrajectoryDatabase
from benchmarks.ai_transpile.transpilers import CircuitMetrics


class _DummyPolicy(torch.nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        logits = torch.tensor([[3.0, 1.0]], dtype=torch.float32, device=state.device)
        return logits.repeat(batch, 1)


class _DummyTrainer:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.policy = _DummyPolicy()

    def _set_train_mode(self) -> None:
        self.policy.train()

    def _set_eval_mode(self) -> None:
        self.policy.eval()


@pytest.fixture
def norm_stats() -> NormalizationStats:
    return NormalizationStats(
        means=np.zeros(26, dtype=np.float32),
        stds=np.ones(26, dtype=np.float32),
        count=10,
    )


@pytest.fixture
def tiny_dataset() -> OfflineRLDataset:
    return OfflineRLDataset(
        observations=np.zeros((4, 26), dtype=np.float32),
        actions=np.array([0, 1, 0, 1], dtype=np.int64),
        rewards=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        next_observations=np.ones((4, 26), dtype=np.float32),
        terminals=np.array([0, 0, 1, 1], dtype=np.float32),
        norm_stats=NormalizationStats(
            means=np.zeros(26, dtype=np.float32),
            stds=np.ones(26, dtype=np.float32),
            count=4,
        ),
        action_map={1: 0, 2: 1},
        action_names=["opt_a", "opt_b"],
        circuit_ids=np.array([1, 2, 3, 4], dtype=np.int64),
    )


@pytest.fixture
def circuit_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "circuits.db"
    db = TrajectoryDatabase(db_path)
    circuits = [
        CircuitRecord(
            id=None,
            name="orig_a",
            category="qft",
            source="local",
            qasm_path="/tmp/orig_a.qasm",
            num_qubits=4,
            initial_depth=10,
            initial_two_qubit_gates=8,
            initial_two_qubit_depth=6,
            initial_total_gates=12,
            gate_density=3.0,
            two_qubit_ratio=0.66,
        ),
        CircuitRecord(
            id=None,
            name="artifact_orig_b__tket",
            category="qft",
            source="local",
            qasm_path="/tmp/artifact.qasm",
            num_qubits=4,
            initial_depth=10,
            initial_two_qubit_gates=8,
            initial_two_qubit_depth=6,
            initial_total_gates=12,
            gate_density=3.0,
            two_qubit_ratio=0.66,
        ),
        CircuitRecord(
            id=None,
            name="orig_c",
            category="qaoa",
            source="local",
            qasm_path="/tmp/orig_c.qasm",
            num_qubits=6,
            initial_depth=12,
            initial_two_qubit_gates=7,
            initial_two_qubit_depth=5,
            initial_total_gates=14,
            gate_density=2.3,
            two_qubit_ratio=0.5,
        ),
        CircuitRecord(
            id=None,
            name="artifact_orig_d__qiskit_ai",
            category="qaoa",
            source="local",
            qasm_path="/tmp/artifact_d.qasm",
            num_qubits=6,
            initial_depth=12,
            initial_two_qubit_gates=7,
            initial_two_qubit_depth=5,
            initial_total_gates=14,
            gate_density=2.3,
            two_qubit_ratio=0.5,
        ),
    ]
    for circuit in circuits:
        db.insert_circuit(circuit)
    db.close()
    return db_path


def test_filter_dataset_by_circuit_kind(tiny_dataset: OfflineRLDataset, circuit_db: Path) -> None:
    original = filter_dataset_by_circuit_kind(tiny_dataset, circuit_db, "original")
    artifact = filter_dataset_by_circuit_kind(tiny_dataset, circuit_db, "artifact")

    assert len(original) == 2
    assert len(artifact) == 2
    assert original.circuit_ids is not None
    assert artifact.circuit_ids is not None
    assert original.circuit_ids.tolist() == [1, 3]
    assert artifact.circuit_ids.tolist() == [2, 4]


def test_concat_datasets_repeats_online_samples(tiny_dataset: OfflineRLDataset) -> None:
    online = OfflineRLDataset(
        observations=np.full((1, 26), 2.0, dtype=np.float32),
        actions=np.array([1], dtype=np.int64),
        rewards=np.array([0.9], dtype=np.float32),
        next_observations=np.full((1, 26), 3.0, dtype=np.float32),
        terminals=np.array([1], dtype=np.float32),
        norm_stats=tiny_dataset.norm_stats,
        action_map=tiny_dataset.action_map,
        action_names=tiny_dataset.action_names,
        circuit_ids=np.array([99], dtype=np.int64),
    )

    mixed = concat_datasets([tiny_dataset, online], repeat_factors=[1, 3])

    assert len(mixed) == len(tiny_dataset) + 3
    assert mixed.circuit_ids is not None
    assert mixed.circuit_ids.tolist()[-3:] == [99, 99, 99]


def test_rollout_policy_stops_on_degradation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    norm_stats: NormalizationStats,
) -> None:
    qasm_path = tmp_path / "circuit.qasm"
    qasm_path.write_text("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\n")
    circuit = CircuitRecord(
        id=7,
        name="orig_rollout",
        category="qft",
        source="local",
        qasm_path=str(qasm_path),
        num_qubits=4,
        initial_depth=12,
        initial_two_qubit_gates=10,
        initial_two_qubit_depth=8,
        initial_total_gates=15,
        gate_density=3.75,
        two_qubit_ratio=0.66,
    )

    initial_metrics = CircuitMetrics(depth=12, two_qubit_gates=10, two_qubit_depth=8, total_gates=15)
    next_metrics = [
        CircuitMetrics(depth=10, two_qubit_gates=6, two_qubit_depth=5, total_gates=12),
        CircuitMetrics(depth=11, two_qubit_gates=8, two_qubit_depth=6, total_gates=13),
    ]
    analyze_calls = {"count": 0}
    execute_calls = {"count": 0}

    def fake_analyze(_circuit):
        analyze_calls["count"] += 1
        return initial_metrics

    def fake_execute_chain(_circuit, steps, chain_name, output_dir, save_intermediates):
        idx = execute_calls["count"]
        execute_calls["count"] += 1
        return SimpleNamespace(
            step_results=[
                SimpleNamespace(
                    output_metrics=next_metrics[idx],
                    duration_seconds=1.5,
                    artifact_path=None,
                )
            ],
            final_circuit=object(),
        )

    monkeypatch.setattr("benchmarks.ai_transpile.rl_training.online.analyze_circuit", fake_analyze)
    monkeypatch.setattr("benchmarks.ai_transpile.rl_training.online.execute_chain", fake_execute_chain)

    rollout = rollout_policy(
        trainer=_DummyTrainer(),
        circuit_record=circuit,
        action_names=["wisq_rules", "tket"],
        norm_stats=norm_stats,
        max_steps=3,
        time_budget=30.0,
        degradation_threshold=0.2,
    )

    assert rollout["success"] is True
    assert rollout["num_steps"] == 2
    assert rollout["terminated_reason"] == "degradation_threshold"
    assert rollout["optimizers"] == ["wisq_rules", "wisq_rules"]


def test_record_rollout_persists_online_trajectory(tmp_path: Path) -> None:
    db = TrajectoryDatabase(tmp_path / "online.db")
    circuit = CircuitRecord(
        id=None,
        name="orig_record",
        category="qaoa",
        source="local",
        qasm_path=str(tmp_path / "orig_record.qasm"),
        num_qubits=5,
        initial_depth=10,
        initial_two_qubit_gates=8,
        initial_two_qubit_depth=6,
        initial_total_gates=12,
        gate_density=2.4,
        two_qubit_ratio=0.66,
    )
    rollout = {
        "success": True,
        "chain_name": "online_wisq_rules__tket",
        "num_steps": 2,
        "circuit_kind": "original",
        "terminated_reason": "max_steps",
        "optimizers": ["wisq_rules", "tket"],
        "initial_2q": 8,
        "final_2q": 3,
        "improvement_pct": 62.5,
        "total_duration_s": 3.0,
        "total_reward_efficiency": 0.4,
        "per_step": [
            {
                "step_index": 0,
                "optimizer": "wisq_rules",
                "duration_s": 1.0,
                "state_time_budget_remaining": 30.0,
                "next_state_time_budget_remaining": 29.0,
                "reward_improvement_only": 0.25,
                "reward_efficiency": 0.2,
                "reward_multi_objective": 0.2,
                "reward_sparse_final": 0.0,
                "reward_efficiency_normalized": 0.2,
                "state_metrics": {"depth": 10, "two_qubit_gates": 8, "two_qubit_depth": 6, "total_gates": 12},
                "next_state_metrics": {"depth": 8, "two_qubit_gates": 6, "two_qubit_depth": 5, "total_gates": 10},
            },
            {
                "step_index": 1,
                "optimizer": "tket",
                "duration_s": 2.0,
                "state_time_budget_remaining": 29.0,
                "next_state_time_budget_remaining": 27.0,
                "reward_improvement_only": 0.5,
                "reward_efficiency": 0.2,
                "reward_multi_objective": 0.2,
                "reward_sparse_final": 0.5,
                "reward_efficiency_normalized": 0.2,
                "state_metrics": {"depth": 8, "two_qubit_gates": 6, "two_qubit_depth": 5, "total_gates": 10},
                "next_state_metrics": {"depth": 6, "two_qubit_gates": 3, "two_qubit_depth": 3, "total_gates": 7},
            },
        ],
    }

    trajectory_id = record_rollout(db, circuit, rollout, action_names=["wisq_rules", "tket"])

    assert trajectory_id > 0
    stats = db.get_statistics()
    assert stats["num_trajectories"] == 1
    assert stats["num_trajectory_steps"] == 2
    db.close()


def test_summarize_rollouts_groups_by_kind() -> None:
    summary = summarize_rollouts(
        [
            {"success": True, "circuit_kind": "original", "improvement": 0.2, "num_steps": 2},
            {"success": True, "circuit_kind": "artifact", "improvement": -0.1, "num_steps": 1},
            {"success": False, "circuit_kind": "original"},
        ]
    )

    assert summary["num_executed"] == 2
    assert summary["by_circuit_kind"]["original"]["num_executed"] == 1
    assert summary["by_circuit_kind"]["artifact"]["num_executed"] == 1
