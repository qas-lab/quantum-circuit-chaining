"""Tests for rl_training algorithms (BC, CQL, IQL).

Each test runs a single training step to verify the algorithm works
end-to-end: forward pass, loss computation, gradient step.
"""

from __future__ import annotations

import torch
from benchmarks.ai_transpile.rl_training.algorithms.behavioral_cloning import BehavioralCloning
from benchmarks.ai_transpile.rl_training.algorithms.cql import CQL
from benchmarks.ai_transpile.rl_training.algorithms.iql import IQL
from benchmarks.ai_transpile.rl_training.config import TrainingConfig


def _make_batch(batch_size: int = 8, state_dim: int = 26, action_dim: int = 5) -> dict[str, torch.Tensor]:
    """Create a synthetic batch for testing."""
    return {
        "observations": torch.randn(batch_size, state_dim),
        "actions": torch.randint(0, action_dim, (batch_size,)),
        "rewards": torch.randn(batch_size),
        "next_observations": torch.randn(batch_size, state_dim),
        "terminals": torch.zeros(batch_size),
    }


class TestBehavioralCloning:
    def test_train_step(self) -> None:
        """BC training step should return loss and accuracy."""
        config = TrainingConfig(algorithm="bc", device="cpu")
        trainer = BehavioralCloning(config)
        batch = _make_batch()

        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] > 0

    def test_select_action(self) -> None:
        """select_action should return valid action index."""
        config = TrainingConfig(algorithm="bc", device="cpu")
        trainer = BehavioralCloning(config)
        state = torch.randn(26)

        action = trainer.select_action(state)
        assert 0 <= action < 5

    def test_state_dict_roundtrip(self) -> None:
        """Save and load state dict."""
        config = TrainingConfig(algorithm="bc", device="cpu")
        trainer = BehavioralCloning(config)

        # Train one step
        batch = _make_batch()
        trainer.train_step(batch)

        # Save and restore
        state = trainer.state_dict()
        trainer2 = BehavioralCloning(config)
        trainer2.load_state_dict(state)

        # Should produce same output
        test_state = torch.randn(26)
        a1 = trainer.select_action(test_state)
        a2 = trainer2.select_action(test_state)
        assert a1 == a2

    def test_loss_decreases(self) -> None:
        """Loss should decrease over multiple steps on same data."""
        config = TrainingConfig(algorithm="bc", device="cpu", learning_rate=0.01)
        trainer = BehavioralCloning(config)
        batch = _make_batch(batch_size=32)

        losses = []
        for _ in range(20):
            metrics = trainer.train_step(batch)
            losses.append(metrics["loss"])

        assert losses[-1] < losses[0], "Loss should decrease"


class TestCQL:
    def test_train_step(self) -> None:
        """CQL training step should return all expected metrics."""
        config = TrainingConfig(algorithm="cql", device="cpu")
        trainer = CQL(config)
        batch = _make_batch()

        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert "td_loss" in metrics
        assert "cql_loss" in metrics
        assert "mean_q" in metrics
        assert "accuracy" in metrics

    def test_select_action(self) -> None:
        """select_action should return valid action index."""
        config = TrainingConfig(algorithm="cql", device="cpu")
        trainer = CQL(config)
        state = torch.randn(26)

        action = trainer.select_action(state)
        assert 0 <= action < 5

    def test_target_network_different(self) -> None:
        """After training, target should lag behind online network."""
        config = TrainingConfig(algorithm="cql", device="cpu")
        trainer = CQL(config)
        batch = _make_batch()

        # Train a few steps to diverge target from online
        for _ in range(5):
            trainer.train_step(batch)

        # Target and online should differ (target lags via soft update)
        online_params = list(trainer.q_network.parameters())
        target_params = list(trainer.target_network.parameters())
        any_different = any(
            not torch.equal(op, tp) for op, tp in zip(online_params, target_params)
        )
        assert any_different

    def test_state_dict_roundtrip(self) -> None:
        """Save and load CQL state dict."""
        config = TrainingConfig(algorithm="cql", device="cpu")
        trainer = CQL(config)
        batch = _make_batch()
        trainer.train_step(batch)

        state = trainer.state_dict()
        assert "q_network" in state
        assert "target_network" in state

        trainer2 = CQL(config)
        trainer2.load_state_dict(state)


class TestIQL:
    def test_train_step(self) -> None:
        """IQL training step should return all expected metrics."""
        config = TrainingConfig(algorithm="iql", device="cpu")
        trainer = IQL(config)
        batch = _make_batch()

        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert "v_loss" in metrics
        assert "q_loss" in metrics
        assert "policy_loss" in metrics
        assert "mean_advantage" in metrics

    def test_select_action(self) -> None:
        """select_action should return valid action index."""
        config = TrainingConfig(algorithm="iql", device="cpu")
        trainer = IQL(config)
        state = torch.randn(26)

        action = trainer.select_action(state)
        assert 0 <= action < 5

    def test_state_dict_roundtrip(self) -> None:
        """Save and load IQL state dict."""
        config = TrainingConfig(algorithm="iql", device="cpu")
        trainer = IQL(config)
        batch = _make_batch()
        trainer.train_step(batch)

        state = trainer.state_dict()
        assert "q_network" in state
        assert "value_network" in state
        assert "policy" in state

        trainer2 = IQL(config)
        trainer2.load_state_dict(state)

        # Should produce same output
        test_state = torch.randn(26)
        a1 = trainer.select_action(test_state)
        a2 = trainer2.select_action(test_state)
        assert a1 == a2

    def test_three_networks_update(self) -> None:
        """All three IQL networks should have gradient updates."""
        config = TrainingConfig(algorithm="iql", device="cpu")
        trainer = IQL(config)
        batch = _make_batch()

        # Store initial params
        q_init = {n: p.clone() for n, p in trainer.q_network.named_parameters()}
        v_init = {n: p.clone() for n, p in trainer.value_network.named_parameters()}
        p_init = {n: p.clone() for n, p in trainer.policy.named_parameters()}

        trainer.train_step(batch)

        # All networks should have changed
        q_changed = any(
            not torch.equal(p, q_init[n]) for n, p in trainer.q_network.named_parameters()
        )
        v_changed = any(
            not torch.equal(p, v_init[n]) for n, p in trainer.value_network.named_parameters()
        )
        p_changed = any(
            not torch.equal(p, p_init[n]) for n, p in trainer.policy.named_parameters()
        )

        assert q_changed, "Q-network should update"
        assert v_changed, "Value network should update"
        assert p_changed, "Policy network should update"


class TestTrainingLoop:
    """Test the full training loop with DataLoader integration."""

    def test_bc_full_loop(self) -> None:
        """BC should complete a full training loop."""
        config = TrainingConfig(
            algorithm="bc", device="cpu", num_epochs=3, eval_interval=1,
        )
        trainer = BehavioralCloning(config)

        # Create a simple DataLoader
        from torch.utils.data import DataLoader, Dataset, TensorDataset

        dataset = TensorDataset(
            torch.randn(20, 26),
            torch.randint(0, 5, (20,)),
            torch.randn(20),
            torch.randn(20, 26),
            torch.zeros(20),
        )

        # Wrap to return dict
        class DictDataset(Dataset):
            def __init__(self, ds):
                self.ds = ds

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, idx):
                obs, act, rew, nobs, term = self.ds[idx]
                return {
                    "observations": obs,
                    "actions": act,
                    "rewards": rew,
                    "next_observations": nobs,
                    "terminals": term,
                }

        train_loader = DataLoader(DictDataset(dataset), batch_size=8)
        val_loader = DataLoader(DictDataset(dataset), batch_size=8)

        history = trainer.train(train_loader, val_loader, num_epochs=3)
        assert "train/loss" in history
        assert len(history["train/loss"]) == 3
