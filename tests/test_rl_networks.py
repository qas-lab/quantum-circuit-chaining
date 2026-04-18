"""Tests for rl_training neural network architectures."""

from __future__ import annotations

import torch
from benchmarks.ai_transpile.rl_training.networks import (
    PolicyNetwork,
    QNetwork,
    ValueNetwork,
)


class TestQNetwork:
    def test_forward_shape(self) -> None:
        """Q-network should output (batch, action_dim)."""
        net = QNetwork(state_dim=22, action_dim=5, hidden_dims=(128, 128))
        x = torch.randn(8, 22)
        q = net(x)
        assert q.shape == (8, 5)

    def test_single_sample(self) -> None:
        """Should handle batch size 1."""
        net = QNetwork(state_dim=22, action_dim=5)
        x = torch.randn(1, 22)
        q = net(x)
        assert q.shape == (1, 5)

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the network."""
        net = QNetwork(state_dim=22, action_dim=5)
        x = torch.randn(4, 22)
        q = net(x)
        loss = q.sum()
        loss.backward()
        # Check that at least one parameter has gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
        assert has_grad

    def test_custom_hidden_dims(self) -> None:
        """Should work with different hidden layer configs."""
        net = QNetwork(state_dim=10, action_dim=3, hidden_dims=(64,))
        x = torch.randn(4, 10)
        q = net(x)
        assert q.shape == (4, 3)

    def test_with_dropout(self) -> None:
        """Should work with dropout enabled."""
        net = QNetwork(state_dim=22, action_dim=5, dropout=0.5)
        x = torch.randn(8, 22)
        net.train()
        q_train = net(x)
        assert q_train.shape == (8, 5)


class TestValueNetwork:
    def test_forward_shape(self) -> None:
        """Value network should output (batch, 1)."""
        net = ValueNetwork(state_dim=22)
        x = torch.randn(8, 22)
        v = net(x)
        assert v.shape == (8, 1)

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the network."""
        net = ValueNetwork(state_dim=22)
        x = torch.randn(4, 22)
        v = net(x)
        loss = v.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
        assert has_grad


class TestPolicyNetwork:
    def test_forward_shape(self) -> None:
        """Policy network should output (batch, action_dim)."""
        net = PolicyNetwork(state_dim=22, action_dim=5)
        x = torch.randn(8, 22)
        logits = net(x)
        assert logits.shape == (8, 5)

    def test_get_action(self) -> None:
        """get_action should return integer indices."""
        net = PolicyNetwork(state_dim=22, action_dim=5)
        x = torch.randn(4, 22)
        actions = net.get_action(x)
        assert actions.shape == (4,)
        assert actions.dtype == torch.long
        assert (actions >= 0).all() and (actions < 5).all()

    def test_get_log_probs(self) -> None:
        """Log-probs should sum to ~0 (in log space, exp sums to 1)."""
        net = PolicyNetwork(state_dim=22, action_dim=5)
        x = torch.randn(4, 22)
        log_probs = net.get_log_probs(x)
        assert log_probs.shape == (4, 5)
        # Check that exp(log_probs) sums to ~1
        probs_sum = log_probs.exp().sum(dim=-1)
        torch.testing.assert_close(probs_sum, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the network."""
        net = PolicyNetwork(state_dim=22, action_dim=5)
        x = torch.randn(4, 22)
        logits = net(x)
        loss = logits.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
        assert has_grad


class TestParameterCounts:
    def test_small_network_size(self) -> None:
        """Networks should be small (~20-40K params) for this problem."""
        q_net = QNetwork(state_dim=22, action_dim=5, hidden_dims=(128, 128))
        n_params = sum(p.numel() for p in q_net.parameters())
        # 22*128 + 128 + 128*128 + 128 + 128*5 + 5 = 2816 + 128 + 16384 + 128 + 640 + 5 = ~20K
        assert 10_000 < n_params < 50_000, f"Expected ~20K params, got {n_params}"
