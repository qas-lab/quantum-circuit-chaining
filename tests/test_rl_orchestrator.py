"""Tests for RL orchestration prototype."""

from __future__ import annotations

from qiskit import QuantumCircuit


def test_optimization_environment_initialization() -> None:
    """Test OptimizationEnvironment initialization."""
    from benchmarks.ai_transpile.rl_orchestrator import OptimizationEnvironment

    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    env = OptimizationEnvironment(circuit)

    assert env.initial_circuit == circuit
    assert not env.done
    assert env.state.steps_taken == 0


def test_optimization_environment_reset() -> None:
    """Test environment reset."""
    from benchmarks.ai_transpile.rl_orchestrator import OptimizationEnvironment

    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)

    env = OptimizationEnvironment(circuit)

    # Take some steps
    from benchmarks.ai_transpile.rl_orchestrator import OptimizationAction

    env.step(OptimizationAction.RULES_ONLY)

    # Reset
    state = env.reset()

    assert state.steps_taken == 0
    assert not env.done
    assert env.total_reward == 0.0


def test_optimization_action_enum() -> None:
    """Test OptimizationAction enum."""
    from benchmarks.ai_transpile.rl_orchestrator import OptimizationAction

    assert OptimizationAction.RULES_ONLY.value == "rules_only"
    assert OptimizationAction.RESYNTHESIS.value == "resynthesis"
    assert OptimizationAction.END_EPISODE.value == "end_episode"


def test_environment_step() -> None:
    """Test taking a step in the environment."""
    from benchmarks.ai_transpile.rl_orchestrator import OptimizationAction, OptimizationEnvironment

    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)

    env = OptimizationEnvironment(circuit)

    state, reward, done, info = env.step(OptimizationAction.RULES_ONLY)

    assert state.steps_taken == 1
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "action" in info


def test_environment_end_episode() -> None:
    """Test ending episode early."""
    from benchmarks.ai_transpile.rl_orchestrator import OptimizationAction, OptimizationEnvironment

    circuit = QuantumCircuit(4)
    circuit.h(0)

    env = OptimizationEnvironment(circuit)

    state, reward, done, info = env.step(OptimizationAction.END_EPISODE)

    assert done
    assert env.done
    assert info["action"] == "end_episode"


def test_random_policy() -> None:
    """Test RandomPolicy."""
    from benchmarks.ai_transpile.rl_orchestrator import OptimizationAction, OptimizationState, RandomPolicy

    policy = RandomPolicy(seed=42)

    state = OptimizationState(
        circuit_metrics=None,  # type: ignore[arg-type]
        time_budget_remaining=100.0,
        steps_taken=0,
        max_steps=10,
        previous_metric=50.0,
    )

    actions = [OptimizationAction.RULES_ONLY, OptimizationAction.RESYNTHESIS]
    action = policy.select_action(state, actions)

    assert action in actions


def test_greedy_policy() -> None:
    """Test GreedyPolicy."""
    from benchmarks.ai_transpile.rl_orchestrator import GreedyPolicy, OptimizationAction, OptimizationState

    policy = GreedyPolicy()

    state = OptimizationState(
        circuit_metrics=None,  # type: ignore[arg-type]
        time_budget_remaining=100.0,
        steps_taken=0,
        max_steps=10,
        previous_metric=50.0,
    )

    actions = [OptimizationAction.RULES_ONLY, OptimizationAction.RESYNTHESIS]
    action = policy.select_action(state, actions)

    # Should prioritize rules-only (fastest)
    assert action == OptimizationAction.RULES_ONLY


def test_fixed_schedule_policy() -> None:
    """Test FixedSchedulePolicy."""
    from benchmarks.ai_transpile.rl_orchestrator import (
        FixedSchedulePolicy,
        OptimizationAction,
        OptimizationState,
    )

    policy = FixedSchedulePolicy()

    state = OptimizationState(
        circuit_metrics=None,  # type: ignore[arg-type]
        time_budget_remaining=100.0,
        steps_taken=0,
        max_steps=10,
        previous_metric=50.0,
    )

    all_actions = list(OptimizationAction)

    # First action should be RULES_ONLY
    action1 = policy.select_action(state, all_actions)
    assert action1 == OptimizationAction.RULES_ONLY

    # Second action should be RESYNTHESIS
    action2 = policy.select_action(state, all_actions)
    assert action2 == OptimizationAction.RESYNTHESIS


def test_evaluate_policy() -> None:
    """Test policy evaluation."""
    from benchmarks.ai_transpile.rl_orchestrator import (
        OptimizationConfig,
        RandomPolicy,
        evaluate_policy,
    )
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)

    policy = RandomPolicy(seed=42)
    config = OptimizationConfig(time_budget=30.0, max_steps=5)

    results = evaluate_policy(policy, circuit, config, num_episodes=3)

    assert "mean_reward" in results
    assert "mean_final_metric" in results
    assert "mean_steps" in results
    assert len(results["total_rewards"]) == 3


def test_environment_with_custom_config() -> None:
    """Test environment with custom configuration."""
    from benchmarks.ai_transpile.rl_orchestrator import OptimizationConfig, OptimizationEnvironment

    circuit = QuantumCircuit(4)
    circuit.h(0)

    config = OptimizationConfig(
        time_budget=100.0,
        max_steps=5,
        alpha=2.0,
        beta=0.5,
        gamma=0.05,
    )

    env = OptimizationEnvironment(circuit, config)

    assert env.config.time_budget == 100.0
    assert env.config.max_steps == 5
    assert env.config.alpha == 2.0


def test_get_available_actions() -> None:
    """Test getting available actions based on budget."""
    from benchmarks.ai_transpile.rl_orchestrator import OptimizationAction, OptimizationEnvironment

    circuit = QuantumCircuit(4)
    circuit.h(0)

    env = OptimizationEnvironment(circuit)
    available = env.get_available_actions()

    # Should include at least END_EPISODE
    assert OptimizationAction.END_EPISODE in available
    # Should include some optimization actions
    assert len(available) > 1

