import numpy as np

from dp import (
    create_random_policy,
    extract_greedy_policy,
    policy_evaluation,
    value_iteration,
)
from env import GridWorld


def test_random_policy_has_valid_probabilities():
    env = GridWorld()

    policy = create_random_policy(env)

    assert policy.shape == (env.rows, env.cols, len(env.ACTIONS))
    assert policy.dtype == np.float64

    for state in env.valid_states():
        probability_sum = policy[state].sum()
        expected_sum = 0.0 if state == env.goal else 1.0
        assert np.isclose(probability_sum, expected_sum)

    for obstacle in env.obstacles:
        np.testing.assert_array_equal(
            policy[obstacle],
            np.zeros(len(env.ACTIONS)),
        )


def test_policy_evaluation_returns_finite_values_without_mutating_env():
    env = GridWorld()
    policy = create_random_policy(env)
    state_before = env.state
    steps_before = env.steps

    values = policy_evaluation(env, policy)

    assert values.shape == (env.rows, env.cols)
    assert np.isfinite(values).all()
    assert values[env.goal] == 0.0
    assert all(values[obstacle] == 0.0 for obstacle in env.obstacles)
    assert env.state == state_before
    assert env.steps == steps_before


def test_evaluated_values_satisfy_bellman_expectation_equation():
    env = GridWorld()
    policy = create_random_policy(env)
    gamma = 0.9

    values = policy_evaluation(env, policy, gamma=gamma, theta=1e-10)

    for state in env.valid_states():
        if state == env.goal:
            continue

        expected_value = 0.0
        for action in env.ACTIONS:
            next_state, reward, terminated = env.transition(state, action)
            future_value = 0.0 if terminated else values[next_state]
            expected_value += policy[state][action] * (
                reward + gamma * future_value
            )

        assert np.isclose(values[state], expected_value, atol=1e-8)


def test_zero_discount_uses_only_immediate_rewards():
    env = GridWorld()
    policy = create_random_policy(env)

    values = policy_evaluation(env, policy, gamma=0.0)

    for state in env.valid_states():
        if state == env.goal:
            continue

        expected_value = sum(
            policy[state][action] * env.transition(state, action)[1]
            for action in env.ACTIONS
        )
        assert np.isclose(values[state], expected_value)


def test_value_iteration_returns_finite_values_without_mutating_env():
    env = GridWorld()
    state_before = env.state
    steps_before = env.steps

    values = value_iteration(env)

    assert values.shape == (env.rows, env.cols)
    assert np.isfinite(values).all()
    assert values[env.goal] == 0.0
    assert all(values[obstacle] == 0.0 for obstacle in env.obstacles)
    assert env.state == state_before
    assert env.steps == steps_before


def test_optimal_values_satisfy_bellman_optimality_equation():
    env = GridWorld()
    gamma = 0.9

    values = value_iteration(env, gamma=gamma, theta=1e-10)

    for state in env.valid_states():
        if state == env.goal:
            continue

        action_values = []
        for action in env.ACTIONS:
            next_state, reward, terminated = env.transition(state, action)
            future_value = 0.0 if terminated else values[next_state]
            action_values.append(reward + gamma * future_value)

        assert np.isclose(values[state], max(action_values), atol=1e-8)


def test_optimal_values_are_not_below_random_policy_values():
    env = GridWorld()
    random_policy = create_random_policy(env)

    random_values = policy_evaluation(env, random_policy)
    optimal_values = value_iteration(env)

    for state in env.valid_states():
        assert optimal_values[state] >= random_values[state] - 1e-8


def test_greedy_policy_has_one_action_for_each_nonterminal_state():
    env = GridWorld()
    values = value_iteration(env)

    policy = extract_greedy_policy(env, values)

    assert policy.shape == (env.rows, env.cols, len(env.ACTIONS))
    for state in env.valid_states():
        expected_sum = 0.0 if state == env.goal else 1.0
        assert np.isclose(policy[state].sum(), expected_sum)
        assert np.all((policy[state] == 0.0) | (policy[state] == 1.0))

    for obstacle in env.obstacles:
        assert policy[obstacle].sum() == 0.0


def test_greedy_policy_selects_a_bellman_maximizing_action():
    env = GridWorld()
    gamma = 0.9
    values = value_iteration(env, gamma=gamma)

    policy = extract_greedy_policy(env, values, gamma=gamma)

    for state in env.valid_states():
        if state == env.goal:
            continue

        selected_action = int(np.argmax(policy[state]))
        action_values = []
        for action in env.ACTIONS:
            next_state, reward, terminated = env.transition(state, action)
            future_value = 0.0 if terminated else values[next_state]
            action_values.append(reward + gamma * future_value)

        assert np.isclose(
            action_values[selected_action],
            max(action_values),
        )


def test_greedy_policy_reaches_goal_before_time_limit():
    env = GridWorld()
    values = value_iteration(env)
    policy = extract_greedy_policy(env, values)
    state = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = int(np.argmax(policy[state]))
        state, _, terminated, truncated = env.step(action)

    assert state == env.goal
    assert terminated is True
    assert truncated is False
