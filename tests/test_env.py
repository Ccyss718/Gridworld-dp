from env import GridWorld


def test_reset_restores_initial_state():
    env = GridWorld()
    env.state = (2, 3)
    env.steps = 12

    result = env.reset()

    assert result == env.start
    assert env.state == env.start
    assert env.steps == 0


def test_transition_moves_to_free_cell():
    env = GridWorld()

    next_state, reward, terminated = env.transition((0, 0), 1)

    assert next_state == (0, 1)
    assert reward == env.step_reward
    assert terminated is False


def test_transition_stays_when_hitting_boundary():
    env = GridWorld()

    next_state, reward, terminated = env.transition((0, 0), 0)

    assert next_state == (0, 0)
    assert reward == env.step_reward
    assert terminated is False


def test_transition_stays_when_hitting_obstacle():
    env = GridWorld()

    next_state, reward, terminated = env.transition((1, 0), 1)

    assert next_state == (1, 0)
    assert reward == env.step_reward
    assert terminated is False


def test_transition_reaching_goal_returns_goal_reward():
    env = GridWorld()

    next_state, reward, terminated = env.transition((3, 2), 1)

    assert next_state == env.goal
    assert reward == env.goal_reward
    assert terminated is True


def test_goal_is_absorbing_terminal_state():
    env = GridWorld()

    next_state, reward, terminated = env.transition(env.goal, 3)

    assert next_state == env.goal
    assert reward == 0.0
    assert terminated is True


def test_step_updates_internal_state_and_step_count():
    env = GridWorld()

    state, reward, terminated, truncated = env.step(1)

    assert state == (0, 1)
    assert env.state == state
    assert env.steps == 1
    assert reward == env.step_reward
    assert terminated is False
    assert truncated is False


def test_step_truncates_at_max_steps():
    env = GridWorld()
    env.max_steps = 2

    first_result = env.step(0)
    second_result = env.step(0)

    assert first_result[3] is False
    assert second_result[3] is True
    assert second_result[2] is False


def test_reaching_goal_at_time_limit_is_terminated_not_truncated():
    env = GridWorld()
    env.state = (3, 2)
    env.steps = env.max_steps - 1

    state, reward, terminated, truncated = env.step(1)

    assert state == env.goal
    assert reward == env.goal_reward
    assert terminated is True
    assert truncated is False


def test_valid_states_excludes_obstacles_only():
    env = GridWorld()

    states = env.valid_states()

    assert len(states) == env.rows * env.cols - len(env.obstacles)
    assert env.start in states
    assert env.goal in states
    assert all(obstacle not in states for obstacle in env.obstacles)


def test_render_contains_agent_goal_and_obstacles():
    env = GridWorld()

    picture = env.render()

    assert picture == "A . . .\n. X . .\n. X . .\n. . . G"
