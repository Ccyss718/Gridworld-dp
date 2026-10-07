import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dp import extract_greedy_policy, value_iteration
from env import GridWorld
from visualize import format_policy, plot_values


def test_format_policy_returns_expected_arrow_grid():
    env = GridWorld()
    values = value_iteration(env)
    policy = extract_greedy_policy(env, values)

    result = format_policy(env, policy)

    expected = "→ → → ↓\n↓ X → ↓\n↓ X → ↓\n→ → → G"
    assert result == expected


def test_format_policy_has_environment_dimensions():
    env = GridWorld()
    values = value_iteration(env)
    policy = extract_greedy_policy(env, values)

    result = format_policy(env, policy)
    lines = result.splitlines()

    assert len(lines) == env.rows
    assert all(len(line.split()) == env.cols for line in lines)


def test_plot_values_builds_expected_heatmap_without_changing_values():
    env = GridWorld()
    values = value_iteration(env)
    values_before = values.copy()

    fig, ax = plot_values(env, values)

    try:
        np.testing.assert_array_equal(values, values_before)
        assert ax.get_title() == "GridWorld Optimal State Values"
        assert ax.get_xlabel() == "Column"
        assert ax.get_ylabel() == "Row"
        assert len(ax.images) == 1
        assert ax.images[0].get_array().shape == (env.rows, env.cols)
        assert len(ax.texts) == env.rows * env.cols

        text_by_state = {
            (int(text.get_position()[1]), int(text.get_position()[0])):
                text.get_text()
            for text in ax.texts
        }
        assert text_by_state[env.goal] == "G"
        assert all(text_by_state[obstacle] == "X" for obstacle in env.obstacles)
        assert text_by_state[env.start] == f"{values[env.start]:.2f}"
    finally:
        plt.close(fig)
