import numpy as np
import matplotlib.pyplot as plt

from env import GridWorld


ACTION_SYMBOLS = {
    0: "↑",
    1: "→",
    2: "↓",
    3: "←",
}

def format_policy(
        env: GridWorld,
        policy:np.ndarray
) ->str:
    lines = []

    for row in range(env.rows):
        row_symbols = []

        for col in range(env.cols):
            state = (row, col)

            if state == env.goal:
                symbol = "G"
            elif state in env.obstacles:
                symbol = "X"
            else:
                action_idx = int(np.argmax(policy[state]))
                symbol = ACTION_SYMBOLS[action_idx]

            row_symbols.append(symbol)

        lines.append(" ".join(row_symbols))

    return ("\n".join(lines))


def plot_values(
        env: GridWorld,
        values: np.ndarray,
):
    display_values = values.copy()

    for obstacle in env.obstacles:
        display_values[obstacle] = np.nan

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(
        display_values,
        cmap=cmap,
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("State value")

    finite_values = display_values[np.isfinite(display_values)]
    color_threshold = (
        finite_values.min() + finite_values.max()
    )/2

    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)

            if state in env.obstacles:
                text = "X"
                text_color = "black"
            elif state == env.goal:
                text = "G"
                text_color = "white"
            else:
                value = values[row, col]
                text = f"{value:.2f}"

                if value > color_threshold:
                    text_color = "black"
                else:
                    text_color = "white"

            ax.text(
                col,
                row,
                text,
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=11,
            )

    ax.set_xticks(np.arange(env.cols))
    ax.set_yticks(np.arange(env.rows))

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("GridWorld Optimal State Values")

    ax.set_xticks(
        np.arange(-0.5, env.cols, 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, env.rows, 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.5,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    fig.tight_layout()

    return fig, ax