from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dp import(
    create_random_policy,
    extract_greedy_policy,
    policy_evaluation,
    value_iteration,
)
from env import GridWorld
from visualize import format_policy, plot_values

def main() -> None:

    env = GridWorld()
    random_policy = create_random_policy(env)

    random_values = policy_evaluation(
        env,
        random_policy
    )

    optimal_values = value_iteration(env)

    optimal_policy = extract_greedy_policy(
        env,
        optimal_values
    )

    policy_text = format_policy(
        env,
        optimal_policy
    )

    print("随机策略状态价值：")
    print(np.round(random_values, 2))

    print("\n最优状态价值：")
    print(np.round(optimal_values, 2))

    print("\n最优策略：")
    print(policy_text)

    project_dir = Path(__file__).resolve().parent

    output_dir = project_dir/ "outputs"
    output_dir.mkdir(exist_ok=True)

    fig, ax = plot_values(
        env,
        optimal_values,
    )

    output_path = output_dir / "value_heatmap.png"

    fig.savefig(
        output_path,
        dpi = 150,
        bbox_inches="tight"
    )

    print(f"\n热力图已保存到：{output_path}")

    plt.show()

if __name__ == "__main__":
    main()





