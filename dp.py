import numpy as np

from env import GridWorld

def create_random_policy(env: GridWorld) -> np.ndarray:
    """
    随机策略 向每个方向行动的概率平均 障碍和终点的概率是0
    """
    action_count = len(env.ACTIONS)

    policy = np.zeros(
        (env.rows, env.cols, action_count), 
        dtype = np.float64
    )

    for state in env.valid_states():
        if state == env.goal:
            continue

        row, col = state

        policy[row, col] = 1.0 / action_count

    return policy

def policy_evaluation(
        env: GridWorld,
        policy: np.ndarray,
        gamma: float = 0.9,
        theta: float = 1e-8,
) -> np.ndarray:
    values = np.zeros(
        (env.rows,env.cols),
        dtype = np.float64
    )

    while True:
        new_values = values.copy()
        delta = 0.0

        for state in env.valid_states():
            if state == env.goal:
                continue
            row, col = state
            state_value = 0.0

            for action in env.ACTIONS:
                action_pob = policy[row,col,action]

                next_state, reward, is_goal = env.transition(
                    state,
                    action,
                )

                next_row, next_col = next_state

                bellman_t = reward + gamma*values[next_row,next_col] if not is_goal else reward 
                v_a = action_pob*bellman_t
                state_value += v_a

            new_values[row,col] = state_value
            difference = abs(new_values[row, col] - values[row, col])
            delta = max(delta,difference)

        values = new_values

        if delta < theta:
            break

    return values

def value_iteration(
        env: GridWorld, 
        gamma: float = 0.9,
        theta: float = 1e-8
) -> np.ndarray:
    values = np.zeros(
        (env.rows, env.cols),
        dtype= np.float64,
    )

    while True:
        new_values = values.copy()
        delta = 0.0

        for state in env.valid_states():
            if state == env.goal:
                continue

            row, col = state
            action_values = []

            for action in env.ACTIONS:
                next_state, reward, terminated = env.transition(
                    state,
                    action,
                )

                next_row, next_col = next_state

                bellman_t = reward + gamma*values[next_row, next_col] if not terminated else reward
                action_values.append(bellman_t)

            action_max_v = max(action_values)

            new_values[row, col] = action_max_v
            difference = abs(new_values[row, col] - values[row, col])
            delta = max(difference,delta)
            

        values = new_values
        if delta < theta:
            break

    return values

def extract_greedy_policy(
        env: GridWorld,
        values: np.ndarray,
        gamma: float = 0.9
) -> np.ndarray:
    action_cnt = len(env.ACTIONS)

    policy = np.zeros(
        (env.rows, env.cols, action_cnt),
        dtype=np.float64,
    )

    for state in env.valid_states():
        if state == env.goal:
            continue

        row, col = state
        action_values = []

        for action in env.ACTIONS:
            next_state, reward, terminated = env.transition(
                state,
                action,
            )

            next_row, next_col = next_state

            bellman_t = reward + gamma*values[next_row, next_col] if not terminated else reward
            action_values.append((bellman_t, action))

        best_pair = max(
            action_values,
            key = lambda pair: pair[0],
        )
        best_value, b_action = best_pair
        policy[row, col, b_action] =1.00
        
    return policy
        

        
