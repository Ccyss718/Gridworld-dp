State = tuple[int, int]


class GridWorld:
    ACTIONS: dict[int, State] = {
        0: (-1, 0),  # 上
        1: (0, 1),   # 右
        2: (1, 0),   # 下
        3: (0, -1),  # 左
    }

    def __init__(self) -> None:
        """
        属性
        """
        self.rows = 4
        self.cols = 4

        self.start = (0, 0)
        self.goal = (3, 3)

        self.obstacles = {
            (1, 1),
            (2, 1),
        }

        self.step_reward = -1.0
        self.goal_reward = 10.0
        self.max_steps = 30


        self.state = self.start
        self.steps = 0

    def reset(self) -> State:
        """
        归零 初始化
        """
        self.state = self.start
        self.steps = 0
        return self.state

    def _is_inside(self, position: State) -> bool:
        row, col = position

        row_is_valid = 0 <= row < self.rows
        col_is_valid = 0 <= col < self.cols

        return row_is_valid and col_is_valid

    def transition(self, state: State, action: int) -> tuple[State, float, bool]:
        """
        模拟采取action后的下一状态、收获、是否终点
        """
        if state == self.goal:
            return state, 0.0, True

        row, col = state
        row_change, col_change = self.ACTIONS[action]
        candidate = (
            row + row_change,
            col + col_change,
        )

        is_inside = self._is_inside(candidate)
        notin_obstacle = candidate not in self.obstacles

        next_state = candidate if is_inside and notin_obstacle else state 
        
        is_goal = next_state == self.goal
        reward = self.goal_reward if is_goal else self.step_reward
        return (next_state,reward,is_goal)

        

    def step(self, action: int) -> tuple[State, float, bool, bool]:
        
        next_state, reward, terminated = self.transition(self.state, action)
        self.state = next_state
        self.steps += 1

        truncated = (self.steps >= self.max_steps and not terminated)

        return self.state, reward, terminated, truncated

    def valid_states(self) -> list[State]:
        res = []
        for i in range(self.rows):
            for j in range(self.cols):
                if((i,j) not in self.obstacles):
                    res.append((i,j))

        return res


    def render(self) -> str:
        lines = []
        for row in range (self.rows):
            row_symbols = []

            for col in range(self.cols):
                position = (row, col)

                if position == self.state:
                    row_symbols.append("A")
                elif position == self.goal:
                    row_symbols.append("G")
                elif position in self.obstacles:
                    row_symbols.append("X")
                else:
                    row_symbols.append(".")

            lines.append(" ".join(row_symbols))

        return "\n".join(lines)
