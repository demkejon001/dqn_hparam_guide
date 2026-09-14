import numpy as np
import torch


class CliffWalking:
    def __init__(
        self, 
        num_envs: int = 1, 
        max_episode_steps: int = 200, 
        seed: int = 42, 
    ):
        self.n_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.rng = np.random.default_rng(seed)

        self.obs_dim = 2
        self.obs_dtype = np.float32
        self.n_actions = 4

        # Grid dimensions
        self.rows = 4
        self.cols = 12
        self.start_pos = np.array([3, 0])
        self.goal_pos = np.array([3, 11])
        self.start_state_idx = 36

        self.agent_pos = np.empty((num_envs, 2), dtype=np.int8)
        self.agent_pos[:] = self.start_pos

        self.steps = np.zeros(num_envs, dtype=np.int32)

        self.min_bounds = np.array([[0, 0]])
        self.max_bounds = np.array([[3, 11]])
        # Action mapping for vector operations
        # 0: Up, 1: Right, 2: Down, 3: Left
        self.deltas = np.array([
            [-1, 0], # Up
            [0, 1],  # Right
            [1, 0],  # Down
            [0, -1]  # Left
        ])

    def step(self, action: torch.Tensor):
        actions = action.cpu().numpy()

        is_slip_action = actions == 1
        slip = self.rng.random(actions.shape) < .66
        actions = np.where(is_slip_action & slip, 2, actions)
        
        # Calculate new positions
        move_deltas = self.deltas[actions]
        agent_pos = self.agent_pos + move_deltas
        agent_pos = np.clip(agent_pos, self.min_bounds, self.max_bounds)

        # Check for cliff: row 3, cols 1 through 10
        is_cliff = (agent_pos[:, 0] == 3) & (agent_pos[:, 1] > 0) & (agent_pos[:, 1] < 11)
        agent_pos = np.where(is_cliff[:, None], self.start_pos, agent_pos)
        
        is_goal = np.all(agent_pos == self.goal_pos, axis=1)

        reward = (np.sum(agent_pos, axis=-1) - 14) / 100.0
        reward = np.where(is_goal, 1.0, reward).astype(np.float32)
        
        true_next_state = agent_pos
        
        terminated = is_goal
        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        done = terminated | truncated

        self.agent_pos = np.where(done[:, np.newaxis], self.start_pos, agent_pos)
        self.steps[done] = 0
        
        return (self.agent_pos.astype(np.float32), true_next_state.astype(np.float32)), reward, terminated, truncated, {}

    def reset(self):
        self.agent_pos[:] = self.start_pos
        self.steps[:] = 0
        return self.agent_pos.astype(np.float32), {}