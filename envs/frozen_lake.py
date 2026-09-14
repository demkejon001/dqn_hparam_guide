import numpy as np
import torch


class FrozenLake:
    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 200,
        is_slippery: bool = True,
        success_rate: float = 7.0 / 10.0,
        seed: int = 42,
    ):
        self.n_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.is_slippery = is_slippery
        self.success_rate = success_rate
        self.rng = np.random.default_rng(seed)

        self.obs_dim = 2
        self.obs_dtype = np.float32
        self.n_actions = 4

        self.desc = np.asarray([
            "S...........",
            ".........H..",
            "...H........",
            ".....H......",
            "...H........",
            "......H.....",
            "...........H",
            "...H........",
            ".....H......",
            "...H......H.",
            "......H.....",
            "....H.H....G"
        ], dtype="c")
        self.move_deltas = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        
        self.agent_pos = np.zeros((num_envs, 2), dtype=np.int32)
        self.steps = np.zeros(num_envs, dtype=np.int32)

        self.min_bounds = np.array([[0, 0]])
        self.max_bounds = np.array([[11, 11]])
        self.is_stuck = np.zeros((num_envs,), dtype=np.bool_)

    def step(self, action: torch.Tensor):
        action = action.cpu().numpy()
        
        if self.is_slippery:
            probs = [self.success_rate, (1-self.success_rate)/2, (1-self.success_rate)/2]
            offset = self.rng.choice([0, -1, 1], size=self.n_envs, p=probs)
            action = (action + offset) % 4
        
        move_deltas = self.move_deltas[action]
        agent_pos = self.agent_pos + move_deltas
        agent_pos = np.clip(agent_pos, self.min_bounds, self.max_bounds)

        is_hole = self.desc[agent_pos[:, 0], agent_pos[:, 1]] == b'H'
        is_goal = self.desc[agent_pos[:, 0], agent_pos[:, 1]] == b'G'

        terminated = is_goal

        reward = (-22 + (agent_pos[:, 0] + agent_pos[:, 1])) / 44.0
        reward = np.where(is_goal, 1.0, reward)
        
        agent_pos = np.where(is_hole[:, None], 0, agent_pos)

        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
            
        true_next_state = agent_pos

        done = terminated | truncated
        self.agent_pos = np.where(done[:, None], 0, agent_pos)
        self.steps[done] = 0

        return (self.agent_pos.astype(np.float32), true_next_state.astype(np.float32)), reward, terminated, truncated, {}

    def reset(self):
        self.agent_pos[:] = 0
        self.steps[:] = 0
        self.is_stuck[:] = False
        return self.agent_pos.astype(np.float32), {}

