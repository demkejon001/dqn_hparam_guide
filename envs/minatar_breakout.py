# Modified from https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/breakout.py
import numpy as np
import torch


class Breakout:
    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 1000,
        seed: int = 42,
    ):
        self.n_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.rng = np.random.default_rng(seed)

        self.channels = {
            'paddle': 0,
            'ball': 1,
            'trail': 2,
            'brick': 3,
        }
        self.obs_dtype = np.float32
        self.n_actions = 3
        self.obs_dim = (4, 10, 10)

        # Environment states
        self.pos = np.zeros(num_envs, dtype=np.int32)
        self.ball_x = np.zeros(num_envs, dtype=np.int32)
        self.ball_y = np.zeros(num_envs, dtype=np.int32)
        self.last_x = np.zeros(num_envs, dtype=np.int32)
        self.last_y = np.zeros(num_envs, dtype=np.int32)
        self.ball_dir = np.zeros(num_envs, dtype=np.int32)
        self.brick_map = np.zeros((num_envs, 10, 10), dtype=np.int32)
        self.strike = np.zeros(num_envs, dtype=bool)
        self.terminal = np.zeros(num_envs, dtype=bool)
        
        self.steps = np.zeros(num_envs, dtype=np.int32)

        self.reset()

    def _get_state(self):
        """Constructs the (N, 4, 10, 10) state tensor."""
        state = np.zeros((self.n_envs, len(self.channels), 10, 10), dtype=self.obs_dtype)
        env_idx = np.arange(self.n_envs)
        
        state[env_idx, self.channels['paddle'], 9, self.pos] = 1.0
        state[env_idx, self.channels['ball'], self.ball_y, self.ball_x] = 1.0
        state[env_idx, self.channels['trail'], self.last_y, self.last_x] = 1.0
        state[:, self.channels['brick'], :, :] = self.brick_map
        
        return state

    def _reset_envs(self, mask):
        """Resets specific environments based on a boolean mask."""
        num_reset = np.sum(mask)
        if num_reset == 0:
            return

        self.ball_y[mask] = 3
        ball_start = self.rng.integers(0, 2, size=num_reset)
        
        self.ball_x[mask] = np.where(ball_start == 0, 0, 9)
        self.ball_dir[mask] = np.where(ball_start == 0, 2, 3)
        self.pos[mask] = 4
        
        self.brick_map[mask] = 0
        self.brick_map[mask, 1:4, :] = 1
        
        self.strike[mask] = False
        self.last_x[mask] = self.ball_x[mask]
        self.last_y[mask] = self.ball_y[mask]
        self.terminal[mask] = False
        self.steps[mask] = 0

    def step(self, action: torch.Tensor):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = action.astype(np.int32)

        env_idx = np.arange(self.n_envs)
        reward = np.zeros(self.n_envs, dtype=np.float32)

        pos_delta = action - 1
        self.pos = np.clip(self.pos + pos_delta, 0, 9)

        self.last_x = self.ball_x.copy()
        self.last_y = self.ball_y.copy()

        dx = np.where((self.ball_dir == 1) | (self.ball_dir == 2), 1, -1)
        dy = np.where((self.ball_dir == 2) | (self.ball_dir == 3), 1, -1)

        new_x = self.ball_x + dx
        new_y = self.ball_y + dy

        hit_left = new_x < 0
        hit_right = new_x > 9
        new_x = np.where(hit_left, 0, new_x)
        new_x = np.where(hit_right, 9, new_x)
        
        flip_x = hit_left | hit_right
        self.ball_dir = np.where(flip_x, self.ball_dir ^ 1, self.ball_dir)

        handled_y = np.zeros(self.n_envs, dtype=bool)

        hit_top = new_y < 0
        new_y = np.where(hit_top, 0, new_y)
        self.ball_dir = np.where(hit_top, self.ball_dir ^ 3, self.ball_dir)
        handled_y = handled_y | hit_top

        check_brick = (new_y >= 0) & (new_y <= 9) & ~handled_y
        
        hit_brick = np.zeros(self.n_envs, dtype=bool)
        hit_brick[check_brick] = self.brick_map[env_idx[check_brick], new_y[check_brick], new_x[check_brick]] == 1

        reward_mask = hit_brick & ~self.strike
        reward = np.where(reward_mask, 1.0, reward)
        
        self.strike = np.where(hit_brick, True, self.strike)
        
        self.brick_map[env_idx[hit_brick], new_y[hit_brick], new_x[hit_brick]] = 0
        new_y = np.where(hit_brick, self.last_y, new_y)
        self.ball_dir = np.where(hit_brick, self.ball_dir ^ 3, self.ball_dir)
        
        handled_y = handled_y | hit_brick

        self.strike = np.where(~hit_brick, False, self.strike)

        hit_bottom = (new_y == 9) & ~handled_y
        
        brick_counts = np.sum(self.brick_map, axis=(1, 2))
        replenish = hit_bottom & (brick_counts == 0)
        self.brick_map[replenish, 1:4, :] = 1

        hit_center = hit_bottom & (self.ball_x == self.pos)
        self.ball_dir = np.where(hit_center, self.ball_dir ^ 3, self.ball_dir)
        new_y = np.where(hit_center, self.last_y, new_y)

        hit_edge = hit_bottom & ~hit_center & (new_x == self.pos)
        self.ball_dir = np.where(hit_edge, self.ball_dir ^ 2, self.ball_dir)
        new_y = np.where(hit_edge, self.last_y, new_y)

        missed = hit_bottom & ~hit_center & ~hit_edge
        self.terminal = np.where(missed, True, self.terminal)

        self.ball_x = new_x
        self.ball_y = new_y

        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        terminated = self.terminal.copy()
        done = terminated | truncated

        true_next_state = self._get_state()
        self._reset_envs(done)
        state = self._get_state()

        return (state, true_next_state), reward, terminated, truncated, {}

    def reset(self):
        mask = np.ones(self.n_envs, dtype=bool)
        self._reset_envs(mask)
        return self._get_state(), {}
