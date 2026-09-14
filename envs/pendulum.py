import numpy as np
import torch


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def angle_normalize_np(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class Pendulum:
    def __init__(
        self, 
        num_envs: int = 1,
        max_episode_steps: int = 500,
        seed: int = 42,
        g=10.0
    ):
        self.n_envs = num_envs
        self.max_episode_steps = max_episode_steps

        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.obs_dim = 3
        self.n_actions = 3

        self.rng = np.random.default_rng(seed)

        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.high = np.array([DEFAULT_X, DEFAULT_Y], dtype=np.float32)
        self.low = -self.high

    def step(self, u: torch.Tensor):
        u = u.cpu().numpy().astype(np.float32)
        u = (u - 1) * self.max_torque
        th = self.state[:, 0]  # th := theta
        thdot = self.state[:, 1]

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        self.steps += 1
        truncated = self.steps >= self.max_episode_steps

        costs = angle_normalize_np(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        new_state = np.stack((newth, newthdot), axis=1)
        self.state = np.where(truncated[:, None], self._rand_uniform((self.n_envs, 2)), new_state)
        self.steps[truncated] = 0
        terminated = np.zeros((self.n_envs,), dtype=np.bool_)

        return (self._get_obs(self.state[:, 0], self.state[:, 1]), self._get_obs(newth, newthdot)), -costs, terminated, truncated, {}

    def _rand_uniform(self, shape):
        return (self.high - self.low) * self.rng.random(shape, dtype=np.float32) + self.low

    def _get_obs(self, theta, thetadot):
        return np.stack((np.cos(theta), np.sin(theta), thetadot), axis=1)

    def reset(self):
        self.state = self._rand_uniform((self.n_envs, 2))
        self.steps = np.zeros((self.n_envs,), dtype=np.int32)
        return self._get_obs(self.state[:, 0], self.state[:, 1]), {}
