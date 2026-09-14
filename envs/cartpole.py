import math
import numpy as np
import torch


class CartPole:
    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 500,
        seed: int = 42,
    ):
        self.n_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.rng = np.random.default_rng(seed)

        self.obs_dim = 4
        self.obs_dtype = np.float32
        self.n_actions = 2

        # Physics constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        # Termination thresholds
        self.theta_threshold_radians = np.array(12 * 2 * math.pi / 360, dtype=np.float32)
        self.x_threshold = np.array(2.4, dtype=np.float32)

        self.low = -0.05
        self.high = 0.05

        # State: [num_envs, 4]
        self.state = None

        self.steps = np.zeros(num_envs, dtype=np.int32)

    def _rand_uniform(self, shape):
        return ((self.high - self.low) * self.rng.random(
            shape, 
        ) + self.low).astype(np.float32)

    def step(self, action: torch.Tensor):
        """
        action: shape [num_envs], values in {0,1} or continuous in [0,1]
        """
        action = action.cpu().numpy().astype(np.float32)
        x = self.state[:, 0]
        x_dot = self.state[:, 1]
        theta = self.state[:, 2]
        theta_dot = self.state[:, 3]

        force = np.sign(action - 0.5) * self.force_mag

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            force + self.polemass_length * np.pow(theta_dot, 2) * sintheta
        ) / self.total_mass

        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.pow(costheta, 2) / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.stack((x, x_dot, theta, theta_dot), axis=1)

        terminated = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        self.steps += 1
        truncated = self.steps >= self.max_episode_steps

        reward = np.ones_like(terminated, dtype=np.float32)

        true_next_state = self.state.copy()

        done = terminated | truncated
        self.state = np.where(done[:, np.newaxis], self._rand_uniform((self.n_envs, 4)), self.state)
        self.steps[done] = 0

        return (self.state, true_next_state), reward, terminated, truncated, {}

    def reset(self):
        self.state = self._rand_uniform((self.n_envs, 4))
        self.steps[:] = 0
        return self.state, {}
