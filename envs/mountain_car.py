import numpy as np
import torch


class MountainCar:
    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 400,
        seed: int = 42,
        goal_velocity: float = 0.0,
    ):
        self.n_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

        self.obs_dim = 2
        self.obs_dtype = np.float32
        self.n_actions = 3

        # Physics constants
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        self.force = 0.001
        self.gravity = 0.0025

        # Reset bounds
        self.res_low = -0.6
        self.res_high = -0.4

        self.state = None
        self.steps = np.zeros(num_envs, dtype=np.int32)

    def _get_reset_state(self, size):
        # Position is random, velocity is always 0
        positions = self.rng.uniform(low=self.res_low, high=self.res_high, size=size)
        velocities = np.zeros(size)
        return np.stack((positions, velocities), axis=1).astype(np.float32)

    def step(self, action: torch.Tensor):
        # Convert torch actions to numpy
        # Action space: 0 (left), 1 (none), 2 (right)
        act = action.cpu().numpy().astype(np.float32)
        
        position = self.state[:, 0]
        velocity = self.state[:, 1]

        # Transition Dynamics
        velocity += (act - 1) * self.force + np.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        # Inelastic collision at left wall
        collision_mask = (position == self.min_position) & (velocity < 0)
        velocity[collision_mask] = 0

        # Check termination
        terminated = (position >= self.goal_position) & (velocity >= self.goal_velocity)
        
        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        
        reward = ((position - self.max_position) / 20.0) - 20.0 * (self.max_speed - np.abs(velocity))
        reward = np.where(terminated, 0, reward)

        # Save the "true" next state before autoreset
        self.state = np.stack((position, velocity), axis=1)
        true_next_state = self.state.copy()

        # Autoreset logic
        done = terminated | truncated
        if np.any(done):
            reset_states = self._get_reset_state(self.n_envs)
            self.state = np.where(done[:, np.newaxis], reset_states, self.state)
            self.steps[done] = 0

        return (self.state, true_next_state), reward, terminated, truncated, {}

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self._get_reset_state(self.n_envs)
        self.steps[:] = 0
        return self.state, {}
