import numpy as np
import torch


class Acrobot:
    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 500,
        seed: int = 42,
    ):
        self.n_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.rng = np.random.default_rng(seed)

        self.obs_dim = 6
        self.obs_dtype = np.float32
        self.n_actions = 3

        # Physics constants
        self.dt = 0.2
        self.LINK_LENGTH_1 = 1.0
        self.LINK_LENGTH_2 = 1.0
        self.LINK_MASS_1 = 1.0
        self.LINK_MASS_2 = 1.0
        self.LINK_COM_POS_1 = 0.5
        self.LINK_COM_POS_2 = 0.5
        self.LINK_MOI = 1.0
        self.gravity = 9.8

        self.MAX_VEL_1 = 4 * np.pi
        self.MAX_VEL_2 = 9 * np.pi

        # State: [num_envs, 4] -> [theta1, theta2, dtheta1, dtheta2]
        self.state = None
        self.steps = np.zeros(num_envs, dtype=np.int32)

        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = self.gravity

        self.d1_part1 = m1 * lc1**2 + m2
        self.d1_part2 = l1**2 + lc2**2
        self.d1_part3 = 2 * l1 * lc2
        self.d1_part4 = I1 + I2
        self.d1_part1_2 = self.d1_part1 * self.d1_part2
        self.d1_part1_2_4 = self.d1_part1_2 + self.d1_part4
        self.d1_part1_3 = self.d1_part1 * self.d1_part3

        self.phi1_part1 = -m2 * l1 * lc2 
        self.phi1_part2 = - 2 * m2 * l1 * lc2
        self.phi1_part3 = (m1 * lc1 + m2 * l1) * g

        self.phi2_coeff = m2 * lc2 * g
        
        self.d2_part1 = m2 * lc2**2 + I2
        self.d2_part2 = m2 * l1 * lc2

        self.t2_part1 = m2 * l1 * lc2
        self.t2_part2 = m2 * lc2**2 + I2

        # Grouping all purely constant parts of d1
        self.d1_const = (m1 * lc1**2 + 
                        m2 * (l1**2 + lc2**2) + 
                        I1 + I2)
        # The coefficient for cos(theta2) in d1
        self.d1_cos_coeff = 2 * m2 * l1 * lc2

        # For d2
        self.d2_const = m2 * lc2**2 + I2
        self.d2_cos_coeff = m2 * l1 * lc2

        # For phi1
        self.phi1_v2_coeff = -m2 * l1 * lc2
        self.phi1_v1v2_coeff = -2 * m2 * l1 * lc2
        self.phi1_gravity_coeff = (m1 * lc1 + m2 * l1) * 9.8

        # For phi2
        self.phi2_coeff = m2 * lc2 * 9.8

        # For the final ddtheta2 acceleration
        self.ddtheta2_v1_coeff = m2 * l1 * lc2
        self.ddtheta2_denom_const = m2 * lc2**2 + I2

    def _get_ob(self, state):
        """Converts internal state [theta1, theta2, d1, d2] to Gym observations."""
        theta1 = state[:, 0]
        theta2 = state[:, 1]
        dtheta1 = state[:, 2]
        dtheta2 = state[:, 3]
        return np.stack([
            np.cos(theta1), np.sin(theta1),
            np.cos(theta2), np.sin(theta2),
            dtheta1, dtheta2
        ], axis=1).astype(np.float32)

    def _dsdt(self, s, torque):
        theta1, theta2, dtheta1, dtheta2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

        _cos_theta2 = np.cos(theta2)
        _sin_theta2 = np.sin(theta2)

        d1 = self.d1_const + self.d1_cos_coeff * _cos_theta2
        d2 = self.d2_const + self.d2_cos_coeff * _cos_theta2

        phi2 = self.phi2_coeff * np.cos(theta1 + theta2 - np.pi / 2.0)

        phi1 = (
            self.phi1_v2_coeff * (dtheta2**2) * _sin_theta2 +
            self.phi1_v1v2_coeff * dtheta1 * dtheta2 * _sin_theta2 +
            self.phi1_gravity_coeff * np.cos(theta1 - np.pi / 2.0) +
            phi2
        )

        num = (torque + (d2 / d1) * phi1 - 
            self.ddtheta2_v1_coeff * (dtheta1**2) * _sin_theta2 - 
            phi2)
        denom = self.ddtheta2_denom_const - (d2**2 / d1)
        
        ddtheta2 = num / denom
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        return np.stack([dtheta1, dtheta2, ddtheta1, ddtheta2], axis=1)

    def _rk4(self, s, torque):
        """Vectorized 4th-order Runge-Kutta integration."""
        dt = self.dt
        dt2 = dt / 2.0

        k1 = self._dsdt(s, torque)
        k2 = self._dsdt(s + dt2 * k1, torque)
        k3 = self._dsdt(s + dt2 * k2, torque)
        k4 = self._dsdt(s + dt * k3, torque)
        
        return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _rand_state(self, n):
        return self.rng.uniform(low=-0.1, high=0.1, size=(n, 4)).astype(np.float32)

    def step(self, action: torch.Tensor):
        # Convert action tensor [0, 1, 2] to torques [-1, 0, 1]
        action_np = action.cpu().numpy().astype(np.float32)
        torque = action_np - 1.0 

        # Integration
        new_state = self._rk4(self.state, torque)

        # Constraints: Wrap angles and clip velocities
        new_state[:, 0] = (new_state[:, 0] + np.pi) % (2 * np.pi) - np.pi
        new_state[:, 1] = (new_state[:, 1] + np.pi) % (2 * np.pi) - np.pi
        new_state[:, 2] = np.clip(new_state[:, 2], -self.MAX_VEL_1, self.MAX_VEL_1)
        new_state[:, 3] = np.clip(new_state[:, 3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = new_state

        terminated = -np.cos(self.state[:, 0]) - np.cos(self.state[:, 0] + self.state[:, 1]) > 1.0
        
        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        done = terminated | truncated

        reward = np.where(terminated, 0.0, -1.0).astype(np.float32)
        true_next_obs = self._get_ob(self.state)
        
        if np.any(done):
            self.state[done] = self._rand_state(np.sum(done))
            self.steps[done] = 0

        current_obs = self._get_ob(self.state)

        return (current_obs, true_next_obs), reward, terminated, truncated, {}

    def reset(self):
        self.state = self._rand_state(self.n_envs)
        self.steps[:] = 0
        return self._get_ob(self.state), {}
