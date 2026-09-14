from typing import Any, NamedTuple
import torch
import numpy as np


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    

class ReplayBuffer:
    def __init__(
        self,
        n_agents,
        buffer_size: int,
        obs_dim: int | tuple[int],
        device: torch.device | str = "auto",
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.device = device
        self.n_agents = n_agents
        self.n_envs = n_envs

        if isinstance(obs_dim, int):
            obs_dim = (obs_dim, )

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        self.obs = np.zeros((self.buffer_size, self.n_envs, self.n_agents, *obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.buffer_size, self.n_envs, self.n_agents, *obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.n_agents), dtype=np.int32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs, self.n_agents), dtype=np.float32)

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        infos: list[dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        self.obs[self.pos] = obs.swapaxes(0, 1)
        self.next_obs[self.pos] = next_obs.swapaxes(0, 1)
        self.actions[self.pos] = action.swapaxes(0, 1)
        self.rewards[self.pos] = reward.swapaxes(0, 1)
        self.dones[self.pos] = done.swapaxes(0, 1)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=(batch_size,), ) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=(batch_size,), )
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: torch.Tensor) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        data = (
            self.obs[batch_inds, env_indices].swapaxes(0, 1),
            self.actions[batch_inds, env_indices].swapaxes(0, 1),
            self.next_obs[batch_inds, env_indices].swapaxes(0, 1),
            self.dones[batch_inds, env_indices].swapaxes(0, 1),
            self.rewards[batch_inds, env_indices].swapaxes(0, 1),
        )
        return ReplayBufferSamples(*data)
