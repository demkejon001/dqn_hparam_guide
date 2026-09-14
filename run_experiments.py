# Modified from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
import copy
import os
import itertools
import time
from dataclasses import dataclass
import tyro
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, functional_call, stack_module_state

from optim import BatchedAdam
from envs import Acrobot, Breakout, CartPole, CliffWalking, FrozenLake, MountainCar, Pendulum, NAgentWrapper
from utils import set_seed, get_version_num
from buffer import ReplayBuffer


@dataclass
class Args:
    test: bool = False
    """whether to run the test experiments"""
    env_type: str = "cartpole"
    """environment type"""
    n_seeds: int = 10
    """number of copies for each config"""
    n_agents: int = 1
    """number of agents to run in parallel"""
    n_envs: int = 1
    """number of envs per agent"""
    n_eval_envs: int = 40
    """number of eval envs per agent"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    train_freq: int = 10
    """the (env/training) step ratio"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    eval_freq: int = 1000
    """the frequency of evaluating the eval_envs"""
    learning_starts: int = 10000
    """timestep to start learning"""
    buffer_size: int = 10000
    batch_size: int = 128
    lr: float = 2.5e-4
    gamma: float = .99
    target_net_freq: float = 500
    tau: float = 1.0
    end_e: float = .05
    explore_frac: float = .5


@dataclass
class Hyperparams:
    lr: list = None
    gamma: list = None
    tau: list = None
    target_net_freq: list = None
    end_e: list = None
    explore_frac: list = None


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        n_hidden = 16
        if isinstance(obs_dim, int):
            self.network = nn.Sequential(
                nn.Linear(obs_dim, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, n_actions),
            )

        else:
            self.network = nn.Sequential(
                nn.Conv2d(4, n_hidden, 4),  # 10x10 -> 7x7
                nn.ReLU(),
                nn.Conv2d(n_hidden, n_hidden, 3, 2),  # 7x7 -> 3x3
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(n_hidden * 3 * 3, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, n_actions),
            )

    def forward(self, x):
        return self.network(x)


@dataclass
class Agent:
    n: int
    n_actions: int
    q_net_params: dict[torch.Tensor]
    q_net_buffer: dict[torch.Tensor]
    target_net_params: dict[torch.Tensor]
    target_net_buffer: dict[torch.Tensor]
    qnet_forward: callable


def make_agents(n_agents, obs_dim, n_actions, device):
    models = [QNetwork(obs_dim, n_actions).to(device) for _ in range(n_agents)]
    base_model = models[0]
    q_net_params, q_net_buffer = stack_module_state(models)
    def qnet_forward(params, buffers, data):
        return functional_call(base_model, (params, buffers), data)
    vec_qnet_forward = vmap(qnet_forward, in_dims=(0, 0, 0), out_dims=0)

    target_net_params = copy.deepcopy(q_net_params)
    target_net_buffer = copy.deepcopy(q_net_buffer)

    return Agent(n_agents, n_actions, q_net_params, q_net_buffer, target_net_params, target_net_buffer, vec_qnet_forward)


@torch.no_grad()
def evaluate(env, agent: Agent, device) -> tuple[np.ndarray, np.ndarray]:
    dones = np.zeros((agent.n, env.n_envs), dtype=np.bool_)
    eps_returns = np.zeros((agent.n, env.n_envs), )
    eps_lengths = np.zeros((agent.n, env.n_envs), )

    obs, _ = env.reset()
    while not np.all(dones):
        q_values = agent.qnet_forward(agent.q_net_params, agent.q_net_buffer, torch.from_numpy(obs).to(device))
        actions = torch.argmax(q_values, dim=-1)
        (obs, _), rewards, terminations, truncations, _ = env.step(actions.flatten())

        one_minus_dones = 1 - dones.astype(np.float32)
        eps_returns += one_minus_dones * rewards
        eps_lengths += one_minus_dones
        dones = np.logical_or(dones, np.logical_or(terminations, truncations))

    return eps_returns, eps_lengths


def reshape_hparam(param, n_grids, dtype: torch.dtype, device: torch.device):
    param = np.tile(param, n_grids)
    return torch.from_numpy(param).to(device, dtype)


def batched_train(
    agent: Agent,
    optimizer: BatchedAdam,
    env,
    eval_env,
    rb: ReplayBuffer,
    hparams: Hyperparams,
    args: Args,
):
    set_seed(args.seed, args.torch_deterministic)
    start_time = time.time()
    last_log_time = start_time

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    n_agents = agent.n

    eval_returns = np.zeros((n_agents, args.total_timesteps//args.eval_freq), dtype=np.float32)
    eval_lengths = np.zeros((n_agents, args.total_timesteps//args.eval_freq), dtype=np.float32)
    log_loss = np.zeros((n_agents, args.total_timesteps//args.eval_freq), dtype=np.float32)
    log_qvals = np.zeros((n_agents, args.total_timesteps//args.eval_freq), dtype=np.float32)
    eval_pos = 0

    gamma = hparams.gamma.unsqueeze(-1)

    max_loss = torch.full((n_agents,), 1e7, dtype=torch.float32, device=device)
    td_target = torch.zeros((n_agents, args.batch_size), dtype=torch.float32, device=device)
    loss_per_agent = torch.zeros((n_agents,), dtype=torch.float32, device=device)

    start_e = torch.full_like(hparams.end_e, 1.)
    epsilon_delta = (hparams.end_e - start_e) / (hparams.explore_frac * args.total_timesteps)

    if args.train_freq <= 0:
        raise ValueError(f"args.training_freq cannot be <= 0")
    elif args.train_freq > 0:
        train_freq = args.train_freq * args.n_envs

    learning_starts = args.learning_starts
    obs, _ = env.reset()
    for global_step in range(1, args.total_timesteps+1, args.n_envs):
        with torch.no_grad():
            epsilon = torch.clip(epsilon_delta * global_step + start_e, hparams.end_e, start_e)
            rand_action_mask = (torch.rand(n_agents, device=device) < epsilon).unsqueeze(-1)
            rand_actions = torch.randint(0, agent.n_actions, size=(n_agents, args.n_envs), device=device)
            q_values = agent.qnet_forward(agent.q_net_params, agent.q_net_buffer, torch.from_numpy(obs).to(device))

            policy_actions = torch.argmax(q_values, dim=-1)
            actions = torch.where(rand_action_mask, rand_actions, policy_actions)

        # Step
        (next_obs, true_next_obs), rewards, terminations, truncations, infos = env.step(actions.flatten())

        rb.add(obs, true_next_obs, actions.cpu().numpy(), rewards, terminations, infos)
        obs = next_obs

        if global_step % 1000 == 0:
            t = time.time()
            print(f"[{global_step} / {args.total_timesteps}] Elapsed Time: (full)={(t - start_time):.2f}, (partial)={(t - last_log_time):.2f}, (partial/agent)={((t - last_log_time)/n_agents):.2f}")
            last_log_time = t

        # Training
        if global_step > learning_starts:
            if global_step % train_freq == 0:
                data = rb.sample(args.batch_size)
                s_ = torch.from_numpy(data.next_observations).to(device)
                d_ = torch.from_numpy(data.dones).to(device)
                r_ = torch.from_numpy(data.rewards).to(device)
                with torch.no_grad():
                    q_values = agent.qnet_forward(agent.target_net_params, agent.target_net_buffer, s_)
                    target_max, _ = torch.max(q_values, dim=-1)
                    td_target = r_ + gamma * target_max * (1 - d_)
                s_ = torch.from_numpy(data.observations).to(device)
                a_ = torch.from_numpy(data.actions).to(device)
                old_val = agent.qnet_forward(agent.q_net_params, agent.q_net_buffer, s_).gather(-1, a_.unsqueeze(-1)).squeeze(-1)
                loss_per_agent = F.mse_loss(td_target, old_val, reduction="none").mean(-1)

                # Avoid NaNs polluting other agent's gradients
                loss_per_agent = torch.clamp_max(loss_per_agent, max_loss)

                # Each agent has independent parameters, and loss_i depends only on agent i.
                #  Summing preserves the same gradients each agent would get if trained alone.
                #  Averaging would incorrectly scale gradients by 1 / n_agents.
                loss = loss_per_agent.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update target network
                with torch.no_grad():
                    update_mask = (global_step % hparams.target_net_freq) == 0
                    if torch.any(update_mask):
                        masked_tau = update_mask * hparams.tau

                        for target_net_param, q_net_param in zip(agent.target_net_params.values(), agent.q_net_params.values()):
                            tau_shape = [n_agents] + [1 for _ in range(target_net_param.ndim - 1)]
                            masked_tau_shaped = masked_tau.view(tau_shape)
                            target_net_param.data.copy_(
                                masked_tau_shaped * q_net_param.data + (1.0 - masked_tau_shaped) * target_net_param.data
                            )

        # eval
        if global_step % args.eval_freq == 0:
            eval_return, eval_length = evaluate(eval_env, agent, device)

            eval_returns[:, eval_pos] = eval_return.mean(1)
            eval_lengths[:, eval_pos] = eval_length.mean(1)
            log_loss[:, eval_pos] = loss_per_agent.detach().cpu().numpy()
            log_qvals[:, eval_pos] = td_target.mean(1).cpu().numpy()
            eval_pos += 1

    return eval_returns, eval_lengths, log_loss, log_qvals


def grid_search(args: Args, device, n_copies=1):
    lr_starts = [1.0, .5]
    lr_powers = [1e-2, 1e-3, 1e-4]
    lrs = []
    for lr_p in lr_powers:
        for lr_s in lr_starts:
            lrs.append(lr_s * lr_p)
    lrs.append(1e-5)

    gammas = [.8, .9, .95, .975, .99, .995, .999]
    target_net_freqs = [20, 100, 400, 700, 1000]

    end_es = [.01, .05, .2, .5]
    lr = []
    gamma = []
    target_net_freq = []
    end_e = []
    for (l, g, t, ee) in itertools.product(lrs, gammas, target_net_freqs, end_es):
        lr.append(l)
        gamma.append(g)
        target_net_freq.append(t)
        end_e.append(ee)

    n_agents = len(lr)

    tau = [1.0] * n_agents
    explore_frac = [args.explore_frac] * n_agents

    lr = reshape_hparam(lr, n_copies, torch.float32, device)
    target_net_freq = reshape_hparam(target_net_freq, n_copies, torch.int32, device)
    gamma = reshape_hparam(gamma, n_copies, torch.float32, device)
    tau = reshape_hparam(tau, n_copies, torch.float32, device)
    end_e = reshape_hparam(end_e, n_copies, torch.float32, device)
    explore_frac = reshape_hparam(explore_frac, n_copies, torch.float32, device)
    return Hyperparams(lr=lr, gamma=gamma, tau=tau,
                       target_net_freq=target_net_freq, end_e=end_e,
                       explore_frac=explore_frac)


def make_envs(env_type, n_agents, n_envs, n_eval_envs, seed):
    if env_type == "acrobot":
        Env = Acrobot
    elif env_type == "cartpole":
        Env = CartPole
    elif env_type == "cliffwalking":
        Env = CliffWalking
    elif env_type == "frozenlake":
        Env = FrozenLake
    elif env_type == "mountaincar":
        Env = MountainCar
    elif env_type == "breakout":
        Env = Breakout
    elif env_type == "pendulum":
        Env = Pendulum
    else:
        raise ValueError(f"Unrecognized env_type={env_type}")

    env = Env(n_agents * n_envs, seed=seed + 42)
    eval_env = Env(n_agents * n_eval_envs, seed=seed + 42)

    env = NAgentWrapper(env, n_agents, n_envs)
    eval_env = NAgentWrapper(eval_env, n_agents, n_eval_envs)
    return env, eval_env


def main_gridsearch():
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    hparams = grid_search(args, device, args.n_seeds)

    n_agents_copied = len(hparams.lr)
    args.n_agents = n_agents_copied // args.n_seeds
    n_agents = args.n_agents
    print(f"Gridsearch Training {args.n_agents} Agents")

    env, eval_env = make_envs(args.env_type, n_agents_copied, args.n_envs, args.n_eval_envs, args.seed)

    agent = make_agents(n_agents_copied, env.obs_dim, env.n_actions, device)
    optimizer = BatchedAdam(n_agents_copied, [p for p in agent.q_net_params.values()], lr=hparams.lr)

    rb = ReplayBuffer(n_agents_copied, args.buffer_size, env.obs_dim, device, args.n_envs)

    data_dir = f"data/raw/{args.env_type}"
    os.makedirs(data_dir, exist_ok=True)

    returns, lengths, losses, qvals = batched_train(agent, optimizer, env, eval_env, rb, hparams, args)

    data = dict()
    n_agents = len(returns)
    data["agent_id"] = np.tile(list(range(n_agents // args.n_seeds)), args.n_seeds)
    data["lr"] = hparams.lr.cpu().numpy()
    data["gamma"] = hparams.gamma.cpu().numpy()
    data["tau"] = hparams.tau.cpu().numpy()
    data["target_net_freq"] = hparams.target_net_freq.cpu().numpy()
    data["end_e"] = hparams.end_e.cpu().numpy()
    data["explore_frac"] = hparams.explore_frac.cpu().numpy()
    for i in range(returns.shape[1]):
        data[f"ret_{i}"] = returns[:, i]
        data[f"len_{i}"] = lengths[:, i]
        data[f"loss_{i}"] = losses[:, i]
        data[f"qval_{i}"] = qvals[:, i]

    env_name = args.env_type
    batch_size_name = f"bch={args.batch_size}"
    seed_name = f"seed={args.seed}"
    buf_name = f"buf={args.buffer_size}"
    freq_name = f"freq={args.train_freq}"
    filename = f"{env_name}-{buf_name}-{batch_size_name}-{freq_name}-{seed_name}"
    file_version = get_version_num(f"{data_dir}/{filename}")
    filename = f"{filename}_v{file_version}.feather"

    pd.DataFrame(data).astype(np.float32).to_feather(f"{data_dir}/{filename}")


def main_test():
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    data_dir = f"data/test/{args.env_type}"
    os.makedirs(data_dir, exist_ok=True)

    gammas = [.8, .95, .995]
    lrs = [5e-5, 5e-4, 5e-3]

    lr = []
    gamma = []
    for (l, g) in itertools.product(lrs, gammas):
        lr.append(l)
        gamma.append(g)

    n_agents = len(lr)

    tau = [1.0] * n_agents
    explore_frac = [.1] * n_agents
    end_e = [.01] * n_agents
    target_net_freq = [700] * n_agents

    n_copies = args.n_seeds
    lr = reshape_hparam(lr, n_copies, torch.float32, device)
    target_net_freq = reshape_hparam(target_net_freq, n_copies, torch.int32, device)
    gamma = reshape_hparam(gamma, n_copies, torch.float32, device)
    tau = reshape_hparam(tau, n_copies, torch.float32, device)
    end_e = reshape_hparam(end_e, n_copies, torch.float32, device)
    explore_frac = reshape_hparam(explore_frac, n_copies, torch.float32, device)
    hparams = Hyperparams(lr=lr, gamma=gamma, tau=tau,
                       target_net_freq=target_net_freq, end_e=end_e,
                       explore_frac=explore_frac)

    args.n_agents = n_agents
    n_agents_copied = args.n_agents * args.n_seeds

    env, eval_env = make_envs(args.env_type, n_agents_copied, args.n_envs, args.n_eval_envs, args.seed)

    rb = ReplayBuffer(n_agents_copied, args.buffer_size, env.obs_dim, device, args.n_envs)

    agent = make_agents(n_agents_copied, env.obs_dim, env.n_actions, device)
    optimizer = BatchedAdam(agent.n, [p for p in agent.q_net_params.values()], lr=hparams.lr)

    returns, lengths, losses, qvals = batched_train(agent, optimizer, env, eval_env, rb, hparams, args)

    data = dict()
    n_agents = len(returns)
    data["agent_id"] = np.tile(list(range(n_agents // args.n_seeds)), args.n_seeds)
    data["lr"] = hparams.lr.cpu().numpy()
    data["gamma"] = hparams.gamma.cpu().numpy()
    data["tau"] = hparams.tau.cpu().numpy()
    data["target_net_freq"] = hparams.target_net_freq.cpu().numpy()
    data["end_e"] = hparams.end_e.cpu().numpy()
    data["explore_frac"] = hparams.explore_frac.cpu().numpy()
    for i in range(returns.shape[1]):
        data[f"ret_{i}"] = returns[:, i]
        data[f"len_{i}"] = lengths[:, i]
        data[f"loss_{i}"] = losses[:, i]
        data[f"qval_{i}"] = qvals[:, i]

    env_name = args.env_type
    batch_size_name = f"bch={args.batch_size}"
    seed_name = f"seed={args.seed}"
    buf_name = f"buf={args.buffer_size}"
    freq_name = f"freq={args.train_freq}"
    filename = f"TEST-{env_name}-{buf_name}-{batch_size_name}-{freq_name}-{seed_name}"
    file_version = get_version_num(f"{data_dir}/{filename}")
    filename = f"{filename}_v{file_version}.feather"

    pd.DataFrame(data).astype(np.float32).to_feather(f"{data_dir}/{filename}")


def main():
    args = tyro.cli(Args)
    if args.test:
        main_test()
    else:
        main_gridsearch()


if __name__ == "__main__":
    main()