def n_agent_reshape(array_like, n_a, n_e):
    return array_like.reshape(n_a, n_e, *array_like.shape[1:])


class NAgentWrapper:
    def __init__(self, env, n_agents, n_envs_per_agent, ):
        self.env = env
        self.n_agents = n_agents
        self.n_envs = n_envs_per_agent
    
    def reset(self):
        obs, info = self.env.reset()
        return n_agent_reshape(obs, self.n_agents, self.n_envs), info
    
    def step(self, action, log=False):
        # (obs, true_obs), reward, term, trun, info = self.env.step(action, log)
        (obs, true_obs), reward, term, trun, info = self.env.step(action)
        return (
            (n_agent_reshape(obs, self.n_agents, self.n_envs), n_agent_reshape(true_obs, self.n_agents, self.n_envs)),
            n_agent_reshape(reward, self.n_agents, self.n_envs), 
            n_agent_reshape(term, self.n_agents, self.n_envs), 
            n_agent_reshape(trun, self.n_agents, self.n_envs), 
            info,
        )
    
    @property
    def n_actions(self):
        return self.env.n_actions

    @property
    def obs_dim(self):
        return self.env.obs_dim

    @property
    def obs_dtype(self):
        return self.env.obs_dtype

