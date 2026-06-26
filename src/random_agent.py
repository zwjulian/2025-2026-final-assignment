import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class RandomNetwork(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int):
        super().__init__()
        # Keep one trainable parameter so state_dict isn't empty
        self.log_std = nn.Parameter(torch.zeros(1, n_outputs))

    def forward(self, state: np.ndarray):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]
        # Mean of the policy is zero; RandomAgent will sample from env.action_space
        mu = torch.zeros(batch_size, self.log_std.shape[1])
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        # Random policy has no value estimate; return zeros
        value = torch.zeros(batch_size, 1)
        return dist, value


class RandomAgent:
    """A minimal agent that selects random actions from the environment's
    action space. Implements the same interface used by A2CAgent so it can be
    integrated into training/evaluation pipelines.
    """

    def __init__(self, env):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.network = RandomNetwork(obs_dim, action_dim)

        # Keep buffers so the API mirrors A2CAgent
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def choose_action(self, observation: np.ndarray, deterministic: bool = False):
        # Random action drawn from the environment's action space
        return self.env.action_space.sample()

    def learn(self, *args, **kwargs):
        # Random agent has no learning
        return {}
