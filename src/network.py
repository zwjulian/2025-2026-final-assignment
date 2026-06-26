"""Neural network components for the A2C agent.

The policy is a tanh-squashed Gaussian. The network predicts the mean and
standard deviation of an unconstrained Gaussian, samples an unconstrained action
z, and the agent maps tanh(z) to the environment's bounded action range. This is
more stable for MuJoCo tasks than sampling an unbounded Gaussian and clipping the
action afterwards.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCriticNetwork(nn.Module):
    """Shared-body actor-critic network for continuous-control A2C.

    Parameters
    ----------
    n_inputs:
        Observation dimensionality.
    n_outputs:
        Action dimensionality.
    hidden_size:
        Width of the two hidden layers.
    initial_log_std:
        Initial log standard deviation of the Gaussian policy. A negative value
        reduces initial exploration noise, which is important for Pusher-v5.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_size: int = 256,
        initial_log_std: float = -0.5,
    ) -> None:
        super().__init__()

        self.shared_body = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_size, n_outputs)
        self.critic_head = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((1, n_outputs), initial_log_std))

        self._orthogonal_init()

    def _orthogonal_init(self) -> None:
        """Use orthogonal initialization, a common stabilizer for actor-critic."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                gain = nn.init.calculate_gain("tanh")
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

    def forward(self, state: np.ndarray | torch.Tensor) -> tuple[Normal, torch.Tensor]:
        """Return the Gaussian policy distribution and state value."""
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.float32)
        else:
            state = state.float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.shared_body(state)
        mu = self.actor_head(features)
        value = self.critic_head(features)

        log_std = torch.clamp(self.log_std, min=-5.0, max=1.0)
        std = log_std.exp().expand_as(mu)
        return Normal(mu, std), value
