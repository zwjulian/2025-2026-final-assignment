"""Agents used in the assignment.

The main agent is an Advantage Actor-Critic (A2C) implementation for continuous
MuJoCo control. It includes several practical stabilizers:

* tanh-squashed Gaussian actions instead of post-hoc clipping;
* observation normalization with saved/restored running statistics;
* n-step bootstrapping with Generalized Advantage Estimation (GAE);
* advantage normalization, entropy regularization, Huber value loss, and gradient
  clipping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

try:  
    from network import ActorCriticNetwork
except ImportError: 
    from .network import ActorCriticNetwork


LOG_STD_EPS = 1e-6


@dataclass
class RunningMeanStd:
    """Running mean and variance for observation normalization."""

    shape: tuple[int, ...]
    epsilon: float = 1e-4

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(self.epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.shape):
            x = x.reshape(1, *self.shape)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = total_count

    def normalize(self, x: np.ndarray | torch.Tensor, clip: float = 10.0):
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=torch.float32, device=x.device)
            std = torch.as_tensor(np.sqrt(self.var), dtype=torch.float32, device=x.device)
            return torch.clamp((x - mean) / (std + 1e-8), -clip, clip)

        x = np.asarray(x, dtype=np.float32)
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -clip, clip).astype(
            np.float32
        )

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class Agent(ABC):
    def __init__(self, env) -> None:
        self.env = env

    @abstractmethod
    def choose_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def learn(self, last_state: np.ndarray, last_done: bool) -> dict[str, float]:
        pass


class A2CAgent(Agent):
    """Advantage Actor-Critic agent for bounded continuous action spaces."""

    def __init__(
        self,
        env,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.005,
        value_loss_coef: float = 0.5,
        grad_clip: float = 0.5,
        hidden_size: int = 256,
        initial_log_std: float = -0.5,
        normalize_obs: bool = True,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.grad_clip = grad_clip
        self.normalize_obs = normalize_obs
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        obs_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.shape[0])
        self.obs_rms = RunningMeanStd((obs_dim,))

        self.action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self.network = ActorCriticNetwork(
            obs_dim,
            action_dim,
            hidden_size=hidden_size,
            initial_log_std=initial_log_std,
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)

        self.states: list[np.ndarray] = []
        self.raw_actions: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def _prepare_observation(self, observation: np.ndarray, update_stats: bool) -> torch.Tensor:
        observation = np.asarray(observation, dtype=np.float32)
        if self.normalize_obs:
            if update_stats:
                self.obs_rms.update(observation)
            observation = self.obs_rms.normalize(observation)
        return torch.as_tensor(observation, dtype=torch.float32, device=self.device)

    def _scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_action) * self.action_scale + self.action_bias

    def _log_prob_from_raw_action(self, dist, raw_action: torch.Tensor) -> torch.Tensor:
        """Log probability of a tanh-squashed action with change-of-variables."""
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        squash_correction = torch.log(1.0 - torch.tanh(raw_action).pow(2) + LOG_STD_EPS).sum(dim=-1)
        return log_prob - squash_correction

    def choose_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action.

        During training, the agent samples from the policy and stores the raw
        Gaussian action for the later policy-gradient update. During evaluation,
        it uses the policy mean.
        """
        obs_tensor = self._prepare_observation(observation, update_stats=not deterministic)

        self.network.eval()
        with torch.no_grad():
            dist, _ = self.network(obs_tensor)
            raw_action = dist.mean if deterministic else dist.rsample()
            scaled_action = self._scale_action(raw_action)

        if not deterministic:
            self.states.append(np.asarray(observation, dtype=np.float32))
            self.raw_actions.append(raw_action.detach().cpu())

        return scaled_action.squeeze(0).cpu().numpy()

    def store_outcome(self, reward: float, done: bool) -> None:
        """Store reward and termination flag for the latest transition."""
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def learn(self, last_state: np.ndarray, last_done: bool) -> dict[str, float]:
        """Update actor and critic from the currently stored rollout."""
        if len(self.rewards) == 0:
            return {}

        self.network.train()

        states_np = np.asarray(self.states, dtype=np.float32)
        if self.normalize_obs:
            states_np = self.obs_rms.normalize(states_np)
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        raw_actions = torch.cat(self.raw_actions, dim=0).to(self.device)
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.dones, dtype=torch.float32, device=self.device)

        dist, values = self.network(states)
        values = values.squeeze(-1)
        log_probs = self._log_prob_from_raw_action(dist, raw_actions)
        entropy = dist.entropy().sum(dim=-1).mean()

        with torch.no_grad():
            last_obs = self._prepare_observation(last_state, update_stats=False)
            if last_done:
                next_value = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            else:
                _, next_value = self.network(last_obs)
                next_value = next_value.squeeze()

            advantages = torch.zeros_like(rewards)
            gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            for t in reversed(range(len(rewards))):
                bootstrap_value = next_value if t == len(rewards) - 1 else values[t + 1]
                nonterminal = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * bootstrap_value * nonterminal - values[t]
                gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
                advantages[t] = gae
            returns = advantages + values

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.smooth_l1_loss(values, returns.detach())
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.optimizer.step()

        self.clear_memory()
        return {
            "total_loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    def clear_memory(self) -> None:
        self.states.clear()
        self.raw_actions.clear()
        self.rewards.clear()
        self.dones.clear()

    def save(self, path: str) -> None:
        checkpoint = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "normalize_obs": self.normalize_obs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "entropy_coef": self.entropy_coef,
            "value_loss_coef": self.value_loss_coef,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False
        )
        if isinstance(checkpoint, dict) and "network" in checkpoint:
            self.network.load_state_dict(checkpoint["network"])

            if "obs_rms" in checkpoint:
                self.obs_rms.load_state_dict(checkpoint["obs_rms"])

            self.normalize_obs = bool(
                checkpoint.get("normalize_obs", self.normalize_obs)
            )
        elif isinstance(checkpoint, dict) and "network_state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["network_state_dict"])

            if "obs_rms" in checkpoint:
                self.obs_rms.load_state_dict(checkpoint["obs_rms"])

            self.normalize_obs = bool(
                checkpoint.get("normalize_obs", self.normalize_obs)
            )
        else:
            self.network.load_state_dict(checkpoint)

        self.network.eval()