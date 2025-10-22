"""Implementation of a Soft Actor-Critic agent with a Lagrangian safety constraint.

The agent augments the classic SAC formulation with an additional cost critic
that estimates the discounted cumulative safety cost. A dual variable (lambda)
keeps the expected cost below a user specified threshold. The environment is
expected to expose a "cost" entry in the info dictionary of ``step`` which is
interpreted as the instantaneous safety cost (here the collision risk).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MLP(nn.Module):
    """Simple helper module used by the actor and critic networks."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int, activate_final: bool = False):
        super().__init__()
        layers = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.ReLU())
            last_dim = dim
        layers.append(nn.Linear(last_dim, output_dim))
        if activate_final:
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Gaussian policy with Tanh squashing used for the SAC actor."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...],
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        log_std_bounds: Tuple[float, float] = (-20.0, 2.0),
    ) -> None:
        super().__init__()
        self.net = MLP(obs_dim, hidden_dims, 2 * action_dim)
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.log_std_bounds = log_std_bounds

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_log_std = self.net(obs)
        mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
        log_std = torch.tanh(log_std)
        min_log_std, max_log_std = self.log_std_bounds
        log_std = min_log_std + 0.5 * (max_log_std - min_log_std) * (log_std + 1)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action


class QNetwork(nn.Module):
    """Critic network approximating Q(s, a)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        self.net = MLP(obs_dim + action_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class ReplayBuffer:
    """Simple replay buffer for experience replay."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.cost_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, cost: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.cost_buf[self.ptr] = cost
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=torch.as_tensor(self.obs_buf[idxs], device=self.device),
            actions=torch.as_tensor(self.act_buf[idxs], device=self.device),
            rewards=torch.as_tensor(self.rew_buf[idxs], device=self.device).unsqueeze(-1),
            costs=torch.as_tensor(self.cost_buf[idxs], device=self.device).unsqueeze(-1),
            next_obs=torch.as_tensor(self.next_obs_buf[idxs], device=self.device),
            dones=torch.as_tensor(self.done_buf[idxs], device=self.device).unsqueeze(-1),
        )
        return batch


@dataclass
class SafeSACConfig:
    obs_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    actor_hidden_dims: Tuple[int, ...] = (256, 256)
    critic_hidden_dims: Tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    cost_critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    lambda_lr: float = 1e-4
    target_entropy: Optional[float] = None
    init_temperature: float = 0.2
    init_lambda: float = 0.0
    buffer_size: int = 1_000_000
    batch_size: int = 256
    start_steps: int = 10000
    updates_per_step: int = 1
    device: torch.device = torch.device("cpu")
    cost_limit: float = 1.0
    log_std_bounds: Tuple[float, float] = (-20.0, 2.0)


class SafeSACAgent:
    """Soft Actor-Critic agent augmented with a safety cost constraint."""

    def __init__(self, config: SafeSACConfig) -> None:
        self.cfg = config
        self.device = config.device
        action_scale = torch.from_numpy((config.action_high - config.action_low) / 2.0).to(self.device)
        action_bias = torch.from_numpy((config.action_high + config.action_low) / 2.0).to(self.device)

        self.actor = GaussianPolicy(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            hidden_dims=config.actor_hidden_dims,
            action_scale=action_scale,
            action_bias=action_bias,
            log_std_bounds=config.log_std_bounds,
        ).to(self.device)

        self.critic1 = QNetwork(config.obs_dim, config.action_dim, config.critic_hidden_dims).to(self.device)
        self.critic2 = QNetwork(config.obs_dim, config.action_dim, config.critic_hidden_dims).to(self.device)
        self.critic1_target = QNetwork(config.obs_dim, config.action_dim, config.critic_hidden_dims).to(self.device)
        self.critic2_target = QNetwork(config.obs_dim, config.action_dim, config.critic_hidden_dims).to(self.device)

        self.cost_critic1 = QNetwork(config.obs_dim, config.action_dim, config.critic_hidden_dims).to(self.device)
        self.cost_critic2 = QNetwork(config.obs_dim, config.action_dim, config.critic_hidden_dims).to(self.device)
        self.cost_critic1_target = QNetwork(config.obs_dim, config.action_dim, config.critic_hidden_dims).to(self.device)
        self.cost_critic2_target = QNetwork(config.obs_dim, config.action_dim, config.critic_hidden_dims).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.cost_critic1_target.load_state_dict(self.cost_critic1.state_dict())
        self.cost_critic2_target.load_state_dict(self.cost_critic2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=config.critic_lr)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=config.critic_lr)
        self.cost_critic1_optim = torch.optim.Adam(self.cost_critic1.parameters(), lr=config.cost_critic_lr)
        self.cost_critic2_optim = torch.optim.Adam(self.cost_critic2.parameters(), lr=config.cost_critic_lr)

        self.log_alpha = torch.tensor(np.log(config.init_temperature), device=self.device, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=config.temperature_lr)
        self.target_entropy = config.target_entropy if config.target_entropy is not None else -float(config.action_dim)

        init_lambda = max(config.init_lambda, 0.0)
        self.lambda_param = torch.tensor(init_lambda, device=self.device, requires_grad=True)
        self.lambda_optim = torch.optim.Adam([self.lambda_param], lr=config.lambda_lr)

        self.replay_buffer = ReplayBuffer(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            capacity=config.buffer_size,
            device=self.device,
        )

        self.total_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def lambda_value(self) -> float:
        return float(torch.clamp(self.lambda_param, min=0.0).item())

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(obs_tensor)
            else:
                action, _ = self.actor.sample(obs_tensor)
        return action.cpu().numpy().squeeze(0)

    def store_transition(self, obs: np.ndarray, action: np.ndarray, reward: float, cost: float, next_obs: np.ndarray, done: bool) -> None:
        self.replay_buffer.add(obs, action, reward, cost, next_obs, done)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.cfg.tau
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update_parameters(self) -> Optional[Dict[str, float]]:
        if self.replay_buffer.size < self.cfg.batch_size:
            return None

        batch = self.replay_buffer.sample(self.cfg.batch_size)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        costs = batch["costs"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            next_q1_target = self.critic1_target(next_obs, next_action)
            next_q2_target = self.critic2_target(next_obs, next_action)
            next_q_target = torch.min(next_q1_target, next_q2_target) - self.alpha * next_log_prob
            q_target = rewards + (1.0 - dones) * self.cfg.gamma * next_q_target

        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        critic1_loss = F.mse_loss(current_q1, q_target)
        critic2_loss = F.mse_loss(current_q2, q_target)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        with torch.no_grad():
            next_cost_q1 = self.cost_critic1_target(next_obs, next_action)
            next_cost_q2 = self.cost_critic2_target(next_obs, next_action)
            next_cost_q = torch.min(next_cost_q1, next_cost_q2)
            cost_target = costs + (1.0 - dones) * self.cfg.gamma * next_cost_q

        current_cost_q1 = self.cost_critic1(obs, actions)
        current_cost_q2 = self.cost_critic2(obs, actions)
        cost_critic1_loss = F.mse_loss(current_cost_q1, cost_target)
        cost_critic2_loss = F.mse_loss(current_cost_q2, cost_target)

        self.cost_critic1_optim.zero_grad()
        cost_critic1_loss.backward()
        self.cost_critic1_optim.step()

        self.cost_critic2_optim.zero_grad()
        cost_critic2_loss.backward()
        self.cost_critic2_optim.step()

        pi_action, log_pi = self.actor.sample(obs)
        q1_pi = self.critic1(obs, pi_action)
        q2_pi = self.critic2(obs, pi_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        cost_q1_pi = self.cost_critic1(obs, pi_action)
        cost_q2_pi = self.cost_critic2(obs, pi_action)
        cost_pi = torch.max(cost_q1_pi, cost_q2_pi)

        lambda_clamped = torch.clamp(self.lambda_param, min=0.0)
        actor_loss = (self.alpha.detach() * log_pi - min_q_pi + lambda_clamped * cost_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        with torch.no_grad():
            cost_constraint = cost_pi.mean()
        lambda_loss = -(lambda_clamped * (self.cfg.cost_limit - cost_constraint)).mean()
        self.lambda_optim.zero_grad()
        lambda_loss.backward()
        self.lambda_optim.step()
        self.lambda_param.data.clamp_(min=0.0)

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        self._soft_update(self.cost_critic1, self.cost_critic1_target)
        self._soft_update(self.cost_critic2, self.cost_critic2_target)

        self.total_updates += 1

        metrics = {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "cost_critic1_loss": cost_critic1_loss.item(),
            "cost_critic2_loss": cost_critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "lambda_loss": lambda_loss.item(),
            "alpha": self.alpha.item(),
            "lambda": self.lambda_value,
            "cost_constraint": cost_constraint.item(),
        }
        return metrics
