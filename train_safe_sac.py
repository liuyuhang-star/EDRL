"""Train a safety constrained SAC agent that treats collision risk as cost."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from env.make_env import make_srl_purestate
from safe_rl import SafeSACAgent, SafeSACConfig


def _load_agent_config(cfg: Dict, device: torch.device, cost_limit: float) -> SafeSACConfig:
    agent_cfg = cfg.get("agent", {}).get("SafeSAC", {})
    obs_dim = cfg.get("state_shape")  # placeholder, overwritten later
    action_dim = cfg.get("action_dim")  # placeholder, overwritten later

    actor_hidden_dims = tuple(agent_cfg.get("actor_hidden_dims", [256, 256]))
    critic_hidden_dims = tuple(agent_cfg.get("critic_hidden_dims", [256, 256]))

    gamma = agent_cfg.get("gamma", cfg.get("gamma", 0.99))
    tau = agent_cfg.get("tau", cfg.get("tau", 0.005))
    actor_lr = agent_cfg.get("actor_lr", cfg.get("lr_actor", 3e-4))
    critic_lr = agent_cfg.get("critic_lr", cfg.get("lr_critic", 3e-4))
    cost_critic_lr = agent_cfg.get("cost_critic_lr", critic_lr)
    temperature_lr = agent_cfg.get("temperature_lr", agent_cfg.get("lr_temp", cfg.get("lr_temp", 3e-4)))
    lambda_lr = agent_cfg.get("lambda_lr", 1e-4)
    init_temp = agent_cfg.get("init_temp", agent_cfg.get("init_temperature", 0.2))
    init_lambda = agent_cfg.get("init_lambda", 0.0)
    buffer_size = agent_cfg.get("buffer_size", cfg.get("buffer_length", 100000))
    batch_size = agent_cfg.get("batch_size", cfg.get("batch_size", 256))
    start_steps = agent_cfg.get("start_steps", cfg.get("act_start_step", 5000))
    updates_per_step = agent_cfg.get("updates_per_step", max(1, cfg.get("upd_every", 1)))
    log_std_bounds = tuple(agent_cfg.get("log_std_bounds", (-20.0, 2.0)))
    target_entropy = agent_cfg.get("target_entropy")

    # placeholders (will be replaced after environment construction)
    action_low = np.array([-1.0])
    action_high = np.array([1.0])

    return SafeSACConfig(
        obs_dim=obs_dim if isinstance(obs_dim, int) else 0,
        action_dim=action_dim if isinstance(action_dim, int) else 0,
        action_low=action_low,
        action_high=action_high,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        gamma=gamma,
        tau=tau,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        cost_critic_lr=cost_critic_lr,
        temperature_lr=temperature_lr,
        lambda_lr=lambda_lr,
        target_entropy=target_entropy,
        init_temperature=init_temp,
        init_lambda=init_lambda,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_steps=start_steps,
        updates_per_step=updates_per_step,
        device=device,
        cost_limit=cost_limit,
        log_std_bounds=log_std_bounds,
    )


def _evaluate_policy(env, agent: SafeSACAgent, episodes: int) -> Dict[str, float]:
    rewards: List[float] = []
    costs: List[float] = []

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_cost = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)
            ep_cost += float(info.get("cost", 0.0))
        rewards.append(ep_reward)
        costs.append(ep_cost)

    return {
        "reward": float(np.mean(rewards)),
        "cost": float(np.mean(costs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("Metadrive.yaml"), help="Path to the YAML config file.")
    parser.add_argument("--cost-limit", type=float, default=1.0, help="Upper bound for the expected discounted cost.")
    parser.add_argument("--cost-mode", type=str, default="standard", help="Risk aggregation mode used by the environment.")
    parser.add_argument("--total-steps", type=int, default=500_000, help="Number of environment interaction steps.")
    parser.add_argument("--eval-interval", type=int, default=50_000, help="Frequency of evaluation runs in environment steps.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes used during evaluation.")
    parser.add_argument("--log-interval", type=int, default=5_000, help="Frequency of logging training statistics.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device identifier (e.g. 'cpu' or 'cuda').")
    args = parser.parse_args()

    with args.config.open("r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    env = make_srl_purestate(cost_fn_mode=args.cost_mode)
    eval_env = make_srl_purestate(cost_fn_mode=args.cost_mode)

    env.seed(args.seed)
    eval_env.seed(args.seed + 1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent_cfg = _load_agent_config(cfg, device=device, cost_limit=args.cost_limit)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent_cfg.obs_dim = obs_dim
    agent_cfg.action_dim = action_dim
    agent_cfg.action_low = env.action_space.low.astype(np.float32)
    agent_cfg.action_high = env.action_space.high.astype(np.float32)
    agent = SafeSACAgent(agent_cfg)

    obs, info = env.reset()
    episode_reward = 0.0
    episode_cost = 0.0
    episode_length = 0
    episode_rewards: List[float] = []
    episode_costs: List[float] = []
    metrics_window: List[Dict[str, float]] = []

    start_time = time.time()

    for step in range(1, args.total_steps + 1):
        if step <= agent.cfg.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, done, info = env.step(action)
        cost = float(info.get("cost", 0.0))
        agent.store_transition(obs, action, float(reward), cost, next_obs, done)

        obs = next_obs
        episode_reward += float(reward)
        episode_cost += cost
        episode_length += 1

        if step > agent.cfg.start_steps:
            for _ in range(agent.cfg.updates_per_step):
                metrics = agent.update_parameters()
                if metrics is not None:
                    metrics_window.append(metrics)

        if done:
            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)
            obs, info = env.reset()
            episode_reward = 0.0
            episode_cost = 0.0
            episode_length = 0

        if args.log_interval > 0 and step % args.log_interval == 0:
            mean_ret = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
            mean_cost = float(np.mean(episode_costs[-10:])) if episode_costs else 0.0
            if metrics_window:
                avg_metrics = {k: float(np.mean([m[k] for m in metrics_window])) for k in metrics_window[0]}
                metrics_window.clear()
            else:
                avg_metrics = {}
            elapsed = (time.time() - start_time) / 3600.0
            print(
                f"[Step {step}] avg_return={mean_ret:.2f} avg_cost={mean_cost:.3f} alpha={avg_metrics.get('alpha', float('nan')):.4f} "
                f"lambda={avg_metrics.get('lambda', float('nan')):.4f} updates={agent.total_updates} runtime_h={elapsed:.2f}"
            )

        if args.eval_interval > 0 and step % args.eval_interval == 0:
            stats = _evaluate_policy(eval_env, agent, episodes=args.eval_episodes)
            print(
                f"[Eval @ step {step}] reward={stats['reward']:.2f} cost={stats['cost']:.3f} "
                f"lambda={agent.lambda_value:.4f}"
            )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
