"""Train and evaluate an A2C agent on Gymnasium MuJoCo environments.

Recommended commands:

    python src/main.py --env InvertedPendulum-v5 --mode train_and_test
    python src/main.py --env Pusher-v5 --mode train_and_test
    python src/main.py --env Pusher-v5 --mode analysis

The implementation is intentionally conservative: Pusher-v5 is much harder than
InvertedPendulum-v5, so it uses a smaller learning rate, lower exploration noise,
and many more training episodes.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

try: 
    from agent import A2CAgent
except ImportError:
    from .agent import A2CAgent


@dataclass(frozen=True)
class EnvConfig:
    episodes: int
    lr: float
    gamma: float
    rollout_steps: int
    gae_lambda: float
    entropy_coef: float
    value_loss_coef: float
    grad_clip: float
    hidden_size: int
    initial_log_std: float


ENV_CONFIGS: dict[str, EnvConfig] = {
    "InvertedPendulum-v5": EnvConfig(
        episodes=1500,
        lr=7e-4,
        gamma=0.99,
        rollout_steps=64,
        gae_lambda=0.95,
        entropy_coef=0.001,
        value_loss_coef=0.5,
        grad_clip=0.5,
        hidden_size=128,
        initial_log_std=-0.5,
    ),
    "Pusher-v5": EnvConfig(
        episodes=20000,
        lr=1e-4,
        gamma=0.98,
        rollout_steps=32,
        gae_lambda=0.95,
        entropy_coef=0.002,
        value_loss_coef=0.5,
        grad_clip=0.5,
        hidden_size=256,
        initial_log_std=-1.0,
    ),
}

RESULTS_DIR = "data"
PLOT_DIR = "data"
BASE_SEED = 42
BEST_MODEL_WINDOW = 100


def env_slug(env_name: str) -> str:
    return env_name.lower().split("-v")[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_name: str, seed: int | None = None):
    env = gym.make(env_name)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_agent(env, cfg: EnvConfig, lr: float | None = None) -> A2CAgent:
    return A2CAgent(
        env,
        learning_rate=cfg.lr if lr is None else lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        entropy_coef=cfg.entropy_coef,
        value_loss_coef=cfg.value_loss_coef,
        grad_clip=cfg.grad_clip,
        hidden_size=cfg.hidden_size,
        initial_log_std=cfg.initial_log_std,
        normalize_obs=True,
    )


def run_training_episode(env, agent: A2CAgent, cfg: EnvConfig, reset_seed: int | None) -> float:
    """Run one episode and update the agent every cfg.rollout_steps steps."""
    state, _ = env.reset(seed=reset_seed)
    terminated = False
    truncated = False
    episode_reward = 0.0
    rollout_len = 0

    while not (terminated or truncated):
        action = agent.choose_action(state, deterministic=False)
        next_state, reward, terminated, truncated, _ = env.step(action)

        agent.store_outcome(reward, done=terminated)

        state = next_state
        episode_reward += float(reward)
        rollout_len += 1

        should_update = rollout_len >= cfg.rollout_steps or terminated or truncated
        if should_update:
            agent.learn(last_state=state, last_done=terminated)
            rollout_len = 0

    return episode_reward


def train(
    env_name: str,
    total_episodes: int,
    cfg: EnvConfig,
    num_runs: int = 5,
) -> list[str]:
    """Train several independent A2C agents and save the best checkpoint per run."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    all_rewards: list[list[float]] = []
    model_paths: list[str] = []
    slug = env_slug(env_name)

    for run in range(num_runs):
        run_seed = BASE_SEED + 1000 * run
        seed_everything(run_seed)

        env = make_env(env_name, seed=run_seed)
        agent = make_agent(env, cfg)
        episode_rewards: list[float] = []

        model_path = os.path.join(RESULTS_DIR, f"a2c_{slug}_run{run}.pth")
        best_score = -float("inf")

        for episode in tqdm(range(total_episodes), desc=f"Training {env_name} run {run + 1}/{num_runs}"):
            reset_seed = run_seed + episode
            reward = run_training_episode(env, agent, cfg, reset_seed=reset_seed)
            episode_rewards.append(reward)

            if len(episode_rewards) >= BEST_MODEL_WINDOW:
                rolling_score = float(np.mean(episode_rewards[-BEST_MODEL_WINDOW:]))
                if rolling_score > best_score:
                    best_score = rolling_score
                    agent.save(model_path)

        if not os.path.exists(model_path):
            agent.save(model_path)
            best_score = float(np.mean(episode_rewards))

        all_rewards.append(episode_rewards)
        model_paths.append(model_path)
        env.close()
        print(f"Run {run + 1} complete. Best rolling reward: {best_score:.2f}. Saved: {model_path}")

    rewards_array = np.asarray(all_rewards, dtype=float)
    save_training_outputs(env_name, rewards_array)
    return model_paths


def save_training_outputs(env_name: str, all_rewards: np.ndarray) -> None:
    """Save training rewards as CSV and plot mean episode return ± one std."""
    slug = env_slug(env_name)
    episodes = np.arange(1, all_rewards.shape[1] + 1)

    df = pd.DataFrame({"episode": episodes})
    for run in range(all_rewards.shape[0]):
        df[f"run{run}"] = all_rewards[run]
    csv_path = os.path.join(RESULTS_DIR, f"a2c_{slug}_returns.csv")
    df.to_csv(csv_path, index=False)

    mean_rewards = all_rewards.mean(axis=0)
    std_rewards = all_rewards.std(axis=0)
    smooth_window = min(100, len(mean_rewards))
    smooth_mean = pd.Series(mean_rewards).rolling(smooth_window, min_periods=1).mean().to_numpy()

    plt.figure(figsize=(12, 7))
    plt.plot(episodes, smooth_mean, label=f"A2C mean episode return ({smooth_window}-episode moving average)")
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.25, label="±1 std across runs")
    plt.title(f"A2C Training on {env_name} ({all_rewards.shape[0]} runs)")
    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plot_path = os.path.join(PLOT_DIR, f"a2c_{slug}_train_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Training rewards saved to {csv_path}")
    print(f"Training plot saved to {plot_path}")


def test(env_name: str, model_paths: list[str], cfg: EnvConfig, test_episodes: int = 100) -> list[float]:
    """Evaluate trained agents using deterministic actions."""
    run_means: list[float] = []
    slug = env_slug(env_name)
    all_test_rewards: list[list[float]] = []

    for run, model_path in enumerate(model_paths):
        env = make_env(env_name, seed=BASE_SEED + 10000 + run)
        agent = make_agent(env, cfg)
        agent.load(model_path)

        run_rewards: list[float] = []
        for episode in range(test_episodes):
            state, _ = env.reset(seed=BASE_SEED + 100000 + run * test_episodes + episode)
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):
                action = agent.choose_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)

            run_rewards.append(episode_reward)

        avg_reward = float(np.mean(run_rewards))
        run_means.append(avg_reward)
        all_test_rewards.append(run_rewards)
        env.close()
        print(f"Avg deterministic test reward for run {run + 1}: {avg_reward:.2f}")

    mean_score = float(np.mean(run_means))
    std_score = float(np.std(run_means, ddof=1)) if len(run_means) > 1 else 0.0
    print(f"Mean deterministic performance across {len(model_paths)} runs: {mean_score:.2f} +/- {std_score:.2f}")

    df = pd.DataFrame({"episode": np.arange(1, test_episodes + 1)})
    for run, rewards in enumerate(all_test_rewards):
        df[f"run{run}"] = rewards
    df.to_csv(os.path.join(RESULTS_DIR, f"a2c_{slug}_test_returns.csv"), index=False)

    return run_means


def evaluate_random_agent(env_name: str, num_runs: int = 5, test_episodes: int = 100) -> list[float]:
    """Evaluate a random policy and return one mean score per run."""
    run_means: list[float] = []
    slug = env_slug(env_name)
    all_rewards: list[list[float]] = []

    for run in range(num_runs):
        seed = BASE_SEED + 200000 + run
        env = make_env(env_name, seed=seed)
        run_rewards: list[float] = []

        for episode in tqdm(range(test_episodes), desc=f"Random {env_name} run {run + 1}/{num_runs}"):
            state, _ = env.reset(seed=seed + episode)
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):
                action = env.action_space.sample()
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)

            run_rewards.append(episode_reward)

        run_mean = float(np.mean(run_rewards))
        run_means.append(run_mean)
        all_rewards.append(run_rewards)
        env.close()
        print(f"Random agent average reward for run {run + 1}: {run_mean:.2f}")

    df = pd.DataFrame({"episode": np.arange(1, test_episodes + 1)})
    for run, rewards in enumerate(all_rewards):
        df[f"run{run}"] = rewards
    df.to_csv(os.path.join(RESULTS_DIR, f"random_{slug}_test_returns.csv"), index=False)

    mean_score = float(np.mean(run_means))
    std_score = float(np.std(run_means, ddof=1)) if len(run_means) > 1 else 0.0
    print(f"Random agent mean reward across {num_runs} runs: {mean_score:.2f} +/- {std_score:.2f}")
    return run_means


def existing_model_paths(env_name: str, num_runs: int) -> list[str]:
    slug = env_slug(env_name)
    return [os.path.join(RESULTS_DIR, f"a2c_{slug}_run{run}.pth") for run in range(num_runs)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test an improved A2C agent.")
    parser.add_argument("--env", type=str, default="InvertedPendulum-v5", choices=list(ENV_CONFIGS))
    parser.add_argument("--mode", type=str, default="train_and_test", choices=["train", "test", "train_and_test", "analysis"])
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    args = parser.parse_args()

    base_cfg = ENV_CONFIGS[args.env]
    cfg = EnvConfig(
        episodes=args.episodes if args.episodes is not None else base_cfg.episodes,
        lr=args.lr if args.lr is not None else base_cfg.lr,
        gamma=args.gamma if args.gamma is not None else base_cfg.gamma,
        rollout_steps=args.rollout_steps if args.rollout_steps is not None else base_cfg.rollout_steps,
        gae_lambda=base_cfg.gae_lambda,
        entropy_coef=base_cfg.entropy_coef,
        value_loss_coef=base_cfg.value_loss_coef,
        grad_clip=base_cfg.grad_clip,
        hidden_size=base_cfg.hidden_size,
        initial_log_std=base_cfg.initial_log_std,
    )

    print(f"Environment: {args.env}")
    print(f"Config: {cfg}")

    if args.mode == "train":
        train(args.env, cfg.episodes, cfg, args.num_runs)

    elif args.mode == "test":
        paths = existing_model_paths(args.env, args.num_runs)
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing model files: {missing}")
        test(args.env, paths, cfg, args.test_episodes)

    elif args.mode == "train_and_test":
        paths = train(args.env, cfg.episodes, cfg, args.num_runs)
        test(args.env, paths, cfg, args.test_episodes)

    elif args.mode == "analysis":
        paths = existing_model_paths(args.env, args.num_runs)
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("Train the models first; missing: " + ", ".join(missing))

        a2c_scores = test(args.env, paths, cfg, args.test_episodes)
        random_scores = evaluate_random_agent(args.env, args.num_runs, args.test_episodes)
        t_statistic, p_value = stats.ttest_ind(a2c_scores, random_scores, equal_var=False)

        print("\nWelch t-test on run-level mean test rewards")
        print(f"T-statistic: {t_statistic:.4f}")
        print(f"P-value: {p_value:.4f}")


if __name__ == "__main__":
    main()
