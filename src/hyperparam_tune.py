from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from agent import A2CAgent
from main import EnvConfig, run_training_episode, seed_everything


LEARNING_RATES = [5e-5, 1e-4, 2e-4, 3e-4]
GAMMAS = [0.98, 0.99, 0.995]
EPISODES = 1000
RUNS_PER_SETTING = 2


def evaluate_setting(lr: float, gamma: float) -> float:
    run_scores = []
    cfg = EnvConfig(
        episodes=EPISODES,
        lr=lr,
        gamma=gamma,
        rollout_steps=32,
        gae_lambda=0.95,
        entropy_coef=0.002,
        value_loss_coef=0.5,
        grad_clip=0.5,
        hidden_size=256,
        initial_log_std=-1.0,
    )

    for run in range(RUNS_PER_SETTING):
        seed = 1000 + run
        seed_everything(seed)
        env = gym.make("Pusher-v5")
        env.action_space.seed(seed)
        agent = A2CAgent(
            env,
            learning_rate=lr,
            gamma=gamma,
            gae_lambda=cfg.gae_lambda,
            entropy_coef=cfg.entropy_coef,
            value_loss_coef=cfg.value_loss_coef,
            grad_clip=cfg.grad_clip,
            hidden_size=cfg.hidden_size,
            initial_log_std=cfg.initial_log_std,
            normalize_obs=True,
        )

        rewards = []
        for episode in tqdm(range(EPISODES), desc=f"lr={lr:g}, gamma={gamma:g}, run={run + 1}"):
            rewards.append(run_training_episode(env, agent, cfg, reset_seed=seed + episode))
        env.close()
        run_scores.append(np.mean(rewards[-100:]))

    return float(np.mean(run_scores))


def main() -> None:
    records = []
    for lr in LEARNING_RATES:
        for gamma in GAMMAS:
            score = evaluate_setting(lr, gamma)
            records.append({"lr": lr, "gamma": gamma, "last_100_avg_reward": score})
            print(f"lr={lr:g}, gamma={gamma:g}, score={score:.2f}")

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv("data/pusher_hyperparam_sweep.csv", index=False)
    print(df.pivot(index="lr", columns="gamma", values="last_100_avg_reward"))

    pivot = df.pivot(index="lr", columns="gamma", values="last_100_avg_reward")
    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Mean reward over last 100 episodes")
    plt.xlabel("Gamma")
    plt.ylabel("Learning rate")
    plt.title("Pusher-v5 A2C hyperparameter sweep")
    plt.tight_layout()
    plt.savefig("data/pusher_hyperparam_sweep.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
