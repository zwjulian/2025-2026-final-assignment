from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from agent import A2CAgent
from main import BASE_SEED, ENV_CONFIGS, env_slug, run_training_episode, seed_everything

N_RUNS = 5
MAX_WORKERS = 5
RESULTS_DIR = "data"


def train_one_run(env_name: str, run: int) -> dict:
    cfg = ENV_CONFIGS[env_name]
    seed = BASE_SEED + 1000 * run
    seed_everything(seed)

    env = gym.make(env_name)
    env.action_space.seed(seed)
    agent = A2CAgent(
        env,
        learning_rate=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        entropy_coef=cfg.entropy_coef,
        value_loss_coef=cfg.value_loss_coef,
        grad_clip=cfg.grad_clip,
        hidden_size=cfg.hidden_size,
        initial_log_std=cfg.initial_log_std,
        normalize_obs=True,
    )

    rewards = []
    for episode in tqdm(range(cfg.episodes), desc=f"{env_name} run {run}", position=run % MAX_WORKERS):
        rewards.append(run_training_episode(env, agent, cfg, reset_seed=seed + episode))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, f"a2c_{env_slug(env_name)}_run{run}.pth")
    agent.save(model_path)
    env.close()

    return {"env_name": env_name, "run": run, "rewards": rewards, "model_path": model_path}


def main() -> None:
    tasks = [(env_name, run) for env_name in ENV_CONFIGS for run in range(N_RUNS)]
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(train_one_run, env_name, run): (env_name, run) for env_name, run in tasks}
        for future in as_completed(future_map):
            env_name, run = future_map[future]
            result = future.result()
            results.append(result)
            print(f"Finished {env_name} run {run}; saved {result['model_path']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for env_name in ENV_CONFIGS:
        env_results = sorted([r for r in results if r["env_name"] == env_name], key=lambda r: r["run"])
        max_len = max(len(r["rewards"]) for r in env_results)
        df = pd.DataFrame({"episode": np.arange(1, max_len + 1)})
        for result in env_results:
            df[f"run{result['run']}"] = result["rewards"]
        csv_path = os.path.join(RESULTS_DIR, f"a2c_{env_slug(env_name)}_returns.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
