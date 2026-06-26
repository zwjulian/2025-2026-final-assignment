"""Plot learning curves and compute simple run-level statistics.

This script expects CSVs produced by ``src/main.py`` in the local ``data/``
folder. It plots *episode return*, not cumulative return over all episodes,
because episode return is the quantity that shows whether the policy is improving.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
ROLLING_WINDOW = 100


def load_rewards(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    run_cols = [col for col in df.columns if col.startswith("run")]
    if not run_cols:
        raise ValueError(f"No run columns found in {path}")
    return df["episode"].to_numpy(), df[run_cols].to_numpy(dtype=float)


def rolling_mean(values: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def plot_training_curve(env_short: str, env_full: str) -> None:
    path = os.path.join(DATA_DIR, f"a2c_{env_short}_returns.csv")
    episodes, rewards = load_rewards(path)

    mean = rewards.mean(axis=1)
    std = rewards.std(axis=1)
    smooth = rolling_mean(mean, min(ROLLING_WINDOW, len(mean)))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, smooth, label="A2C mean episode return")
    ax.fill_between(episodes, mean - std, mean + std, alpha=0.25, label="±1 std across runs")
    ax.set_title(f"A2C training performance on {env_full}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{env_short}_training_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def compute_statistics() -> None:
    envs = [
        ("invertedpendulum", "InvertedPendulum-v5"),
        ("pusher", "Pusher-v5"),
    ]
    lines: list[str] = []

    for env_short, env_full in envs:
        a2c_path = os.path.join(DATA_DIR, f"a2c_{env_short}_test_returns.csv")
        random_path = os.path.join(DATA_DIR, f"random_{env_short}_test_returns.csv")
        if not (os.path.exists(a2c_path) and os.path.exists(random_path)):
            lines.append(f"{env_full}: missing test CSVs; run `python src/main.py --env {env_full} --mode analysis` first.")
            lines.append("")
            continue

        _, a2c_rewards = load_rewards(a2c_path)
        _, random_rewards = load_rewards(random_path)
        a2c_run_means = a2c_rewards.mean(axis=0)
        random_run_means = random_rewards.mean(axis=0)

        t_stat, p_value = stats.ttest_ind(a2c_run_means, random_run_means, equal_var=False)

        lines.append(f"--- {env_full} ---")
        lines.append(f"A2C mean ± std    : {a2c_run_means.mean():.2f} ± {a2c_run_means.std(ddof=1):.2f}")
        lines.append(f"Random mean ± std : {random_run_means.mean():.2f} ± {random_run_means.std(ddof=1):.2f}")
        lines.append(f"Welch t-statistic : {t_stat:.4f}")
        lines.append(f"Welch p-value     : {p_value:.4f}")
        lines.append("")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "statistics.txt")
    with open(out_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for short, full in [("invertedpendulum", "InvertedPendulum-v5"), ("pusher", "Pusher-v5")]:
        csv_path = os.path.join(DATA_DIR, f"a2c_{short}_returns.csv")
        if os.path.exists(csv_path):
            plot_training_curve(short, full)
    compute_statistics()
