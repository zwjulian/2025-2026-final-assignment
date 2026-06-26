# Deep Reinforcement Learning Assignment 2

This repository contains an improved Advantage Actor-Critic (A2C) implementation for two Gymnasium MuJoCo continuous-control environments:

- `InvertedPendulum-v5`
- `Pusher-v5`

The main entry point is `src/main.py`.

## Installation

Create a fresh virtual environment and install the requirements:

```bash
pip install -r requirements.txt
```
## Recommended commands

Train and test InvertedPendulum:

```bash
python src/main.py --env InvertedPendulum-v5 --mode train_and_test
```

Train and test Pusher:

```bash
python src/main.py --env Pusher-v5 --mode train_and_test
```

Run analysis against a random baseline after training:

```bash
python src/main.py --env Pusher-v5 --mode analysis
```

Generate plots/statistics from saved CSV files:

```bash
python graph_plotting.py
```

## What was improved

The A2C implementation contains several stability improvements that matter for MuJoCo control:

- tanh-squashed Gaussian actions instead of unbounded actions followed by clipping;
- observation normalization with saved/restored running statistics;
- n-step updates instead of one update only at the end of each episode;
- Generalized Advantage Estimation (GAE) for lower-variance advantage estimates;
- advantage normalization;
- entropy regularization;
- Huber critic loss;
- gradient clipping;
- environment-specific hyperparameters.

Pusher-v5 is substantially harder than InvertedPendulum-v5 and therefore uses more episodes, a lower learning rate, and lower initial policy standard deviation.
