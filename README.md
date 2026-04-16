# BAC ML — DQN CartPole (4/7 Deliverable)

A **Double DQN** agent that solves the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment using PyTorch and Gymnasium.

---

## Task

Balance a pole on a moving cart by applying left/right forces.  
The environment is considered **solved** when the agent achieves an average reward ≥ 475 over 100 consecutive episodes.

## Algorithm

| Component | Detail |
|---|---|
| Algorithm | Double DQN |
| Network | 2-layer MLP (128→128→2) with ReLU |
| Optimizer | Adam (lr = 1e-3) |
| Replay Buffer | 10,000 transitions |
| Batch Size | 64 |
| Discount (γ) | 0.99 |
| ε-greedy decay | Exponential (500 steps) |
| Target sync | Every 10 episodes |

## Setup

```bash
pip install gymnasium torch numpy
```

## Usage

```bash
python dqn_cartpole.py
```

Training prints progress every 50 episodes and saves `dqn_cartpole.pth` once solved.

## Files

| File | Description |
|---|---|
| `dqn_cartpole.py` | Full DQN training script |
| `dqn_cartpole.pth` | Saved model weights (generated after training) |

---

*NYU BAC Machine Learning Team — Spring 2026*
