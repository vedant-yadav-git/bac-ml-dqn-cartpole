import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ── Hyperparameters ──────────────────────────────────────────────────────────
ENV_NAME         = "CartPole-v1"
SEED             = 42
GAMMA            = 0.99          # discount factor
LR               = 1e-3          # Adam learning rate
BATCH_SIZE       = 64
BUFFER_CAPACITY  = 10_000
EPSILON_START    = 1.0
EPSILON_END      = 0.01
EPSILON_DECAY    = 500           # steps for exponential decay
TARGET_UPDATE    = 10            # sync target net every N episodes
NUM_EPISODES     = 500
REWARD_THRESHOLD = 475           # CartPole-v1 solved criterion

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Q-Network ────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    """Two-layer MLP mapping observations to Q-values."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions).unsqueeze(1),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer     = ReplayBuffer(BUFFER_CAPACITY)
        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        eps = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(
            -self.steps_done / EPSILON_DECAY
        )
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            return int(self.policy_net(torch.FloatTensor(state)).argmax().item())

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            best_a  = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q  = self.target_net(next_states).gather(1, best_a).squeeze(1)
            targets = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    env   = gym.make(ENV_NAME)
    agent = DQNAgent(
        state_dim  = env.observation_space.shape[0],
        action_dim = env.action_space.n,
    )
    episode_rewards = []

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset(seed=SEED)
        total_reward = 0.0

        while True:
            action                             = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.learn()
            state        = next_state
            total_reward += reward
            if done:
                break

        if ep % TARGET_UPDATE == 0:
            agent.sync_target()

        episode_rewards.append(total_reward)
        avg_100 = np.mean(episode_rewards[-100:])

        if ep % 50 == 0:
            print(f"Ep {ep:4d} | reward {total_reward:6.1f} | avg-100 {avg_100:6.1f}")

        if avg_100 >= REWARD_THRESHOLD and ep >= 100:
            print(f"Solved in {ep} episodes! avg-100 = {avg_100:.1f}")
            torch.save(agent.policy_net.state_dict(), "dqn_cartpole.pth")
            break

    env.close()
    return agent


if __name__ == "__main__":
    train()
