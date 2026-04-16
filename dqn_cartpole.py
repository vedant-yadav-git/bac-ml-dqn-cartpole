import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# hyperparams
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 10
NUM_EPISODES = 600


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a).unsqueeze(1),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(ns)),
            torch.FloatTensor(d)
        )

    def __len__(self):
        return len(self.buffer)


env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=NUM_EPISODES)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer(BUFFER_SIZE)

eps = EPS_START

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False

    while not done:
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state)).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, float(done))
        state = next_state

        if len(buffer) >= BATCH_SIZE:
            s, a, r, ns, d = buffer.sample(BATCH_SIZE)

            q = policy_net(s).gather(1, a).squeeze()
            with torch.no_grad():
                q_next = target_net(ns).max(1)[0]
                q_target = r + GAMMA * q_next * (1 - d)

            loss = nn.MSELoss()(q, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    eps = max(EPS_END, eps * EPS_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 50 == 0:
        avg = np.mean(list(env.return_queue)[-50:]) if env.return_queue else 0
        print(f"episode {episode} | avg reward (last 50): {avg:.1f} | eps: {eps:.3f}")


# plot training
rolling_length = 50

def moving_avg(arr, window):
    return np.convolve(np.array(arr).flatten(), np.ones(window), mode="valid") / window

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

rewards = moving_avg(env.return_queue, rolling_length)
axs[0].plot(rewards)
axs[0].set_title("reward (rolling avg)")
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")

lengths = moving_avg(env.length_queue, rolling_length)
axs[1].plot(lengths)
axs[1].set_title("episode length (rolling avg)")
axs[1].set_xlabel("episode")
axs[1].set_ylabel("steps")

plt.tight_layout()
plt.savefig("training.png")
plt.show()
