import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os

# ========================
# Q-Network
# ========================
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.model(x)


# ========================
# Replay Buffer
# ========================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


# ========================
# DQN Agent
# ========================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.gamma = 0.99
        self.lr = 1e-3
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.memory = ReplayBuffer()
        self.step_count = 0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.q_network.model[-1].out_features)

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.q_network(state)).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ========================
# Training
# ========================
def train_dqn(env_name, seed, episodes=300):
    env = gym.make(env_name)
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            agent.train_step()

        rewards.append(total_reward)

        print(f"[{env_name}] Seed {seed} | Ep {ep+1} | Reward: {total_reward:.2f}")

    env.close()

    return agent, rewards


# ========================
# Evaluation + State Saving
# ========================
def evaluate_and_collect_states(agent, env_name, episodes=10):
    env = gym.make(env_name)

    all_states = []
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            all_states.append(state)  # 🔥 IMPORTANT

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(agent.q_network(state_tensor)).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    return np.array(all_states), rewards


# ========================
# MAIN PIPELINE
# ========================
def run_experiments():
    envs = ["CartPole-v1", "LunarLander-v3"]
    seeds = [0, 42, 100]

    os.makedirs("models", exist_ok=True)
    os.makedirs("states", exist_ok=True)

    results = {}

    for env_name in envs:
        print(f"\n===== Running {env_name} =====")

        all_rewards = []

        for seed in seeds:
            # Train
            agent, rewards = train_dqn(env_name, seed)

            # Save model
            model_path = f"models/dqn_{env_name}_seed{seed}.pth"
            torch.save(agent.q_network.state_dict(), model_path)

            # Evaluate + collect states
            states, test_rewards = evaluate_and_collect_states(agent, env_name)

            # Save states
            np.save(f"states/{env_name}_seed{seed}_states.npy", states)

            all_rewards.append(np.mean(test_rewards))

        # Metrics
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)

        results[env_name] = (mean_reward, std_reward)

        print(f"\n{env_name} FINAL RESULT:")
        print(f"Mean Reward: {mean_reward:.2f}")
        print(f"Std Dev: {std_reward:.2f}")

    return results


# ========================
# RUN
# ========================
if __name__ == "__main__":
    results = run_experiments()
    print("\nFinal Summary:", results)