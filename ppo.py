import gymnasium as gym
import numpy as np
import torch
import os
import random

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed


# ========================
# Training
# ========================
def train_ppo(env_name, seed, timesteps=50000):
    print(f"\nTraining PPO on {env_name} | Seed {seed}")

    env = gym.make(env_name)

    # Set seeds
    env.reset(seed=seed)
    set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        batch_size=64
    )

    model.learn(total_timesteps=timesteps)

    env.close()
    return model


# ========================
# Evaluation + State Saving
# ========================
def evaluate_and_collect_states(model, env_name, episodes=10):
    env = gym.make(env_name)

    all_states = []
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            all_states.append(state)  # 🔥 important for SHAP

            action, _ = model.predict(state, deterministic=True)

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
        print(f"\n===== PPO on {env_name} =====")

        all_rewards = []

        for seed in seeds:
            # Train
            model = train_ppo(env_name, seed)

            # Save model
            model_path = f"models/ppo_{env_name}_seed{seed}"
            model.save(model_path)

            # Evaluate + collect states
            states, test_rewards = evaluate_and_collect_states(model, env_name)

            # Save states
            np.save(f"states/{env_name}_ppo_seed{seed}_states.npy", states)

            all_rewards.append(np.mean(test_rewards))

        # Metrics
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)

        results[env_name] = (mean_reward, std_reward)

        print(f"\n{env_name} PPO FINAL RESULT:")
        print(f"Mean Reward: {mean_reward:.2f}")
        print(f"Std Dev: {std_reward:.2f}")

    return results


# ========================
# RUN
# ========================
if __name__ == "__main__":
    results = run_experiments()
    print("\nFinal Summary:", results)