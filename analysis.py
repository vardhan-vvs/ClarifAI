import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import os

from dqn import QNetwork
from stable_baselines3 import PPO

from stability import compute_stability
from faith import compute_faithfulness


# ================================
# Wrappers for SHAP
# ================================
class DQNWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        x = torch.FloatTensor(x)
        with torch.no_grad():
            return self.model(x).numpy()      # Q-values


class PPOWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        outputs = []
        for s in x:
            action, _ = self.model.predict(s, deterministic=True)
            outputs.append([action])
        return np.array(outputs)              # actions


# ================================
# Run SHAP
# ================================
def run_shap(model_wrapper, states, model_type, env_name):
    print(f"\nRunning SHAP for {model_type} on {env_name}")

    # Sampling for speed
    if len(states) > 800:
        states = states[np.random.choice(len(states), 800, replace=False)]

    background = states[:80]
    samples = states[80:200]

    explainer = shap.KernelExplainer(model_wrapper.predict, background)
    shap_vals = explainer.shap_values(samples)

    return samples, shap_vals


# ================================
# SHAP Plot
# ================================
def plot_shap(samples, shap_values, env_name, model_type, seed):
    os.makedirs("results", exist_ok=True)

    feature_names = [f"Feature {i}" for i in range(samples.shape[1])]

    plt.figure()
    shap.summary_plot(shap_values, samples, feature_names=feature_names, show=False)
    plt.title(f"{model_type} SHAP — {env_name} (seed {seed})")
    save_path = f"results/{model_type}_{env_name}_seed{seed}_shap.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved SHAP plot → {save_path}")


# ================================
# MAIN PIPELINE
# ================================
def run_analysis():
    envs = ["CartPole-v1", "LunarLander-v2"]
    seeds = [0]     # SHAP is heavy → use 1 seed

    for env_name in envs:
        for seed in seeds:
            print(f"\n===== {env_name} | Seed {seed} =====\n")

            # ------------------------------------------------------
            #                   DQN SECTION
            # ------------------------------------------------------
            try:
                dqn_model_path = f"models/dqn_{env_name}_seed{seed}.pth"
                dqn_state_path = f"states/{env_name}_seed{seed}_states.npy"

                states = np.load(dqn_state_path)
                state_dim = states.shape[1]
                action_dim = 2 if env_name == "CartPole-v1" else 4

                model = QNetwork(state_dim, action_dim)
                model.load_state_dict(torch.load(dqn_model_path))
                model.eval()

                wrapper = DQNWrapper(model)

                samples, shap_vals = run_shap(wrapper, states, "DQN", env_name)
                plot_shap(samples, shap_vals, env_name, "DQN", seed)

                # ===== Stability =====
                stability = compute_stability(shap_vals, samples, wrapper)
                print(f"DQN Stability: {stability}")

                # ===== Faithfulness =====
                important_feature = np.argmax(np.mean(np.abs(shap_vals), axis=0))
                baseline, masked, drop = compute_faithfulness(wrapper, states, env_name, important_feature)

                print(f"DQN Faithfulness → Baseline: {baseline}, Masked: {masked}, Drop: {drop}")

            except Exception as e:
                print(f"[DQN Skipped] Reason → {e}")


            # ------------------------------------------------------
            #                   PPO SECTION
            # ------------------------------------------------------
            try:
                ppo_model_path = f"models/ppo_{env_name}_seed{seed}"
                ppo_state_path = f"states/{env_name}_ppo_seed{seed}_states.npy"

                states = np.load(ppo_state_path)
                model = PPO.load(ppo_model_path)

                wrapper = PPOWrapper(model)

                samples, shap_vals = run_shap(wrapper, states, "PPO", env_name)
                plot_shap(samples, shap_vals, env_name, "PPO", seed)

                # ===== Stability =====
                stability = compute_stability(shap_vals, samples, wrapper)
                print(f"PPO Stability: {stability}")

                # ===== Faithfulness =====
                important_feature = np.argmax(np.mean(np.abs(shap_vals), axis=0))
                baseline, masked, drop = compute_faithfulness(wrapper, states, env_name, important_feature)

                print(f"PPO Faithfulness → Baseline: {baseline}, Masked: {masked}, Drop: {drop}")

            except Exception as e:
                print(f"[PPO Skipped] Reason → {e}")


if __name__ == "__main__":
    run_analysis()