import numpy as np
import gymnasium as gym

def compute_faithfulness(model_wrapper, states, env_name, important_idx, episodes=5):

    def rollout(modifier=None):
        env = gym.make(env_name)
        total = 0
        
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            reward_sum = 0
            
            while not done:
                if modifier is not None:
                    state = modifier(state)

                action = model_wrapper.predict([state])[0][0]
                state, r, term, trunc, _ = env.step(action)
                reward_sum += r
                done = term or trunc

            total += reward_sum
        
        env.close()
        return total / episodes

    baseline = rollout()

    def mask_feature(s):
        s = s.copy()
        s[important_idx] = 0.0
        return s

    masked = rollout(mask_feature)

    return baseline, masked, baseline - masked