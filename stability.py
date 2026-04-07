import numpy as np
import shap

def compute_stability(shap_values, states, model_wrapper, noise_scale=0.01):

    # take subset
    samples = states[:100]
    noise = np.random.normal(0, noise_scale, samples.shape)
    noisy_samples = samples + noise

    # shap for noisy states
    explainer = shap.KernelExplainer(model_wrapper.predict, samples[:50])
    shap_noisy = explainer.shap_values(noisy_samples[:50])

    # stability as L2 distance
    shap_original = np.array(shap_values)[:50]
    shap_noisy = np.array(shap_noisy)

    diff = np.linalg.norm(shap_original - shap_noisy, axis=1)
    return float(np.mean(diff))