# Structural Bayesian: Minimal Success Version (MAP + Normalization + Marginals)

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.special import logsumexp
import matplotlib.pyplot as plt

# === Step 1: Define Parameter Grid ===
K_vals = np.linspace(0.85, 0.95, 51)          # around true K=0.91
D_vals = np.linspace(0.05, 0.09, 41)          # around true D=0.07
sigma_vals = np.linspace(0.45, 0.55, 21)      # around true sigma=0.5
t_vals = np.linspace(1e-3, 50, 3000)          # time domain

# Step sizes
dk = K_vals[1] - K_vals[0]
dd = D_vals[1] - D_vals[0]
ds = sigma_vals[1] - sigma_vals[0]

# === Step 2: Generate Synthetic Data ===
K_true, D_true, sigma_true = 0.91, 0.07, 0.5

def joint_density(t, K, D, sigma):
    p_K = K * t**(K - 1) * np.exp(-t)
    f_D = np.exp(-D * t)
    f_sigma = np.exp(-t**2 / (2 * sigma**2))
    joint = p_K * f_D * f_sigma
    return joint / simpson(joint, t)

joint_true = joint_density(t_vals, K_true, D_true, sigma_true)
cdf = np.cumsum(joint_true)
cdf /= cdf[-1]
t_obs = np.interp(np.random.rand(200), cdf, t_vals)

# === Step 3: Compute Log-Posterior Grid ===
results = []
for K in K_vals:
    for D in D_vals:
        for sigma in sigma_vals:
            joint = joint_density(t_vals, K, D, sigma)
            log_joint = np.log(joint + 1e-20)
            interp_log_joint = np.interp(t_obs, t_vals, log_joint)
            log_like = np.sum(interp_log_joint)
            results.append((K, D, sigma, log_like))

posterior_df = pd.DataFrame(results, columns=["K", "D", "Sigma", "LogPosterior"])

# === Step 4: Normalize Posterior ===
logZ = logsumexp(posterior_df["LogPosterior"].values)
posterior_df["Normalized_Posterior"] = np.exp(posterior_df["LogPosterior"] - logZ)

# === Step 5: MAP Estimate ===
map_idx = posterior_df["Normalized_Posterior"].idxmax()
map_row = posterior_df.loc[map_idx]

# === Step 6: Integral Check ===
norm_sum = posterior_df["Normalized_Posterior"].sum()
integral = norm_sum * dk * dd * ds

print("\u2705 Normalization Check (sum):", norm_sum)
print("\u2705 Posterior Integral (Simpson approx):", integral)
print("\U0001F50D MAP Estimate:\n", map_row)

# === Step 7: Plot Marginal Distributions ===
marg_K = posterior_df.groupby("K")["Normalized_Posterior"].sum() * dd * ds
marg_D = posterior_df.groupby("D")["Normalized_Posterior"].sum() * dk * ds
marg_S = posterior_df.groupby("Sigma")["Normalized_Posterior"].sum() * dk * dd

plt.figure(figsize=(12, 3.5))
plt.subplot(1, 3, 1)
plt.plot(marg_K.index, marg_K.values)
plt.title("Marginal p(K)")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(marg_D.index, marg_D.values)
plt.title("Marginal p(D)")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(marg_S.index, marg_S.values)
plt.title("Marginal p(σ)")
plt.grid(True)

plt.tight_layout()
plt.show()
