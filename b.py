# --- Structural Bayesian Inference (修复均值 + CI + 归一化) ---

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.special import logsumexp
import matplotlib.pyplot as plt

# --- 参数网格 ---
K_vals = np.linspace(0.85, 0.95, 51)
D_vals = np.linspace(0.05, 0.09, 41)
sigma_vals = np.linspace(0.45, 0.55, 21)
t_vals = np.linspace(1e-3, 50, 3000)

dk = K_vals[1] - K_vals[0]
dd = D_vals[1] - D_vals[0]
ds = sigma_vals[1] - sigma_vals[0]
volume_element = dk * dd * ds

# --- 生成观测数据 ---
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
t_obs = np.interp(np.random.rand(300), cdf, t_vals)

# --- 后验计算 ---
results = []
for K in K_vals:
    for D in D_vals:
        for sigma in sigma_vals:
            joint = joint_density(t_vals, K, D, sigma)
            log_joint = np.log(joint + 1e-20)
            interp_log_joint = np.interp(t_obs, t_vals, log_joint)
            log_like = np.sum(interp_log_joint)
            results.append((K, D, sigma, log_like))

df = pd.DataFrame(results, columns=["K", "D", "Sigma", "LogPosterior"])
logZ = logsumexp(df["LogPosterior"])
df["Normalized_Posterior"] = np.exp(df["LogPosterior"] - logZ)

# --- 归一化检查 ---
norm_sum = df["Normalized_Posterior"].sum()
integral = norm_sum * volume_element
print(f"✅ Normalization Check (sum): {norm_sum}")
print(f"✅ Posterior Integral (Simpson approx): {integral}")

# --- MAP ---
map_idx = df["Normalized_Posterior"].idxmax()
map_row = df.loc[map_idx]
print("🔍 MAP Estimate:\n", map_row)

# --- Posterior Mean Estimates（修复版） ---
mean_K = np.sum(df['K'] * df['Normalized_Posterior'])
mean_D = np.sum(df['D'] * df['Normalized_Posterior'])
mean_Sigma = np.sum(df['Sigma'] * df['Normalized_Posterior'])
print("📈 Posterior Mean Estimates:")
print(f" K: {mean_K:.5f}, D: {mean_D:.5f}, σ: {mean_Sigma:.5f}")

# --- 95% Credible Intervals ---
def get_credible_interval(grouped):
    sorted_vals = grouped.sort_values(ascending=False)
    cumulative = np.cumsum(sorted_vals.values)
    cumulative /= cumulative[-1]
    mask = cumulative <= 0.95
    values = sorted_vals.index[mask]
    return float(np.min(values)), float(np.max(values))

marg_K = df.groupby("K")["Normalized_Posterior"].sum()
marg_D = df.groupby("D")["Normalized_Posterior"].sum()
marg_S = df.groupby("Sigma")["Normalized_Posterior"].sum()

ci_K = get_credible_interval(marg_K)
ci_D = get_credible_interval(marg_D)
ci_S = get_credible_interval(marg_S)

print("📊 95% Credible Intervals:")
print(f" K ∈ [{ci_K[0]:.3f}, {ci_K[1]:.3f}]")
print(f" D ∈ [{ci_D[0]:.3f}, {ci_D[1]:.3f}]")
print(f" σ ∈ [{ci_S[0]:.3f}, {ci_S[1]:.3f}]")

# --- 可视化边缘后验 ---
plt.figure(figsize=(12, 3.5))
plt.subplot(1, 3, 1)
plt.plot(marg_K.index, marg_K.values * dd * ds)
plt.title("Marginal p(K)")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(marg_D.index, marg_D.values * dk * ds)
plt.title("Marginal p(D)")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(marg_S.index, marg_S.values * dk * dd)
plt.title("Marginal p(σ)")
plt.grid(True)

plt.tight_layout()
plt.show()
