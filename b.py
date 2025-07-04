import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.special import logsumexp
import matplotlib.pyplot as plt

# === Step 1: Define Parameter Grid ===
K_vals = np.linspace(0.85, 0.95, 51)
D_vals = np.linspace(0.05, 0.09, 41)
S_vals = np.linspace(0.45, 0.55, 21)
t_vals = np.linspace(1e-3, 50, 3000)

dk, dd, ds = K_vals[1] - K_vals[0], D_vals[1] - D_vals[0], S_vals[1] - S_vals[0]

# === Step 2: Generate Synthetic Observation ===
K_true, D_true, S_true = 0.91, 0.07, 0.5
def joint_density(t, K, D, S):
    pK = K * t**(K-1) * np.exp(-t)
    fD = np.exp(-D * t)
    fS = np.exp(-t**2 / (2 * S**2))
    joint = pK * fD * fS
    return joint / simpson(joint, t)

joint_true = joint_density(t_vals, K_true, D_true, S_true)
cdf = np.cumsum(joint_true); cdf /= cdf[-1]
t_obs = np.interp(np.random.rand(200), cdf, t_vals)

# === Step 3: Evaluate Log-Posterior ===
results = []
for K in K_vals:
    for D in D_vals:
        for S in S_vals:
            p = joint_density(t_vals, K, D, S)
            logp = np.log(p + 1e-20)
            interp = np.interp(t_obs, t_vals, logp)
            log_like = np.sum(interp)
            results.append((K, D, S, log_like))

df = pd.DataFrame(results, columns=["K", "D", "Sigma", "LogPosterior"])

# === Step 4: Normalize Posterior ===
logZ = logsumexp(df["LogPosterior"])
df["Normalized_Posterior"] = np.exp(df["LogPosterior"] - logZ)

# === Step 5: MAP + Mean Estimate
map_row = df.loc[df["Normalized_Posterior"].idxmax()]
mean_K = np.sum(df["K"] * df["Normalized_Posterior"]) * dk * dd * ds
mean_D = np.sum(df["D"] * df["Normalized_Posterior"]) * dk * dd * ds
mean_S = np.sum(df["Sigma"] * df["Normalized_Posterior"]) * dk * dd * ds

# === Step 6: Marginal Distributions
marg_K = df.groupby("K")["Normalized_Posterior"].sum() * dd * ds
marg_D = df.groupby("D")["Normalized_Posterior"].sum() * dk * ds
marg_S = df.groupby("Sigma")["Normalized_Posterior"].sum() * dk * dd

def credible_interval(x, px, alpha=0.95):
    px = px / np.sum(px)
    sorted_idx = np.argsort(px)[::-1]
    cum = 0
    mask = np.zeros_like(px, dtype=bool)
    for i in sorted_idx:
        cum += px[i]
        mask[i] = True
        if cum >= alpha:
            break
    return np.min(x[mask]), np.max(x[mask])

K_low, K_high = credible_interval(marg_K.index.values, marg_K.values)
D_low, D_high = credible_interval(marg_D.index.values, marg_D.values)
S_low, S_high = credible_interval(marg_S.index.values, marg_S.values)

# === Step 7: Print Results ===
print("✅ Normalization Check (sum):", df["Normalized_Posterior"].sum())
print("✅ Posterior Integral (Simpson approx):", df["Normalized_Posterior"].sum() * dk * dd * ds)
print("🔍 MAP Estimate:\n", map_row)
print("📈 Posterior Mean Estimates:\n",
      f"K: {mean_K:.5f}, D: {mean_D:.5f}, σ: {mean_S:.5f}")
print("📊 95% Credible Intervals:\n",
      f"K ∈ [{K_low:.3f}, {K_high:.3f}]\n",
      f"D ∈ [{D_low:.3f}, {D_high:.3f}]\n",
      f"σ ∈ [{S_low:.3f}, {S_high:.3f}]")

# === Step 8: Plot Marginals
plt.figure(figsize=(12, 3.5))
plt.subplot(1, 3, 1)
plt.plot(marg_K.index, marg_K.values); plt.title("Marginal p(K)"); plt.grid(True)
plt.axvline(mean_K, color="gray", linestyle="--", label="mean")
plt.axvline(map_row["K"], color="red", linestyle=":", label="MAP")

plt.subplot(1, 3, 2)
plt.plot(marg_D.index, marg_D.values); plt.title("Marginal p(D)"); plt.grid(True)
plt.axvline(mean_D, color="gray", linestyle="--")
plt.axvline(map_row["D"], color="red", linestyle=":")

plt.subplot(1, 3, 3)
plt.plot(marg_S.index, marg_S.values); plt.title("Marginal p(σ)"); plt.grid(True)
plt.axvline(mean_S, color="gray", linestyle="--")
plt.axvline(map_row["Sigma"], color="red", linestyle=":")

plt.tight_layout(); plt.legend(); plt.show()
