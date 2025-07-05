# --- Structural Entropy Dynamics with RK45 + τ(t) Prediction Model (Enhanced with IsolationForest + Multi-System Residuals + Entropy Surface) ---

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp, trapezoid
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from scipy.stats import zscore

# --- Define Parameter Grids ---
K_vals = np.linspace(1.0, 10.0, 201)
D_vals = np.linspace(60, 75, 81)
K_grid, D_grid = np.meshgrid(K_vals, D_vals)

# --- Load Residuals for Different Systems ---
system = 'earthquake'  # options: 'eeg', 'earthquake', 'btc'
residual_files = {
    'eeg': "residuals_eeg.csv",
    'earthquake': "residuals_quake.csv",
    'btc': "residuals_btc.csv"
}
try:
    residuals = np.loadtxt(residual_files[system])
except Exception:
    residuals = np.random.normal(0, 1, size=100)

# --- Dynamic Moving Centers ---
def moving_centers(t):
    return [
        (2.56 + 0.02 * np.sin(2*np.pi*t/100), 68.82 + 0.01*np.cos(2*np.pi*t/100)),
        (2.65 + 0.01 * np.sin(2*np.pi*t/40), 70.0 + 0.01*np.cos(2*np.pi*t/60)),
        (4.5 + 0.05 * np.sin(2*np.pi*t/55), 73.0 + 0.01*np.cos(2*np.pi*t/35))
    ]

# --- Entropy Surface ---
def dynamic_residual_perturbation(K, D, t, residuals, centers, A=1.0, T=30):
    total_perturbation = 0
    residual_factor = np.mean(np.abs(residuals[int(t)])) if t < len(residuals) else 1.0
    for center in centers:
        x0, y0 = center
        dx2 = (K - x0)**2 + (D - y0)**2
        total_perturbation += A * residual_factor * np.exp(-200 * dx2) * np.sin(2 * np.pi * t / T)
    return total_perturbation

def entropy_surface(K, D, t, residuals):
    centers = moving_centers(t)
    return dynamic_residual_perturbation(K, D, t, residuals, centers)

# --- RK45 Dynamics ---
def ids_dynamics(t, phi, K_grid, D_grid, residuals):
    K, D = phi
    K = np.clip(K, K_vals[0], K_vals[-1])
    D = np.clip(D, D_vals[0], D_vals[-1])
    H_t = entropy_surface(K_grid, D_grid, t, residuals)
    H_t = gaussian_filter(H_t, sigma=2.0)
    grad_K, grad_D = np.gradient(H_t, K_vals[1] - K_vals[0], D_vals[1] - D_vals[0])
    interp_gK = RegularGridInterpolator((D_vals, K_vals), grad_K, bounds_error=False, fill_value=0.0)
    interp_gD = RegularGridInterpolator((D_vals, K_vals), grad_D, bounds_error=False, fill_value=0.0)
    gk = interp_gK([[D, K]])[0]
    gd = interp_gD([[D, K]])[0]
    return [-gk, -gd]

# --- Solve System ---
t_span = [0, 100]
t_eval = np.arange(0, 101, 1)
phi0 = [2.57, 69.0]
sol = solve_ivp(lambda t, phi: ids_dynamics(t, phi, K_grid, D_grid, residuals), t_span, phi0, method='RK45', t_eval=t_eval)
trajectory = np.array([sol.y[0], sol.y[1]]).T
K_final, D_final = trajectory[-1]

# --- Compute τ(t) ---
tau_values = np.zeros(len(t_eval))
for i, t in enumerate(t_eval):
    H_t = entropy_surface(K_grid, D_grid, t, residuals)
    H_t = gaussian_filter(H_t, sigma=2.0)
    grad_K, grad_D = np.gradient(H_t, K_vals[1] - K_vals[0], D_vals[1] - D_vals[0])
    idx_K = np.argmin(np.abs(K_vals - trajectory[i, 0]))
    idx_D = np.argmin(np.abs(D_vals - trajectory[i, 1]))
    grad_norm = np.sqrt(grad_K[idx_D, idx_K]**2 + grad_D[idx_D, idx_K]**2)
    tau_values[i] = trapezoid(np.ones(i+1) * grad_norm, t_eval[:i+1]) if i > 0 else 0

# --- Anomaly Detection with Isolation Forest ---
dtau = np.gradient(tau_values)
d2tau = np.gradient(dtau)
features = np.vstack([tau_values, dtau, d2tau]).T
z_scores = zscore(d2tau)
thresh = np.percentile(np.abs(z_scores), 97)
labels = (np.abs(z_scores) > thresh).astype(int)

iso = IsolationForest(contamination=0.03, random_state=0)
y_pred = iso.fit_predict(features)
y_pred = (y_pred == -1).astype(int)

# --- Classification Report ---
print("\n--- τ(t) Anomaly Detection Report (Isolation Forest) ---")
print(classification_report(labels, y_pred))

# --- Output Results ---
print(f"\nFinal Structural Attractor Parameters (RK45):")
print(f"K* ≈ {K_final:.5f}")
print(f"D* ≈ {D_final:.5f}")
print(f"\nτ(t=100) ≈ {tau_values[-1]:.5f}")
print(f"Anomaly Indices in τ(t): {np.where(labels == 1)[0].tolist()}")

# --- Visualization ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=3)
plt.title("Trajectory on (K, D)")
plt.xlabel("K")
plt.ylabel("D")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(t_eval, tau_values, label="τ(t)")
plt.title("Structure Time τ(t)")
plt.xlabel("Time")
plt.ylabel("τ")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(t_eval, z_scores, label="Z-score of d²τ/dt²", color='purple')
plt.scatter(t_eval[labels == 1], z_scores[labels == 1], color='black', label='Anomalies')
plt.title("Anomaly Z-scores")
plt.xlabel("Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Entropy Value at φ* ---
H_star = entropy_surface(K_grid, D_grid, 100, residuals)
idx_K_star = np.argmin(np.abs(K_vals - 2.56))
idx_D_star = np.argmin(np.abs(D_vals - 68.82))
print(f"\nEntropy at φ* = {H_star[idx_D_star, idx_K_star]:.5e}")
