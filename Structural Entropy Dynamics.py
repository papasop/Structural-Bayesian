# --- Structural Entropy Dynamics with RK45 + τ(t) Calculation (Upgraded) ---

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp, trapezoid

# --- Define Parameter Grids ---
K_vals = np.linspace(2.4, 2.7, 81)
D_vals = np.linspace(60, 75, 81)
K_grid, D_grid = np.meshgrid(K_vals, D_vals)

# --- Define Entropy Surface with Dynamic Residual Perturbation (Supports Multiple Moving Centers) ---
def dynamic_residual_perturbation(K, D, t, residuals, centers=[(2.56, 68.82)], A=1.0, T=30):
    total_perturbation = 0
    residual_factor = np.mean(np.abs(residuals)) if residuals is not None else 1.0
    for center in centers:
        x0, y0 = center
        dx2 = (K - x0)**2 + (D - y0)**2
        total_perturbation += A * residual_factor * np.exp(-200 * dx2) * np.sin(2 * np.pi * t / T)
    return total_perturbation

def entropy_surface(K, D, t, residuals=None, centers=[(2.56, 68.82)]):
    return dynamic_residual_perturbation(K, D, t, residuals, centers)

# --- RK45 Dynamics ---
def ids_dynamics(t, phi, K_grid, D_grid, t_actual, residuals, centers):
    K, D = phi
    K = np.clip(K, K_vals[0], K_vals[-1])
    D = np.clip(D, D_vals[0], D_vals[-1])
    H_t = entropy_surface(K_grid, D_grid, t_actual, residuals, centers)
    H_t = gaussian_filter(H_t, sigma=1.0)
    grad_K, grad_D = np.gradient(H_t, K_vals[1] - K_vals[0], D_vals[1] - D_vals[0])
    interp_gK = RegularGridInterpolator((D_vals, K_vals), grad_K, bounds_error=False, fill_value=None)
    interp_gD = RegularGridInterpolator((D_vals, K_vals), grad_D, bounds_error=False, fill_value=None)
    gk = interp_gK([[D, K]])[0]
    gd = interp_gD([[D, K]])[0]
    return [-gk, -gd]

# --- Solve System ---
t_span = [0, 100]
t_eval = np.arange(0, 101, 1)
phi0 = [2.57, 69.0]
residuals = np.random.normal(0, 1, size=100)
centers = [(2.56, 68.82), (2.65, 70.0)]  # Example with multiple centers

sol = solve_ivp(lambda t, phi: ids_dynamics(t, phi, K_grid, D_grid, t, residuals, centers),
                t_span, phi0, method='RK45', t_eval=t_eval)

trajectory = np.array([sol.y[0], sol.y[1]]).T
K_final, D_final = trajectory[-1]

# --- Compute τ(t) using trapezoid ---
tau_values = np.zeros(len(t_eval))
for i, t in enumerate(t_eval):
    H_t = entropy_surface(K_grid, D_grid, t, residuals, centers)
    H_t = gaussian_filter(H_t, sigma=1.0)
    grad_K, grad_D = np.gradient(H_t, K_vals[1] - K_vals[0], D_vals[1] - D_vals[0])
    idx_K = np.argmin(np.abs(K_vals - trajectory[i, 0]))
    idx_D = np.argmin(np.abs(D_vals - trajectory[i, 1]))
    grad_norm = np.sqrt(grad_K[idx_D, idx_K]**2 + grad_D[idx_D, idx_K]**2)
    tau_values[i] = trapezoid(np.ones(i+1) * grad_norm, t_eval[:i+1]) if i > 0 else 0

# --- τ(t) Anomaly Detection ---
dtau = np.gradient(tau_values)
d2tau = np.gradient(dtau)
threshold = 2 * np.std(d2tau)
anomaly_indices = np.where(np.abs(d2tau) > threshold)[0]

# --- Output Results ---
print(f"\nFinal Structural Attractor Parameters (RK45):")
print(f"K* ≈ {K_final:.5f}")
print(f"D* ≈ {D_final:.5f}")
print(f"\nτ(t=100) ≈ {tau_values[-1]:.5f}")
print(f"Anomaly Indices in τ(t): {anomaly_indices.tolist()}")

# --- Optional: Plot Trajectory and τ(t) Curve ---
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
plt.plot(t_eval, d2tau, label="d²τ/dt²", color='red')
plt.scatter(t_eval[anomaly_indices], d2tau[anomaly_indices], color='black', label='Anomalies')
plt.title("τ(t) Second Derivative")
plt.xlabel("Time")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# --- Entropy Value at φ* ---
H_star = entropy_surface(K_grid, D_grid, 100, residuals, centers)
idx_K_star = np.argmin(np.abs(K_vals - 2.56))
idx_D_star = np.argmin(np.abs(D_vals - 68.82))
print(f"\nEntropy at φ* = {H_star[idx_D_star, idx_K_star]:.5e}")
