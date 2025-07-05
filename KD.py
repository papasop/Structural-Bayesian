# --- Structural Entropy Dynamics with RK45 + τ(t) Prediction Model (Enhanced with LSTM, Z-score Anomaly, Labeling, Residual Expansion, Structural Features) ---

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp, trapezoid
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from scipy.stats import zscore

# --- Define Parameter Grids ---
K_vals = np.linspace(2.4, 2.7, 81)
D_vals = np.linspace(60, 75, 81)
K_grid, D_grid = np.meshgrid(K_vals, D_vals)

# --- Load Real Residuals from CSV (if available) ---
try:
    residuals = np.loadtxt("real_residuals.csv")
except Exception:
    residuals = np.random.normal(0, 1, size=100)

# --- Dynamic Moving Centers Function ---
def moving_centers(t):
    return [
        (2.56 + 0.02 * np.sin(2*np.pi*t/100), 68.82 + 0.01*np.cos(2*np.pi*t/100)),
        (2.65 + 0.01 * np.sin(2*np.pi*t/40), 70.0 + 0.01*np.cos(2*np.pi*t/60))
    ]

# --- Define Entropy Surface with Dynamic Perturbation ---
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

# --- τ(t) Feature Construction ---
dtau = np.gradient(tau_values)
d2tau = np.gradient(dtau)
phi_var = np.var(trajectory, axis=1)
phi_energy = np.sum(trajectory**2, axis=1)
features = np.vstack([tau_values, dtau, d2tau, phi_var, phi_energy]).T
labels = np.zeros_like(tau_values)
z_scores = zscore(d2tau)
anomaly_indices = np.where(np.abs(z_scores) > 2)[0]
labels[anomaly_indices] = 1

# --- Prepare Data for LSTM ---
window_size = 5
X_seq = []
y_seq = []
for i in range(len(features) - window_size):
    X_seq.append(features[i:i+window_size])
    y_seq.append(labels[i+window_size])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42, shuffle=False)

# --- Build LSTM Classifier ---
model = Sequential([
    Input(shape=(window_size, X_seq.shape[2])),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=8, verbose=0)

# --- Prediction ---
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs.flatten() > 0.5).astype(int)

# --- Report ---
print("\n--- τ(t) Anomaly Detection Report (LSTM Classifier) ---")
print(classification_report(y_test, y_pred))

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
plt.scatter(t_eval[anomaly_indices], z_scores[anomaly_indices], color='black', label='Anomalies')
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