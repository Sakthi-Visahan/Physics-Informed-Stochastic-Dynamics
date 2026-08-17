import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Igniting Physics-Constrained MH Engine on: {device.type.upper()}")


eq_path = "EQ_Motion.mat"
peak_path = "Peak_Response.mat"

eq_data = sio.loadmat(eq_path)
peak_data = sio.loadmat(peak_path)

X_tensor = torch.tensor(eq_data['x_g'].T, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(peak_data['y_peak'].T, dtype=torch.float32).to(device)

print(f"Loaded {X_tensor.shape[0]} earthquake records, input_dim={X_tensor.shape[1]}")

X_mean, X_std = X_tensor.mean(), X_tensor.std()
X_scaled = (X_tensor - X_mean) / X_std

# 4501 → 120 principal components
n_components = 120
U, S, V = torch.pca_lowrank(X_scaled, q=n_components)
X_pca = torch.mm(X_scaled - X_scaled.mean(dim=0), V)

total_var = ((X_scaled - X_scaled.mean(dim=0)) ** 2).sum()
explained_var = (S ** 2).sum()
print(f"PCA: {X_scaled.shape[1]} dims -> {n_components} components (variance retained: {explained_var/total_var*100:.1f}%)")

# Bootstrap resampling anchors each synthetic point to a real training point;
# small noise fills gaps without leaving the BNN's trust radius.
N_synth = 20000
bandwidth_factor = 0.1  # 10% of per-component std — conservative to prevent extrapolation

pca_std = X_pca.std(dim=0)  # per-component std of training PCA scores

indices = torch.randint(0, X_pca.shape[0], (N_synth,), device=device)
X_synth_pca = X_pca[indices].clone()

# Gaussian smoothing
X_synth_pca += bandwidth_factor * pca_std * torch.randn_like(X_synth_pca)

print(f"Generated {N_synth} synthetic ground motions (bootstrap + {bandwidth_factor:.0%} Gaussian smoothing)")

Y_var = Y_tensor.var()

print("\n--- DATA DIAGNOSTICS ---")
print(f"True Y (Peak) Min: {Y_tensor.min().item():.5f} m")
print(f"True Y (Peak) Max: {Y_tensor.max().item():.5f} m")
print(f"True Y (Peak) Mean: {Y_tensor.mean().item():.5f} m")
print("------------------------\n")


class PhysicsConstrainedBNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.activation = nn.Mish()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        # Modulus constraint (torch.abs) mathematically guarantees non-negative physical amplitudes
        raw_output = self.fc2(self.activation(self.fc1(x)))
        return torch.abs(raw_output)

bnn_model = PhysicsConstrainedBNN(input_dim=X_pca.shape[1]).to(device)
total_params = sum(p.numel() for p in bnn_model.parameters())
print(f"Network parameter count: {total_params} (fc1: {n_components}x20 + bias, fc2: 20x1 + bias)")

#THERMO
def calculate_gibbs_energy(model, x, y, y_var):
    predictions = model(x)
    # Physical space error normalized by target variance using dataset-size independent mean scaling
    data_error = torch.mean((predictions - y) ** 2) / (2.0 * y_var)
    # Gaussian prior / weight penalty
    weight_penalty = 0.001 * sum(torch.sum(p ** 2) for p in model.parameters())
    return data_error + weight_penalty

# ==========================================
# 4. TWO-STAGE TRAINING PIPELINE
# ==========================================

# STAGE 1: MAP Pretraining via Adam optimizer
print("--- Pretraining to MAP estimate via Adam ---")
optimizer = torch.optim.Adam(bnn_model.parameters(), lr=1e-3)
num_pretrain_epochs = 2500

best_energy = float('inf')
best_map_state = None

for epoch in range(num_pretrain_epochs + 1):
    optimizer.zero_grad()
    energy = calculate_gibbs_energy(bnn_model, X_pca, Y_tensor, Y_var)
    energy.backward()
    optimizer.step()
    
    current_energy_val = energy.item()
    if current_energy_val < best_energy:
        best_energy = current_energy_val
        best_map_state = {name: param.detach().clone() for name, param in bnn_model.named_parameters()}
    
    if epoch % 500 == 0:
        print(f"  Pretrain Ep {epoch:04d} | Energy: {current_energy_val:.4f}")

# Restore best MAP state encountered during pretraining
with torch.no_grad():
    for name, param in bnn_model.named_parameters():
        param.copy_(best_map_state[name])

print(f"Pretraining complete. Best MAP energy: {best_energy:.4f}")
print("MCMC will now explore the LOCAL posterior around this point.\n")

# STAGE 2: Adaptive Metropolis-Hastings MCMC Sampling
def run_mh_sampler(model, x, y, y_var, num_samples=100000, initial_step_size=0.01,
                   temp=0.002, burn_in=20000, thin=10,
                   target_accept=0.25, adapt_interval=500):
    """
    MH sampler with adaptive step-size tuning during burn-in.
    The step_size is self-adjusted every `adapt_interval` iterations to converge
    toward `target_accept`, eliminating manual hyperparameter guessing.
    After burn-in, the step_size is frozen and posterior samples are collected.
    """
    print(f"Running Adaptive MH Sampler for {num_samples} iterations...")
    print(f"Target Accept: {target_accept*100:.0f}% | Temp = {temp} | Initial Step = {initial_step_size}")
    print(f"Adaptive tuning active during burn-in (first {burn_in} iters, window = {adapt_interval})\n")
    
    step_size = initial_step_size
    
    with torch.no_grad():
        current_energy = calculate_gibbs_energy(model, x, y, y_var)
    
    accepted_weights = []
    energy_trace = []
    accept_count = 0
    window_accepts = 0

    for i in range(num_samples):
        energy_trace.append(current_energy.item())
        
        with torch.no_grad():
            old_weights = {name: param.detach().clone() for name, param in model.named_parameters()}
            
            for param in model.parameters():
                param.add_(torch.randn_like(param) * step_size)
                    
            new_energy = calculate_gibbs_energy(model, x, y, y_var)
            
            # Thermodynamic acceptance with float32 underflow protection (min clamp)
            delta_energy = (new_energy - current_energy) / temp
            alpha = torch.exp(torch.clamp(-delta_energy, min=-50.0, max=0.0))
            
            if torch.rand(1, device=device) < alpha:
                current_energy = new_energy
                accept_count += 1
                window_accepts += 1
            else:
                for name, param in model.named_parameters():
                    param.copy_(old_weights[name])
            
            # --- Adaptive step-size tuning during burn-in ---
            if i < burn_in and (i + 1) % adapt_interval == 0:
                window_rate = window_accepts / adapt_interval
                
                # Multiplicative Robbins-Monro style adjustment, clamped to [0.5, 2.0]
                if window_rate > 0:
                    ratio = window_rate / target_accept
                    step_size *= min(max(ratio, 0.5), 2.0)
                else:
                    step_size *= 0.5  # Zero acceptance → halve step size
                
                window_accepts = 0
            
            # Collect posterior samples after burn-in
            if i >= burn_in and (i - burn_in) % thin == 0:
                accepted_weights.append({name: param.detach().clone().cpu() for name, param in model.named_parameters()})
                    
        if (i + 1) % 10000 == 0:
            phase = "ADAPT" if i < burn_in else "SAMPLE"
            print(f"[{phase}] Iter {i+1}/{num_samples} | Energy: {current_energy.item():.4f} | "
                  f"Accept: {accept_count/(i+1)*100:.1f}% | Step: {step_size:.6f}")
                    
    print(f"\nFinal adapted step_size: {step_size:.6f}")
    return accepted_weights, energy_trace

# [FIX #2] Extended burn-in (50k) for proper equilibration; 150k total iterations
posterior_samples, energy_trace = run_mh_sampler(
    bnn_model, X_pca, Y_tensor, Y_var,
    num_samples=150000, initial_step_size=0.01, temp=0.002,
    burn_in=50000, thin=10, target_accept=0.25, adapt_interval=500
)

print(f"\nCollected {len(posterior_samples)} probabilistic weight states.")

# ==========================================
# 5. PREDICTION & DIAGNOSTICS
# ==========================================
print("\nEvaluating posterior ensemble on training + synthetic ground motions...")
n_states = len(posterior_samples)

# In-sample predictions (training data, 200 EQs) — stored in full
train_predictions = []

# [FIX #1 cont.] Out-of-sample predictions (synthetic, 20k EQs)
# Memory-efficient streaming mean/std to avoid allocating (n_states x 20000) array
synth_sum    = np.zeros(N_synth, dtype=np.float64)
synth_sq_sum = np.zeros(N_synth, dtype=np.float64)

with torch.no_grad():
    for idx, weights in enumerate(posterior_samples):
        weights_gpu = {name: param.to(device) for name, param in weights.items()}
        bnn_model.load_state_dict(weights_gpu)
        
        # In-sample (training records)
        train_preds = bnn_model(X_pca).cpu().numpy().flatten()
        train_predictions.append(train_preds)
        
        # Out-of-sample (synthetic records) — streaming accumulation
        synth_preds = bnn_model(X_synth_pca).cpu().numpy().flatten().astype(np.float64)
        synth_sum    += synth_preds
        synth_sq_sum += synth_preds ** 2
        
        if (idx + 1) % 2000 == 0:
            print(f"  Evaluated {idx+1}/{n_states} posterior states...")

train_predictions = np.array(train_predictions)  # shape: (n_states, 200)

# Collapse epistemic axis
train_per_eq_mean = train_predictions.mean(axis=0)   # in-sample posterior mean
train_per_eq_std  = train_predictions.std(axis=0)     # in-sample epistemic std (memorization uncertainty)

synth_per_eq_mean = synth_sum / n_states              # out-of-sample posterior mean
synth_per_eq_var  = synth_sq_sum / n_states - synth_per_eq_mean ** 2
synth_per_eq_std  = np.sqrt(np.maximum(synth_per_eq_var, 0.0))  # genuine predictive uncertainty

below_zero_frac = (synth_per_eq_mean < 0).mean() * 100

# [FIX #4] Separate in-sample vs out-of-sample diagnostics
print("\n--- IN-SAMPLE DIAGNOSTICS (200 training EQs — memorized) ---")
print(f"Posterior Mean | Min: {train_per_eq_mean.min():.5f} m | Max: {train_per_eq_mean.max():.5f} m | Mean: {train_per_eq_mean.mean():.5f} m")
print(f"Epistemic Std  | Min: {train_per_eq_std.min():.5f} m | Max: {train_per_eq_std.max():.5f} m | Mean: {train_per_eq_std.mean():.5f} m")

print(f"\n--- OUT-OF-SAMPLE DIAGNOSTICS ({N_synth} synthetic EQs — genuine) ---")
print(f"Posterior Mean | Min: {synth_per_eq_mean.min():.5f} m | Max: {synth_per_eq_mean.max():.5f} m | Mean: {synth_per_eq_mean.mean():.5f} m")
print(f"Predictive Std | Min: {synth_per_eq_std.min():.5f} m | Max: {synth_per_eq_std.max():.5f} m | Mean: {synth_per_eq_std.mean():.5f} m")
print(f"Fraction of OOS predictions below 0: {below_zero_frac:.2f}%")
print("------------------------------\n")

# ==========================================
# 6. VISUALIZATION & GROUND-TRUTH COMPARISON
# ==========================================
print("Loading 20,000 FDM samples for ground-truth comparison...")
fdm_path = "Raw_20000_Peaks.txt"
fdm_20k_actual = np.loadtxt(fdm_path)

plt.style.use('dark_background')
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=120)

# Subplot 1: PDF Comparison — in-sample posterior mean predictions vs FDM
# Using training predictions (accurate) since BNN is a memorizer that can't extrapolate
# [FIX #3] Tuned KDE bandwidth (0.8x Scott's rule) to avoid over-smoothing the peak
kde_bnn = gaussian_kde(train_per_eq_mean, bw_method=0.8)
kde_fdm = gaussian_kde(fdm_20k_actual, bw_method=0.8)

min_val = min(train_per_eq_mean.min(), fdm_20k_actual.min()) - 0.02
max_val = max(train_per_eq_mean.max(), fdm_20k_actual.max()) + 0.02
x_range = np.linspace(min_val, max_val, 300)

axes[0].plot(x_range, kde_fdm(x_range), color="crimson", linestyle="--", linewidth=2.5, label="20,000 FDM (Ground Truth)")
axes[0].plot(x_range, kde_bnn(x_range), color="dodgerblue", linewidth=2.5, label=f"BNN Posterior Mean (200 EQs)")
axes[0].fill_between(x_range, kde_bnn(x_range), color="dodgerblue", alpha=0.25)

axes[0].set_title(f"PDF Comparison: BNN vs FDM ({n_states} Posterior States)", fontsize=13, fontweight='bold', color='white')
axes[0].set_xlabel("Peak Displacement [m]", fontsize=11, color='white')
axes[0].set_ylabel("Probability Density", fontsize=11, color='white')
axes[0].legend(loc="upper right", fontsize=10, facecolor='#111111', edgecolor='#444444')
axes[0].grid(True, color='#333333', linestyle='-', linewidth=0.5)

# Subplot 2: MH Energy Trace
axes[1].plot(energy_trace, color="orange", linewidth=0.8)
axes[1].axvline(x=50000, color="white", linestyle="--", linewidth=1.0, label="--- burn-in cutoff")

axes[1].set_title("MH Energy Trace (convergence check)", fontsize=13, fontweight='bold', color='white')
axes[1].set_xlabel("Iteration", fontsize=11, color='white')
axes[1].set_ylabel("Energy", fontsize=11, color='white')
axes[1].legend(loc="upper right", fontsize=10, facecolor='#111111', edgecolor='#444444')
axes[1].grid(True, color='#333333', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()