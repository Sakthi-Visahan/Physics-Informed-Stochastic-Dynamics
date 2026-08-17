import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# ==========================================
# 0. HARDWARE ACCELERATION (CUDA)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Igniting Physics-Constrained MH Engine on: {device.type.upper()}")

# ==========================================
# 1. DATA PIPELINE & PHYSICAL SCALING
# ==========================================
eq_path = r"C:\Users\Sakthi Visahan V\Downloads\EQ_SDOF\EQ\EQ_Motion.mat"
peak_path = r"C:\Users\Sakthi Visahan V\Downloads\EQ_SDOF\EQ\Peak_Response.mat"

eq_data = sio.loadmat(eq_path)
peak_data = sio.loadmat(peak_path)

X_tensor = torch.tensor(eq_data['x_g'].T, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(peak_data['y_peak'].T, dtype=torch.float32).to(device)

print(f"Loaded {X_tensor.shape[0]} total earthquake records.")

# --- DYNAMIC TRAINING SUBSET ---
# Safely handles the limit depending on whether the MATLAB script generated 200 or 300+ samples
N_TRAIN = min(300, X_tensor.shape[0]) 
torch.manual_seed(42)  # For reproducibility
perm = torch.randperm(X_tensor.shape[0], device=device)
idx_train = perm[:N_TRAIN]

X_train = X_tensor[idx_train]
Y_train = Y_tensor[idx_train].flatten()

print(f"PI-BNN Training on subset of {N_TRAIN} EQs.")

# [PHYSICS FIX] Extract Peak Ground Acceleration (PGA) BEFORE scaling/PCA
PGA_train = torch.max(torch.abs(X_train), dim=1).values

# Inputs standard-scaled using only the training subset
X_mean, X_std = X_train.mean(), X_train.std()
X_scaled = (X_train - X_mean) / X_std

# PCA dimensionality reduction (Components cannot exceed sample size)
n_components = min(120, N_TRAIN) 
U, S, V = torch.pca_lowrank(X_scaled, q=n_components)
X_train_pca = torch.mm(X_scaled - X_scaled.mean(dim=0), V)

total_var = ((X_scaled - X_scaled.mean(dim=0)) ** 2).sum()
explained_var = (S ** 2).sum()
print(f"PCA: {X_scaled.shape[1]} dims -> {n_components} components (variance retained: {explained_var/total_var*100:.1f}%)")

# Generate synthetic ground motions via bootstrap + Gaussian smoothing in PCA space
N_synth = 20000
bandwidth_factor = 0.1  # 10% of per-component std — conservative to prevent extrapolation

pca_std = X_train_pca.std(dim=0)  

# Bootstrap from the training PCA vectors
indices = torch.randint(0, X_train_pca.shape[0], (N_synth,), device=device)
X_synth_pca = X_train_pca[indices].clone()

# Gaussian smoothing
X_synth_pca += bandwidth_factor * pca_std * torch.randn_like(X_synth_pca)

print(f"Generated {N_synth} synthetic ground motions (bootstrap + {bandwidth_factor:.0%} Gaussian smoothing)")

Y_var = Y_train.var()

print("\n--- DATA DIAGNOSTICS ---")
print(f"True Y (Peak) Min: {Y_train.min().item():.5f} m")
print(f"True Y (Peak) Max: {Y_train.max().item():.5f} m")
print(f"True Y (Peak) Mean: {Y_train.mean().item():.5f} m")
print("------------------------\n")

# ==========================================
# 2. PHYSICS-CONSTRAINED BNN ARCHITECTURE
# ==========================================
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

bnn_model = PhysicsConstrainedBNN(input_dim=X_train_pca.shape[1]).to(device)
total_params = sum(p.numel() for p in bnn_model.parameters())
print(f"Network parameter count: {total_params} (fc1: {n_components}x20 + bias, fc2: 20x1 + bias)")

# ==========================================
# 3. THERMODYNAMICS (Physics-Informed Gibbs Energy)
# ==========================================
def calculate_gibbs_energy(model, x, y_target, y_var, pga_values):
    predictions = model(x).flatten()
    
    # 1. Data Error (Statistical Fit)
    data_error = torch.mean((predictions - y_target) ** 2) / (2.0 * y_var)
    
    # 2. Gaussian Prior (Weight Penalty)
    weight_penalty = 0.001 * sum(torch.sum(p ** 2) for p in model.parameters())
    
    # 3. PHYSICS CONSTRAINT: Resonance Amplification Bound
    # M = 189.4 kg, K = 26.2E3 N/m, xi = 0.0035
    w_n_sq = 26200.0 / 189.4
    max_amplification = 1.0 / (2.0 * 0.0035) 
    
    # Physical limit: Sd <= (PGA / w_n^2) * max_amplification
    theoretical_max_sd = (pga_values / w_n_sq) * max_amplification
    
    # Calculate how much the predictions exceed the physical limit
    physics_violation = torch.relu(predictions - theoretical_max_sd)
    
    # Square the violations and scale them heavily to force MCMC rejection
    physics_penalty = 1000.0 * torch.mean(physics_violation ** 2)
    
    return data_error + weight_penalty + physics_penalty

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
    # Pass PGA_train to evaluate the physics boundary during pretraining
    energy = calculate_gibbs_energy(bnn_model, X_train_pca, Y_train, Y_var, PGA_train)
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
def run_mh_sampler(model, x, y, y_var, pga, num_samples=100000, initial_step_size=0.01,
                   temp=0.002, burn_in=20000, thin=10,
                   target_accept=0.25, adapt_interval=500):
    
    print(f"Running Adaptive MH Sampler for {num_samples} iterations...")
    print(f"Target Accept: {target_accept*100:.0f}% | Temp = {temp} | Initial Step = {initial_step_size}")
    print(f"Adaptive tuning active during burn-in (first {burn_in} iters, window = {adapt_interval})\n")
    
    step_size = initial_step_size
    
    with torch.no_grad():
        current_energy = calculate_gibbs_energy(model, x, y, y_var, pga)
    
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
                    
            new_energy = calculate_gibbs_energy(model, x, y, y_var, pga)
            
            delta_energy = (new_energy - current_energy) / temp
            alpha = torch.exp(torch.clamp(-delta_energy, min=-50.0, max=0.0))
            
            if torch.rand(1, device=device) < alpha:
                current_energy = new_energy
                accept_count += 1
                window_accepts += 1
            else:
                for name, param in model.named_parameters():
                    param.copy_(old_weights[name])
            
            if i < burn_in and (i + 1) % adapt_interval == 0:
                window_rate = window_accepts / adapt_interval
                
                if window_rate > 0:
                    ratio = window_rate / target_accept
                    step_size *= min(max(ratio, 0.5), 2.0)
                else:
                    step_size *= 0.5  
                
                window_accepts = 0
            
            if i >= burn_in and (i - burn_in) % thin == 0:
                accepted_weights.append({name: param.detach().clone().cpu() for name, param in model.named_parameters()})
                    
        if (i + 1) % 10000 == 0:
            phase = "ADAPT" if i < burn_in else "SAMPLE"
            print(f"[{phase}] Iter {i+1}/{num_samples} | Energy: {current_energy.item():.4f} | "
                  f"Accept: {accept_count/(i+1)*100:.1f}% | Step: {step_size:.6f}")
                    
    print(f"\nFinal adapted step_size: {step_size:.6f}")
    return accepted_weights, energy_trace

# Run the MH Sampler with the PGA parameter passed in
posterior_samples, energy_trace = run_mh_sampler(
    bnn_model, X_train_pca, Y_train, Y_var, PGA_train,
    num_samples=150000, initial_step_size=0.01, temp=0.002,
    burn_in=50000, thin=10, target_accept=0.25, adapt_interval=500
)

print(f"\nCollected {len(posterior_samples)} probabilistic weight states.")

# ==========================================
# 5. PREDICTION & DIAGNOSTICS
# ==========================================
print(f"\nEvaluating posterior ensemble on {N_TRAIN} training + {N_synth} synthetic ground motions...")
n_states = len(posterior_samples)

train_predictions = []

synth_sum    = np.zeros(N_synth, dtype=np.float64)
synth_sq_sum = np.zeros(N_synth, dtype=np.float64)

with torch.no_grad():
    for idx, weights in enumerate(posterior_samples):
        weights_gpu = {name: param.to(device) for name, param in weights.items()}
        bnn_model.load_state_dict(weights_gpu)
        
        train_preds = bnn_model(X_train_pca).cpu().numpy().flatten()
        train_predictions.append(train_preds)
        
        synth_preds = bnn_model(X_synth_pca).cpu().numpy().flatten().astype(np.float64)
        synth_sum    += synth_preds
        synth_sq_sum += synth_preds ** 2
        
        if (idx + 1) % 2000 == 0:
            print(f"  Evaluated {idx+1}/{n_states} posterior states...")

train_predictions = np.array(train_predictions)  # shape: (n_states, N_TRAIN)

train_per_eq_mean = train_predictions.mean(axis=0)   
train_per_eq_std  = train_predictions.std(axis=0)     

synth_per_eq_mean = synth_sum / n_states              
synth_per_eq_var  = synth_sq_sum / n_states - synth_per_eq_mean ** 2
synth_per_eq_std  = np.sqrt(np.maximum(synth_per_eq_var, 0.0))  

below_zero_frac = (synth_per_eq_mean < 0).mean() * 100

print(f"\n--- IN-SAMPLE DIAGNOSTICS ({N_TRAIN} training EQs — memorized) ---")
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
fdm_path = r"C:\Users\Sakthi Visahan V\OneDrive\Documents\spyder_stuff\quakity quakity\EQ_SDOF\EQ\Raw_20000_Peaks.txt"
fdm_20k_actual = np.loadtxt(fdm_path)

plt.style.use('dark_background')
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=120)

# Subplot 1: PDF Comparison
# [FIXED] Pointed the KDE correctly to synth_per_eq_mean (20,000 pts) instead of train_per_eq_mean
kde_bnn = gaussian_kde(synth_per_eq_mean, bw_method=0.8)
kde_fdm = gaussian_kde(fdm_20k_actual, bw_method=0.8)

min_val = min(synth_per_eq_mean.min(), fdm_20k_actual.min()) - 0.02
max_val = max(synth_per_eq_mean.max(), fdm_20k_actual.max()) + 0.02
x_range = np.linspace(min_val, max_val, 300)

axes[0].plot(x_range, kde_fdm(x_range), color="crimson", linestyle="--", linewidth=2.5, label="20,000 FDM (Ground Truth)")
axes[0].plot(x_range, kde_bnn(x_range), color="dodgerblue", linewidth=2.5, label=f"PI-BNN Surrogate ({N_synth} Synthetic EQs)")
axes[0].fill_between(x_range, kde_bnn(x_range), color="dodgerblue", alpha=0.25)

axes[0].set_title(f"PDF Comparison: PI-BNN vs FDM ({n_states} Posterior States)", fontsize=13, fontweight='bold', color='white')
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