import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde  # <--- Replaced finite difference with KDE

# ==========================================
# 0. CONFIGURATION & HARDWARE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Igniting Physics-Constrained MH Engine on: {device.type.upper()}")

# --- THE PORTABLE PATH FIX ---
# 1. Get the exact folder path where this script is currently located
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Look for the text file inside that exact same folder
file_path = os.path.join(script_dir, "peak_response_base isolation_building.txt")

# ==========================================
# 1. LOAD & PREPARE KDE TARGET DATA
# ==========================================
field_data = np.loadtxt(file_path)

# Extract Top Floor (index 2). Change to 5 for Ground Floor!
target_disp = field_data[:, 5]
N_TOTAL = len(target_disp)

# Sort displacements (X)
X_sorted = np.sort(target_disp)

# --- NEW: Professor's requested ksdensity method ---
# Fit the Gaussian KDE to the raw data
kde = gaussian_kde(target_disp)

# Evaluate the KDE at our sorted X points to get the Y target probabilities
Y_prob_density = kde(X_sorted)

# Convert to Tensors
X_tensor = torch.tensor(X_sorted, dtype=torch.float32).view(-1, 1).to(device)
Y_tensor = torch.tensor(Y_prob_density, dtype=torch.float32).view(-1, 1).to(device)

# ==========================================
# 2. TRAIN-TEST SPLIT & SCALING (ANCHOR METHOD)
# ==========================================
N_TRAIN = 150
torch.manual_seed(42) # Still use a seed for reproducible randomness in the middle

# Create a sequence of indices [0, 1, 2, ..., 796]
all_indices = torch.arange(N_TOTAL, device=device)

# 1. THE FIX: Anchor the tails! Grab the 5 smallest and 5 largest points
n_anchors = 5
anchor_idx = torch.cat([all_indices[:n_anchors], all_indices[-n_anchors:]])

# 2. Randomly sample the remaining 140 points from the middle of the pack
middle_indices = all_indices[n_anchors:-n_anchors]
perm = torch.randperm(len(middle_indices), device=device)
random_train_idx = middle_indices[perm[:N_TRAIN - (2 * n_anchors)]]

# 3. Combine them to form the 150 training points
train_idx = torch.cat([anchor_idx, random_train_idx])

# 4. The test set is everything else left in the middle
test_idx = middle_indices[perm[N_TRAIN - (2 * n_anchors):]]

# Apply the split
X_train_raw = X_tensor[train_idx]
Y_train_raw = Y_tensor[train_idx].flatten()

X_test_raw = X_tensor[test_idx]
Y_test_raw = Y_tensor[test_idx].flatten()

# Standardize X Inputs
X_mean = X_train_raw.mean()
X_std = X_train_raw.std()

X_train = (X_train_raw - X_mean) / X_std
X_test = (X_test_raw - X_mean) / X_std

# Min-Max Scale Y Targets (The "Flatline" Fix)
Y_max = Y_train_raw.max()
Y_train_scaled = Y_train_raw / Y_max
Y_var_scaled = Y_train_scaled.var()

print(f"BNN Input Dim: 1")
print(f"Training on {N_TRAIN} samples (including {n_anchors * 2} anchor points).")
print(f"Testing on {N_TOTAL - N_TRAIN} samples.")

# ==========================================
# 3. 1D BNN ARCHITECTURE (UPGRADED)
# ==========================================
class PDF_BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)  # <-- Added a second hidden layer (the extra joint)
        self.activation = nn.Mish()
        self.fc3 = nn.Linear(20, 1)   # <-- Output layer

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return torch.relu(self.fc3(x)) # ReLU keeps it exactly at 0 where needed

bnn_model = PDF_BNN().to(device)
total_params = sum(p.numel() for p in bnn_model.parameters())
print(f"Network parameter count: {total_params}")

def calculate_gibbs_energy(model, x, y_target, y_var):
    predictions = model(x).flatten()
    data_error = torch.mean((predictions - y_target) ** 2) / (2.0 * y_var)
    # Loosen the penalty to 0.001 to allow the extra layer to bend
    weight_penalty = 0.001 * sum(torch.sum(p ** 2) for p in model.parameters()) 
    return data_error + weight_penalty

# ==========================================
# 4. MAP PRETRAINING
# ==========================================
print("\n--- Pretraining to MAP estimate via Adam ---")
num_pretrain_epochs = 10000
optimizer = torch.optim.Adam(bnn_model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_pretrain_epochs, eta_min=1e-4)

best_energy = float('inf')
best_map_state = None

for epoch in range(num_pretrain_epochs + 1):
    optimizer.zero_grad()
    # Train against the SCALED targets
    energy = calculate_gibbs_energy(bnn_model, X_train, Y_train_scaled, Y_var_scaled)
    energy.backward()
    optimizer.step()
    scheduler.step()

    if energy.item() < best_energy:
        best_energy = energy.item()
        best_map_state = {name: param.detach().clone() for name, param in bnn_model.named_parameters()}

    if epoch % 2000 == 0:
        print(f"  Pretrain Ep {epoch:05d} | Energy: {energy.item():.4f}")

bnn_model.load_state_dict(best_map_state)
print(f"Pretraining complete. Best MAP energy: {best_energy:.4f}\n")

# ==========================================
# 5. MCMC SAMPLER
# ==========================================
def run_mh_sampler(model, x, y, y_var, num_samples=50000, initial_step_size=0.01,
                   temp=0.002, burn_in=10000, thin=10, adapt_interval=500):

    print(f"Running Adaptive MH Sampler for {num_samples} iterations...")
    step_size = initial_step_size
    with torch.no_grad():
        current_energy = calculate_gibbs_energy(model, x, y, y_var)

    accepted_weights = []
    accept_count = 0
    window_accepts = 0

    for i in range(num_samples):
        with torch.no_grad():
            old_weights = {name: param.detach().clone() for name, param in model.named_parameters()}

            for param in model.parameters():
                param.add_(torch.randn_like(param) * step_size)

            new_energy = calculate_gibbs_energy(model, x, y, y_var)
            delta_energy = (new_energy - current_energy) / temp
            alpha = torch.exp(torch.clamp(-delta_energy, min=-50.0, max=0.0))

            if torch.rand(1, device=device) < alpha:
                current_energy = new_energy
                accept_count += 1
                window_accepts += 1
            else:
                bnn_model.load_state_dict(old_weights)

            if i < burn_in and (i + 1) % adapt_interval == 0:
                window_rate = window_accepts / adapt_interval
                ratio = window_rate / 0.25 if window_rate > 0 else 0.5
                step_size *= min(max(ratio, 0.8), 1.25)
                step_size = min(step_size, 0.05)
                window_accepts = 0

            if i >= burn_in and (i - burn_in) % thin == 0:
                accepted_weights.append({name: param.detach().clone().cpu() for name, param in model.named_parameters()})

        if (i + 1) % 10000 == 0:
            print(f"Iter {i+1}/{num_samples} | Energy: {current_energy.item():.4f} | Accept: {accept_count/(i+1)*100:.1f}%")

    return accepted_weights

# Sample against the SCALED targets
posterior_samples = run_mh_sampler(bnn_model, X_train, Y_train_scaled, Y_var_scaled, num_samples=50000, burn_in=10000)

# ==========================================
# 6. EVALUATION & PLOTTING (CONTINUOUS DOMAIN)
# ==========================================
print("\nEvaluating posterior ensemble on Continuous Domain...")

# 1. Create a smooth grid spanning from min X to a little past max X
X_plot_raw = torch.linspace(X_tensor.min(), X_tensor.max() * 1.02, 500).view(-1, 1).to(device)
X_plot_scaled = (X_plot_raw - X_mean) / X_std

test_predictions = []

with torch.no_grad():
    for weights in posterior_samples:
        bnn_model.load_state_dict({name: param.to(device) for name, param in weights.items()})
        
        # Predict on the smooth grid
        preds_scaled = bnn_model(X_plot_scaled).cpu().numpy().flatten()
        
        # Un-scale to physical probability density values
        preds_physical = preds_scaled * Y_max.item()
        test_predictions.append(preds_physical)

test_predictions = np.array(test_predictions)
bnn_mean_pdf = test_predictions.mean(axis=0)
bnn_std_pdf = test_predictions.std(axis=0)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

# Plot KDE Target Points (MCS Truth) - We plot the full dataset here to show the complete baseline
ax.scatter(X_tensor.cpu(), Y_tensor.cpu(), color='dodgerblue', alpha=0.4, s=30, label='Full MCS Points (KDE Ground Truth)')

# Plot BNN Smooth Continuous Prediction
ax.plot(X_plot_raw.cpu(), bnn_mean_pdf, color='crimson', linewidth=3, label='BNN Predicted PDF (Mean)')
ax.fill_between(X_plot_raw.cpu().flatten(), 
                bnn_mean_pdf - 2*bnn_std_pdf, 
                bnn_mean_pdf + 2*bnn_std_pdf, 
                color='crimson', alpha=0.3, label='BNN Epistemic Uncertainty (±2σ)')

ax.set_title("Data-Driven BNN: Predicted PDF vs MCS KDE Ground Truth (Bottom Floor)", fontsize=14, fontweight='bold')
ax.set_xlabel("Peak Displacement [m]", fontsize=12)
ax.set_ylabel("Probability Density", fontsize=12)
ax.legend(fontsize=11, facecolor='#111111', edgecolor='#444444')
ax.grid(color='#333333', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()