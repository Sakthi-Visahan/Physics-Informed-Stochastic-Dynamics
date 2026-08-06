import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# ==========================================
# 1. HARDWARE & DATA LOADING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=====================================")
print(f"Using Hardware Device: {device.type.upper()}")
print(f"=====================================")

# --- THE UNIVERSE FREEZE (Locking RNG) ---
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
base_path = r"C:\Users\Sakthi Visahan V\OneDrive\Documents\spyder_stuff\quakity quakity\EQ_SDOF\EQ"

print("Loading MATLAB Text Files...")
gt_data = np.loadtxt(os.path.join(base_path, 'PDF_Ground_Truth.txt'), delimiter=',') 
x_gt = gt_data[:, 0]
p_gt = gt_data[:, 1]

raw_peaks = np.loadtxt(os.path.join(base_path, 'Raw_20000_Peaks.txt'), delimiter=',')

np.random.shuffle(raw_peaks)

# ==========================================
# 2. NETWORK ARCHITECTURE (STABILIZED FF-PINN)
# ==========================================
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class FourierFeatureMapping(nn.Module):
    def __init__(self, in_features, mapping_size, scale=0.25): 
        super().__init__()
        # Scale reduced to 0.25 to prevent gradient explosions in the PDE loss
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)
        
    def forward(self, x):
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FF_PDEM_PINN(nn.Module):
    def __init__(self, mapping_size=128):
        super().__init__()
        # ONLY map the 2 raw coordinates: y and tau
        self.ff_layer = FourierFeatureMapping(in_features=2, mapping_size=mapping_size, scale=0.25)
        
        self.net = nn.Sequential(
            # 256 (from FF layer) + 3 (bypassed inputs: x_max, sin, cos) = 259
            nn.Linear(2 * mapping_size + 3, 256), Sine(), 
            nn.Linear(256, 256), Sine(),
            nn.Linear(256, 256), Sine(),
            nn.Linear(256, 256), Sine(),
            nn.Linear(256, 1) 
        )

    def forward(self, y, tau, x_max):
        # 1. Precompute the physical frequencies
        sin_feat = torch.sin((5.0 * math.pi / 2.0) * tau)
        cos_feat = torch.cos((5.0 * math.pi / 2.0) * tau)
        
        # 2. Pass ONLY y and tau through the Fourier Mapping
        coords = torch.cat([y, tau], dim=1)
        ff_coords = self.ff_layer(coords)
        
        # 3. Concatenate the high-def coordinates with the bypassed physics parameters
        inputs = torch.cat([ff_coords, x_max, sin_feat, cos_feat], dim=1)
        
        raw_p = self.net(inputs)
        
        # Restored your squaring function to allow true zeroes at the boundaries!
        return raw_p**2

# ==========================================
# 3. PHYSICS LOSS FUNCTIONS (gPINN Upgrade)
# ==========================================
def standard_phy_loss(model, y, tau, x_max):
    y = y.clone().detach().requires_grad_(True)
    tau = tau.clone().detach().requires_grad_(True)
    
    # Forward pass
    p = model(y, tau, x_max)
    
    # --- 1st Order Derivatives (Standard PINN) ---
    dp_dtau = torch.autograd.grad(p, tau, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    dp_dy = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    
    # Calculate the physical velocity and the main PDE Residual (R)
    y_dot = x_max * (5.0 * math.pi / 2.0) * torch.cos((5.0 * math.pi / 2.0) * tau)
    residual = dp_dtau + (y_dot * dp_dy)
    
    # --- The gPINN Mechanism (Gradient of the Residual) ---
    # We take the derivative of the residual itself!
    dR_dy = torch.autograd.grad(residual, y, grad_outputs=torch.ones_like(residual), create_graph=True)[0]
    dR_dtau = torch.autograd.grad(residual, tau, grad_outputs=torch.ones_like(residual), create_graph=True)[0]
    
    # Standard residual loss
    loss_R = torch.mean(residual**2)
    
    # Gradient penalty loss (forces the error landscape to be flat)
    loss_grad_R = torch.mean(dR_dy**2) + torch.mean(dR_dtau**2)
    
    # Weighting factor for the gradient penalty (Usually 0.01 to 0.1)
    lambda_g = 0.0001 
    
    return loss_R + (lambda_g * loss_grad_R)

def bc_loss(model, tau_bc, x_max_bc):
    y_left = torch.full_like(tau_bc, -0.3)
    y_right = torch.full_like(tau_bc, 0.3)
    p_left = model(y_left, tau_bc, x_max_bc)
    p_right = model(y_right, tau_bc, x_max_bc)
    target = torch.zeros_like(p_left)
    return torch.mean((p_left - target)**2) + torch.mean((p_right - target)**2)

def integral_loss(model, tau_int, x_max_int):
    n_points = 200
    y_min, y_max = -0.3, 0.3
    dy = (y_max - y_min) / (n_points - 1)
    
    y_grid = torch.linspace(y_min, y_max, n_points, device=device).unsqueeze(1)
    batch_size = tau_int.shape[0]
    
    y_eval = y_grid.repeat(batch_size, 1) 
    tau_eval = tau_int.repeat_interleave(n_points, dim=0) 
    x_max_eval = x_max_int.repeat_interleave(n_points, dim=0)
    
    p_pred = model(y_eval, tau_eval, x_max_eval)
    p_pred = p_pred.view(batch_size, n_points)
    
    calculated_area = torch.sum(p_pred, dim=1) * dy
    target_area = torch.ones_like(calculated_area) 
    return torch.mean((calculated_area - target_area)**2)

def ic_loss(model, y_ic, x_max_ic):
    tau_zero = torch.zeros_like(y_ic)
    p_pred = model(y_ic, tau_zero, x_max_ic)
    sigma = 0.0035
    p_exact = (1.0 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * (y_ic / sigma)**2)
    return torch.mean((p_pred - p_exact)**2)

# ==========================================
# 4. TWO-STAGE TRAINING LOOP (UNIVERSAL CONTINUOUS)
# ==========================================
epochs_adam = 45000 
batch_size = 6000 # Max safe limit for RTX 2050 VRAM

model = FF_PDEM_PINN(mapping_size=128).to(device)
optimizer_adam = torch.optim.Adam(model.parameters(), lr=2e-3) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=epochs_adam, eta_min=1e-5)

print(f"Starting Phase 1: Universal Adam Optimization ({epochs_adam} Epochs)...")
start_time = time.time()

for epoch in range(epochs_adam):
    optimizer_adam.zero_grad()
    
    tau_col = torch.rand(batch_size, 1, device=device)
    # CONTINUOUS SAMPLE INTENSITY
    x_max_col = torch.rand(batch_size, 1, device=device) * 0.3
    # PURE UNIFORM SPATIAL SAMPLING (The Step Killer)
    y_col = (torch.rand(batch_size, 1, device=device) * 0.6) - 0.3 
    
    tau_bc = torch.rand(batch_size, 1, device=device)
    x_max_bc = torch.rand(batch_size, 1, device=device) * 0.3

    tau_int = torch.rand(50, 1, device=device)
    x_max_int = torch.rand(50, 1, device=device) * 0.3
    
    y_ic = (torch.rand(batch_size, 1, device=device) * 0.6) - 0.3
    x_max_ic = torch.rand(batch_size, 1, device=device) * 0.3

    loss_p = standard_phy_loss(model, y_col, tau_col, x_max_col)
    loss_b = bc_loss(model, tau_bc, x_max_bc)
    loss_int = integral_loss(model, tau_int, x_max_int) 
    loss_i = ic_loss(model, y_ic, x_max_ic)
    
    # BASE PINN STRATEGIC WEIGHT BALANCING: PDE=75, IC=80, BC=50, Int=20
    loss = (75.0 * loss_p) + (100.0 * loss_i) + (50.0 * loss_b) + (20.0 * loss_int)
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer_adam.step()
    scheduler.step()
    
    if epoch % 1500 == 0:
        elapsed = time.time() - start_time
        print(f"Adam Ep {epoch:04d} | Tot: {loss.item():.3f} | PDE: {loss_p.item():.3f} | IC: {loss_i.item():.3f} | Int: {loss_int.item():.3f}")
        start_time = time.time()

# --- PHASE 2: L-BFGS ---
print(f"Starting Phase 2: Gentle L-BFGS Polish on Smooth Continuous Landscape...")

lbfgs_points = 6000 # Max matching safe resolution

tau_col_s = torch.rand(lbfgs_points, 1, device=device)
x_max_col_s = torch.rand(lbfgs_points, 1, device=device) * 0.3
y_col_s = (torch.rand(lbfgs_points, 1, device=device) * 0.6) - 0.3 

tau_bc_s = torch.rand(lbfgs_points, 1, device=device)
x_max_bc_s = torch.rand(lbfgs_points, 1, device=device) * 0.3

tau_int_s = torch.rand(50, 1, device=device)
x_max_int_s = torch.rand(50, 1, device=device) * 0.3

y_ic_s = (torch.rand(lbfgs_points, 1, device=device) * 0.6) - 0.3
x_max_ic_s = torch.rand(lbfgs_points, 1, device=device) * 0.3

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    lr=1.0, 
    max_iter=200,   # Increased from 50 (Let it run deeper!)
    max_eval=250,   # Increased from 60
    history_size=50,
    line_search_fn="strong_wolfe"
)
epochs_lbfgs = 20   # We can lower the outer epochs because max_iter is doing the heavy lifting now

start_time = time.time()

for epoch in range(epochs_lbfgs):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss_p_s = standard_phy_loss(model, y_col_s, tau_col_s, x_max_col_s)
        loss_b_s = bc_loss(model, tau_bc_s, x_max_bc_s)
        loss_int_s = integral_loss(model, tau_int_s, x_max_int_s) 
        loss_i_s = ic_loss(model, y_ic_s, x_max_ic_s)
        
        loss_s = (75.0 * loss_p_s) + (100.0 * loss_i_s) + (50.0 * loss_b_s) + (20.0 * loss_int_s)
        loss_s.backward()
        return loss_s
        
    optimizer_lbfgs.step(closure)
    
    current_loss = closure().item()
    elapsed = time.time() - start_time
    print(f"L-BFGS Ep {epoch:04d} | Tot Loss: {current_loss:.4f} | Time: {elapsed:.2f}s")
    start_time = time.time()

print("Two-Stage RE-PINN Training Complete.")

# ==========================================
# 5. PLOTTING THE SHOWDOWN (Discrete Summation / Mini-MCS)
# ==========================================
print("Evaluating Universal gPINN via Discrete Sample Summation...")
import numpy as np

model.eval()
y_plot_tensor = torch.tensor(x_gt, dtype=torch.float32, device=device).unsqueeze(1)
tau_final = torch.ones_like(y_plot_tensor) 

# 1. Blindly pull our specific challenge samples
np.random.seed(0) # Or whatever seed/sample size you want to test!
blind_sample = np.random.choice(raw_peaks,400 , replace=False)

# 2. DISCRETE SUMMATION (No KDE, No Lognorm!)
p_predicted = torch.zeros_like(y_plot_tensor)

with torch.no_grad():
    for x_m in blind_sample:
        # Feed the exact discrete sample directly into the universal gPINN
        x_m_tensor = torch.full_like(y_plot_tensor, float(x_m))
        p_predicted += model(y_plot_tensor, tau_final, x_m_tensor)
    
    # Average the stacked PDFs
    p_raw = (p_predicted / len(blind_sample)).cpu().numpy().flatten()

# 3. Normalize for plotting
area = np.trapezoid(p_raw, x_gt)
if area > 0:
    p_normalized = p_raw / area
else:
    p_normalized = p_raw

# --- 4. PLOTTING ---
plt.figure(figsize=(10, 6))
plt.plot(x_gt, p_gt, 'b-', linewidth=3, label='MATLAB 20,000 MCS (Ground Truth)')
plt.plot(x_gt, p_normalized, 'r--', linewidth=2.5, label=f'gPINN Unified ({len(blind_sample)} Discrete Samples)')

plt.title(f'Universal gPINN vs. 20,000 MCS (Discrete Summation Test)', fontsize=14)
plt.xlabel('Peak Displacement [m]', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.xlim([-0.02, 0.25])
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()