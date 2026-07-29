import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.interpolate import make_interp_spline # Added for smooth peak reconstruction

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=====================================")
print(f"Using Hardware Device: {device.type.upper()}")
print(f"=====================================")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Define Base Path
base_path = r"C:\Users\Sakthi Visahan V\OneDrive\Documents\spyder_stuff\quakity quakity\EQ_SDOF\EQ"

print("Loading MATLAB Text Files...")
gt_data = np.loadtxt(os.path.join(base_path, 'PDF_Ground_Truth.txt'), delimiter=',')
x_gt = gt_data[:, 0]
p_gt = gt_data[:, 1]

raw_peaks = np.loadtxt(os.path.join(base_path, 'Raw_20000_Peaks.txt'), delimiter=',')

N_SAMPLES = 200
np.random.shuffle(raw_peaks)
x_max_samples_np = raw_peaks[:N_SAMPLES]
x_max_samples = torch.tensor(x_max_samples_np, dtype=torch.float32, device=device).unsqueeze(1)
print(f"Training FNO on exactly {len(x_max_samples)} earthquake samples!")

SIGMA_IC_END = 0.0035

def silverman_kde(samples_np, eval_points_np, bandwidth_factor=0.75):
    N = len(samples_np)
    sigma_hat = np.std(samples_np, ddof=1)
    h = bandwidth_factor * 1.06 * sigma_hat * N ** (-1 / 5)
    diffs = (eval_points_np[:, None] - samples_np[None, :]) / h
    K = np.exp(-0.5 * diffs ** 2) / np.sqrt(2 * np.pi)
    density = K.sum(axis=1) / (N * h)
    return density

# ==========================================
# 2. FOURIER NEURAL OPERATOR (FNO) ARCHITECTURE
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class PINO_Net(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=32):
        super(PINO_Net, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.p = nn.Linear(3, self.width)
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.q = nn.Linear(self.width, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.p(x)
        grid = grid.permute(0, 3, 1, 2) 
        
        x1 = self.conv0(grid)
        x2 = self.w0(grid)
        grid = F.gelu(x1 + x2)
        
        x1 = self.conv1(grid)
        x2 = self.w1(grid)
        grid = F.gelu(x1 + x2)
        
        x1 = self.conv2(grid)
        x2 = self.w2(grid)
        grid = F.gelu(x1 + x2)
        
        x1 = self.conv3(grid)
        x2 = self.w3(grid)
        grid = x1 + x2
        
        grid = grid.permute(0, 2, 3, 1) 
        grid = self.q(grid)
        grid = F.gelu(grid)
        grid = self.out(grid)
        
        return F.softplus(grid) 

# ==========================================
# 3. GLOBAL GRID GENERATOR & AUTOGRAD PHYSICS
# ==========================================
def create_fno_grid(x_max_batch, Ny=64, Ntau=64, device=device):
    B = x_max_batch.shape[0]
    y_axis = torch.linspace(-0.3, 0.3, Ny, device=device)
    tau_axis = torch.linspace(0.0, 1.0, Ntau, device=device)
    grid_y, grid_tau = torch.meshgrid(y_axis, tau_axis, indexing='ij')
    
    y_tensor = grid_y.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1).clone().requires_grad_(True)
    tau_tensor = grid_tau.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1).clone().requires_grad_(True)
    x_max_tensor = x_max_batch.view(B, 1, 1, 1).expand(B, Ny, Ntau, 1)
    
    fno_inputs = torch.cat([y_tensor, tau_tensor, x_max_tensor], dim=-1)
    return fno_inputs, y_tensor, tau_tensor, x_max_tensor

def compute_all_fno_losses(model, x_max_batch, Ny=64, Ntau=64, sigma_ic=0.0035):
    inputs, y, tau, x_max = create_fno_grid(x_max_batch, Ny, Ntau, device)
    
    p = model(inputs) 
    
    # 1. PDE Loss (Accurate Autograd)
    dp_dtau = torch.autograd.grad(p, tau, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    dp_dy = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    
    y_dot = x_max * (5.0 * math.pi / 2.0) * torch.cos((5.0 * math.pi / 2.0) * tau)
    loss_pde = torch.mean((dp_dtau + y_dot * dp_dy) ** 2)
    
    # 2. Initial Condition Loss
    p_ic = p[:, :, 0, 0]
    p_ic_target = (1.0 / (sigma_ic * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * (y[:, :, 0, 0] / sigma_ic) ** 2)
    loss_ic = torch.mean((p_ic - p_ic_target) ** 2)
    
    # 3. Boundary Condition Loss
    loss_bc = torch.mean(p[:, 0, :, 0] ** 2) + torch.mean(p[:, -1, :, 0] ** 2)
    
    # 4. Integral Loss
    dy = 0.6 / (Ny - 1) 
    area = torch.sum(p[:, :, :, 0], dim=1) * dy
    loss_int = torch.mean((area - 1.0) ** 2)
    
    return loss_pde, loss_ic, loss_bc, loss_int, p

# ==========================================
# 4. FNO-PINO TRAINING LOOP (FAST BASELINE)
# ==========================================
Ny, Ntau = 64, 64    # Fast Training Grid
epochs_adam = 15000 
batch_size = 32      

model = PINO_Net(modes1=12, modes2=12, width=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_adam, eta_min=1e-6)

print(f"Starting Fast FNO-PINO Operator Training on {Ny}x{Ntau} Grid...")
train_start = time.time() # Tracks total time
start_time = time.time()  # Tracks time per 500 epochs

y_axis_np = np.linspace(-0.3, 0.3, Ny)
p_ref_kde_uniform_np = silverman_kde(x_max_samples_np, y_axis_np, bandwidth_factor=0.85)
p_ref_kde_uniform = torch.tensor(p_ref_kde_uniform_np, dtype=torch.float32, device=device)

for epoch in range(epochs_adam):
    optimizer.zero_grad()
    
    x_max_col = torch.rand(batch_size, 1, device=device) * 0.3
    loss_pde, loss_ic, loss_bc, loss_int, _ = compute_all_fno_losses(model, x_max_col, Ny, Ntau, SIGMA_IC_END)
    
    idx = torch.randperm(len(x_max_samples))[:batch_size]
    x_max_data_batch = x_max_samples[idx]
    
    _, _, _, _, p_data_grid = compute_all_fno_losses(model, x_max_data_batch, Ny, Ntau, SIGMA_IC_END)
    p_final_tau = p_data_grid[:, :, -1, 0] 
    p_marginal = torch.mean(p_final_tau, dim=0) 
    
    area = torch.trapz(p_marginal, torch.tensor(y_axis_np, dtype=torch.float32, device=device))
    p_marginal_norm = p_marginal / (area + 1e-8)
    
    # --- THE SHOULDER-TO-PEAK FIX (WIDER GAUSSIAN MASK) ---
    y_tensor_1d = torch.tensor(y_axis_np, dtype=torch.float32, device=device)
    
    # Shift center to 0.055 and widen the radius to 0.020. 
    # This blankets the entire 0.04 to 0.08 region with high priority!
    peak_mask = 1.0 + 19.0 * torch.exp(-0.5 * ((y_tensor_1d - 0.055) / 0.020) ** 2)
    
    loss_data = torch.mean(peak_mask * (p_marginal_norm - p_ref_kde_uniform) ** 2)
    
    loss = (100.0 * loss_pde) + (100.0 * loss_ic) + (50.0 * loss_bc) + (20.0 * loss_int) + (50.0 * loss_data)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    
    if epoch % 500 == 0:
        elapsed = time.time() - start_time
        # Calculate seconds per epoch (avoid dividing by zero on the very first print)
        per_epoch = elapsed / 500 if epoch > 0 else 0.0 
        
        # Calculate remaining ETA in minutes
        remaining_epochs = epochs_adam - epoch
        eta_minutes = (remaining_epochs * per_epoch) / 60.0
        
        print(f"FNO Ep {epoch:04d} | Tot: {loss.item():.3f} | PDE: {loss_pde.item():.4f} | IC: {loss_ic.item():.4f} | Data: {loss_data.item():.4f} | {per_epoch:.3f}s/ep | ETA: {eta_minutes:.1f}m")
        start_time = time.time()
        
    # --- UPDATED STRICT EARLY STOPPING LOGIC ---
    # The network is no longer allowed to quit until the peak is fully pushed up!
    if loss_pde.item() < 0.005 and loss_ic.item() < 0.0012 and loss_data.item() < 0.002:
        print(f"\n--- Early Stopping Triggered at Epoch {epoch:04d} ---")
        print(f"Target Thresholds Reached: PDE ({loss_pde.item():.4f}) < 0.005, IC ({loss_ic.item():.4f}) < 0.0012, & Data ({loss_data.item():.4f}) < 0.002")
        break

total_train_time = time.time() - train_start
print(f"FNO-PINO Training Complete in {total_train_time/60:.1f} minutes.")

# ==========================================
# 5. SPLINE SUPER-RESOLUTION EVALUATION
# ==========================================
print("Evaluating Global Operator on Base Grid...")
model.eval()

# FIX: Evaluate on the EXACT same 64x64 grid it trained on.
# Let the SciPy Spline handle the HD smoothing for the plot!
Ny_eval, Ntau_eval = 64, 64 
y_axis_eval_np = np.linspace(-0.3, 0.3, Ny_eval)

with torch.no_grad():
    all_marginals = []
    eval_batch_size = 25 
    for i in range(0, len(x_max_samples), eval_batch_size):
        batch_samples = x_max_samples[i:i+eval_batch_size]
        
        inputs, _, _, _ = create_fno_grid(batch_samples, Ny_eval, Ntau_eval, device)
        p_eval_grid = model(inputs)
        
        all_marginals.append(p_eval_grid[:, :, -1, 0])
    
    p_eval_tau_full = torch.cat(all_marginals, dim=0)
    p_eval_marginal = torch.mean(p_eval_tau_full, dim=0).cpu().numpy()
    
    area = np.trapezoid(p_eval_marginal, y_axis_eval_np)
    p_eval_marginal = p_eval_marginal / area if area > 0 else p_eval_marginal

# ---------------------------------------------------------
# THE FIX: Spline Interpolation instead of Linear Interp
# ---------------------------------------------------------
# This forces the math to draw a natural curve through the peak 
# instead of a flat bridge.
spline = make_interp_spline(y_axis_eval_np, p_eval_marginal, k=3)
p_fno_interpolated = spline(x_gt)

# Ensure no probability drops below 0 due to curve overshoot
p_fno_interpolated = np.maximum(p_fno_interpolated, 0)

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

print(f"\n--- FNO-PINO Performance ({N_SAMPLES} Samples) ---")
print(f"RMSE vs 20,000 MCS Ground Truth: {rmse(p_fno_interpolated, p_gt):.4f}")

plt.figure(figsize=(10, 6))
plt.plot(x_gt, p_gt, 'b-', linewidth=3, label='MATLAB 20,000 MCS (Ground Truth)')
plt.plot(x_gt, p_fno_interpolated, 'm--', linewidth=2.5, label='FNO-PINO (Spectral Operator Prediction)')

plt.title(f'Fourier Neural Operator vs Ground Truth ({N_SAMPLES} Sample Challenge)', fontsize=14)
plt.xlabel('Peak Displacement [m]', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.xlim([-0.02, 0.25])
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()