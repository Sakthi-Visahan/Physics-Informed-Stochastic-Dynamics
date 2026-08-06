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

seed = 42
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

N_SAMPLES = 200
np.random.shuffle(raw_peaks)
x_max_samples_np = raw_peaks[:N_SAMPLES]
x_max_samples = torch.tensor(x_max_samples_np, dtype=torch.float32, device=device).unsqueeze(1)
print(f"Training on exactly {len(x_max_samples)} earthquake samples!")

# USER FIX: Narrowed physics sampling regions for intense apex focus
PEAK_LO, PEAK_HI = 0.035, 0.06   
SHOULDER_LO, SHOULDER_HI = 0.06, 0.08
TAIL_LO, TAIL_HI = 0.08, 0.13

# The Data Sniper Regions
CORE_LO, CORE_HI = 0.045, 0.06  
PRIORITY_LO, PRIORITY_HI = 0.07, 0.10  
LEFT_LO, LEFT_HI = -0.02, 0.035

# ==========================================
# 2. NETWORK ARCHITECTURE
# ==========================================
SIREN_OMEGA_0 = 1.0

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=SIREN_OMEGA_0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.linear.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class PDEM_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SirenLayer(5, 256, is_first=True, omega_0=30.0),
            SirenLayer(256, 256, omega_0=1.0),
            SirenLayer(256, 256, omega_0=1.0),
            SirenLayer(256, 256, omega_0=1.0),
            nn.Linear(256, 1)
        )
        with torch.no_grad():
            bound = math.sqrt(6 / 256) / 1.0
            self.net[-1].weight.uniform_(-bound, bound)

    def forward(self, y, tau, x_max):
        sin_feat = torch.sin((5.0 * math.pi / 2.0) * tau)
        cos_feat = torch.cos((5.0 * math.pi / 2.0) * tau)
        inputs = torch.cat([y, tau, x_max, sin_feat, cos_feat], dim=1)
        raw_p = self.net(inputs)
        return torch.nn.functional.softplus(raw_p)

# ==========================================
# 3. PHYSICS LOSS FUNCTIONS
# ==========================================
def standard_phy_loss(model, y, tau, x_max):
    y = y.clone().detach().requires_grad_(True)
    tau = tau.clone().detach().requires_grad_(True)
    p = model(y, tau, x_max)

    dp_dtau = torch.autograd.grad(p, tau, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    dp_dy = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    y_dot = x_max * (5.0 * math.pi / 2.0) * torch.cos((5.0 * math.pi / 2.0) * tau)
    residual = dp_dtau + (y_dot * dp_dy)
    return torch.mean(residual ** 2)

def bc_loss(model, tau_bc, x_max_bc):
    y_left = torch.full_like(tau_bc, -0.3)
    y_right = torch.full_like(tau_bc, 0.3)
    p_left = model(y_left, tau_bc, x_max_bc)
    p_right = model(y_right, tau_bc, x_max_bc)
    target = torch.zeros_like(p_left)
    return torch.mean((p_left - target) ** 2) + torch.mean((p_right - target) ** 2)

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
    return torch.mean((calculated_area - target_area) ** 2)

SIGMA_IC_START = 0.02
SIGMA_IC_END = 0.0035

def ic_loss(model, y_ic, x_max_ic, sigma=SIGMA_IC_END):
    tau_zero = torch.zeros_like(y_ic)
    p_pred = model(y_ic, tau_zero, x_max_ic)
    p_exact = (1.0 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * (y_ic / sigma) ** 2)
    return torch.mean((p_pred - p_exact) ** 2)

# ==========================================
# 3b. IMPORTANCE SAMPLING
# ==========================================
def sample_y_importance(n, device):
    n_uniform = int(n * 0.4)
    n_peak = int(n * 0.25)
    n_shoulder = int(n * 0.15)
    n_tail = n - n_uniform - n_peak - n_shoulder

    y_uniform = (torch.rand(n_uniform, 1, device=device) * 0.6) - 0.3
    y_peak = torch.rand(n_peak, 1, device=device) * (PEAK_HI - PEAK_LO) + PEAK_LO
    y_shoulder = torch.rand(n_shoulder, 1, device=device) * (SHOULDER_HI - SHOULDER_LO) + SHOULDER_LO
    y_tail = torch.rand(n_tail, 1, device=device) * (TAIL_HI - TAIL_LO) + TAIL_LO

    y = torch.cat([y_uniform, y_peak, y_shoulder, y_tail], dim=0)
    perm = torch.randperm(y.shape[0], device=device)
    return y[perm]

# ==========================================
# 3c. KDE reference density
# ==========================================
# USER FIX: Added bandwidth_factor to stop Silverman from over-smoothing the peak
def silverman_kde(samples_np, eval_points_np, bandwidth_factor=0.75):
    N = len(samples_np)
    sigma_hat = np.std(samples_np, ddof=1)
    h = bandwidth_factor * 1.06 * sigma_hat * N ** (-1 / 5)
    diffs = (eval_points_np[:, None] - samples_np[None, :]) / h
    K = np.exp(-0.5 * diffs ** 2) / np.sqrt(2 * np.pi)
    density = K.sum(axis=1) / (N * h)
    return density

p_ref_kde_np = silverman_kde(x_max_samples_np, x_gt, bandwidth_factor=0.75)
p_ref_kde = torch.tensor(p_ref_kde_np, dtype=torch.float32, device=device)
y_grid_full = torch.tensor(x_gt, dtype=torch.float32, device=device).unsqueeze(1)

w_np = np.ones_like(x_gt)
w_np[(x_gt >= LEFT_LO) & (x_gt <= LEFT_HI)] *= 5.0
w_np[(x_gt >= PEAK_LO) & (x_gt <= PEAK_HI)] *= 2.0
w_np[(x_gt >= CORE_LO) & (x_gt <= CORE_HI)] *= 5.0
w_np[(x_gt >= PRIORITY_LO) & (x_gt <= PRIORITY_HI)] *= 2.0 
w_data = torch.tensor(w_np, dtype=torch.float32, device=device)

def data_fidelity_loss(model, x_max_pool, n_ensemble=30, fixed_subset=None):
    if fixed_subset is not None:
        subset = fixed_subset
    else:
        sorted_pool, _ = torch.sort(x_max_pool, dim=0)
        N_pool = sorted_pool.shape[0]
        bucket_edges = torch.linspace(0, N_pool, n_ensemble + 1).long()
        idx_list = []
        for i in range(n_ensemble):
            lo = bucket_edges[i].item()
            hi = max(bucket_edges[i + 1].item(), lo + 1)
            hi = min(hi, N_pool)
            idx_list.append(torch.randint(lo, hi, (1,), device=x_max_pool.device))
        idx = torch.cat(idx_list)
        subset = sorted_pool[idx]

    N_sub = subset.shape[0]
    N_y = y_grid_full.shape[0]
    y_eval = y_grid_full.repeat(N_sub, 1)
    tau1_eval = torch.ones_like(y_eval)
    x_eval = subset.repeat_interleave(N_y, dim=0)
    p_preds = model(y_eval, tau1_eval, x_eval).view(N_sub, N_y)
    p_avg = p_preds.mean(dim=0)
    area = torch.trapz(p_avg, torch.tensor(x_gt, dtype=torch.float32, device=device))
    p_avg_norm = p_avg / (area + 1e-8)
    return torch.mean(w_data * (p_avg_norm - p_ref_kde) ** 2)

# ==========================================
# 4. TWO-STAGE TRAINING LOOP
# ==========================================
epochs_adam = 45000
batch_size = 6000

model = PDEM_PINN().to(device)
# USER FIX: Removed weight_decay to allow the network to form sharp peak features
optimizer_adam = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=epochs_adam, eta_min=1e-5)

print(f"Starting Phase 1: Universal Adam Optimization ({epochs_adam} Epochs)...")
start_time = time.time()

# USER FIX: Reverted W_DATA to 55.0 and W_IC to 100.0 based on empirical evidence
W_PDE, W_IC, W_BC, W_INT, W_DATA = 75.0, 100.0, 50.0, 20.0, 55.0

for epoch in range(epochs_adam):
    optimizer_adam.zero_grad()
    tau_col = torch.rand(batch_size, 1, device=device)
    x_max_col = torch.rand(batch_size, 1, device=device) * 0.3
    y_col = sample_y_importance(batch_size, device)
    tau_bc = torch.rand(batch_size, 1, device=device)
    x_max_bc = torch.rand(batch_size, 1, device=device) * 0.3
    tau_int = torch.rand(50, 1, device=device)
    x_max_int = torch.rand(50, 1, device=device) * 0.3
    y_ic = sample_y_importance(batch_size, device)
    x_max_ic = torch.rand(batch_size, 1, device=device) * 0.3

    progress = min(epoch / (epochs_adam * 0.8), 1.0)
    current_sigma = SIGMA_IC_START - progress * (SIGMA_IC_START - SIGMA_IC_END)

    loss_p = standard_phy_loss(model, y_col, tau_col, x_max_col)
    loss_b = bc_loss(model, tau_bc, x_max_bc)
    loss_int = integral_loss(model, tau_int, x_max_int)
    loss_i = ic_loss(model, y_ic, x_max_ic, sigma=current_sigma)
    loss_d = data_fidelity_loss(model, x_max_samples, n_ensemble=30)

    loss = (W_PDE * loss_p) + (W_IC * loss_i) + (W_BC * loss_b) + (W_INT * loss_int) + (W_DATA * loss_d)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer_adam.step()
    scheduler.step()

    if epoch % 1500 == 0:
        elapsed = time.time() - start_time
        print(f"Adam Ep {epoch:04d} | Tot: {loss.item():.3f} | PDE: {loss_p.item():.3f} | "
              f"IC: {loss_i.item():.3f} | Data: {loss_d.item():.3f} | Int: {loss_int.item():.3f}")
        start_time = time.time()

print(f"Starting Phase 2: L-BFGS Polish...")

lbfgs_points = 6000
tau_col_s = torch.rand(lbfgs_points, 1, device=device)
x_max_col_s = torch.rand(lbfgs_points, 1, device=device) * 0.3
y_col_s = sample_y_importance(lbfgs_points, device)
tau_bc_s = torch.rand(lbfgs_points, 1, device=device)
x_max_bc_s = torch.rand(lbfgs_points, 1, device=device) * 0.3
tau_int_s = torch.rand(50, 1, device=device)
x_max_int_s = torch.rand(50, 1, device=device) * 0.3
y_ic_s = sample_y_importance(lbfgs_points, device)
x_max_ic_s = torch.rand(lbfgs_points, 1, device=device) * 0.3

n_ens_lbfgs = min(200, len(x_max_samples))
lbfgs_ensemble_idx = torch.randperm(len(x_max_samples))[:n_ens_lbfgs]
x_max_ensemble_s = x_max_samples[lbfgs_ensemble_idx]

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=0.1, max_iter=50, max_eval=60,
    history_size=50, line_search_fn="strong_wolfe"
)

epochs_lbfgs = 10
start_time = time.time()

for epoch in range(epochs_lbfgs):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss_p_s = standard_phy_loss(model, y_col_s, tau_col_s, x_max_col_s)
        loss_b_s = bc_loss(model, tau_bc_s, x_max_bc_s)
        loss_int_s = integral_loss(model, tau_int_s, x_max_int_s)
        loss_i_s = ic_loss(model, y_ic_s, x_max_ic_s, sigma=SIGMA_IC_END)
        loss_d_s = data_fidelity_loss(model, x_max_samples, fixed_subset=x_max_ensemble_s)
        loss_s = (W_PDE * loss_p_s) + (W_IC * loss_i_s) + (W_BC * loss_b_s) + (W_INT * loss_int_s) + (W_DATA * loss_d_s)
        loss_s.backward()
        return loss_s

    optimizer_lbfgs.step(closure)
    current_loss = closure().item()
    elapsed = time.time() - start_time
    print(f"L-BFGS Ep {epoch:04d} | Tot Loss: {current_loss:.4f} | Time: {elapsed:.2f}s")
    start_time = time.time()
    if not math.isfinite(current_loss):
        print("L-BFGS loss went non-finite -- stopping Phase 2 early.")
        break

print("Two-Stage RE-PINN Training Complete.")

# ==========================================
# 5. EVALUATION + RE-PINN LOCAL RESIDUAL ENHANCEMENT
# ==========================================
print("Evaluating via KDE-weighted continuous quadrature over x_max...")
model.eval()
y_plot_tensor = torch.tensor(x_gt, dtype=torch.float32, device=device).unsqueeze(1)
tau_final = torch.ones_like(y_plot_tensor)

def build_xmax_quadrature(samples_np, grid_size=400, pad_frac=0.15):
    lo = max(0.0, samples_np.min() - pad_frac * samples_np.max())
    hi = samples_np.max() * (1 + pad_frac)
    grid = np.linspace(lo, hi, grid_size)
    # Applied the exact same bandwidth sharpening to the evaluation KDE
    dens = silverman_kde(samples_np, grid, bandwidth_factor=0.75) 
    dens = dens / np.trapezoid(dens, grid)
    return grid, dens

xmax_grid_np, xmax_dens_np = build_xmax_quadrature(x_max_samples_np)
dx_xmax = xmax_grid_np[1] - xmax_grid_np[0]
xmax_grid = torch.tensor(xmax_grid_np, dtype=torch.float32, device=device)
xmax_weights = torch.tensor(xmax_dens_np * dx_xmax, dtype=torch.float32, device=device)

p_predicted = torch.zeros_like(y_plot_tensor)
with torch.no_grad():
    for j in range(len(xmax_grid)):
        x_m_tensor = torch.full_like(y_plot_tensor, xmax_grid[j].item())
        p_predicted += xmax_weights[j] * model(y_plot_tensor, tau_final, x_m_tensor)
    p_raw = p_predicted.cpu().numpy().flatten()

area = np.trapezoid(p_raw, x_gt)
p_model = p_raw / area if area > 0 else p_raw

GAMMA_R = 0.15
GAMMA_P = 0.80  
SIGMA_R = 0.008
WINDOW_HALFWIDTH = 0.02

peak_idx = np.argmax(p_model)
y_star = x_gt[peak_idx]
p_ref_kde_norm = p_ref_kde_np / (np.trapezoid(p_ref_kde_np, x_gt) + 1e-8)
residual = p_ref_kde_norm - p_model

g_i = np.exp(-((x_gt - y_star) ** 2) / (2 * SIGMA_R ** 2))
p_hat = p_model + GAMMA_R * g_i * residual

window_mask = np.abs(x_gt - y_star) <= WINDOW_HALFWIDTH
p_hat[window_mask] = (1 - GAMMA_P) * p_hat[window_mask] + GAMMA_P * p_ref_kde_norm[window_mask]

p_hat = np.maximum(p_hat, 0.0)
area_hat = np.trapezoid(p_hat, x_gt)
p_re = p_hat / area_hat if area_hat > 0 else p_hat

def rmse(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(np.sqrt(np.mean((a - b) ** 2)))

tail_mask = (x_gt >= TAIL_LO) & (x_gt <= TAIL_HI)

print("\n--- Honesty check: does RE enhancement beat naive KDE and physics-only? ---")
print(f"Naive KDE-only  (N={N_SAMPLES} samples) vs 20k truth  RMSE: {rmse(p_ref_kde_norm, p_gt):.4f}")
print(f"Physics-informed PINN (pre-enhancement)   vs 20k truth  RMSE: {rmse(p_model, p_gt):.4f}")
print(f"RE-PINN (post-enhancement)                vs 20k truth  RMSE: {rmse(p_re, p_gt):.4f}")
print(f"Tail [{TAIL_LO},{TAIL_HI}] -- naive KDE RMSE: {rmse(p_ref_kde_norm, p_gt, tail_mask):.4f} | "
      f"PINN RMSE: {rmse(p_model, p_gt, tail_mask):.4f} | RE-PINN RMSE: {rmse(p_re, p_gt, tail_mask):.4f}")
print(f"Peak height error -- PINN: {abs(p_model.max()-p_gt.max()):.4f} | RE-PINN: {abs(p_re.max()-p_gt.max()):.4f}")

plt.figure(figsize=(10, 6))
plt.plot(x_gt, p_gt, 'b-', linewidth=3, label='MATLAB 20,000 MCS (Ground Truth)')
plt.plot(x_gt, p_re, 'g-.', linewidth=2.5, label=f'RE-PINN Enhanced ({N_SAMPLES} Samples)')

plt.title(f'RE-PINN vs Ground Truth ({N_SAMPLES} Sample Challenge)', fontsize=14)
plt.xlabel('Peak Displacement [m]', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.xlim([-0.02, 0.25])
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()