import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# ==========================================
# 0. FIX: REPRODUCIBILITY
# ==========================================
# Original code had no seeding at all -> every run trains on a different
# random 425-sample subset AND different collocation batches, so "least
# samples needed" comparisons across runs were not actually controlled.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# 1. HARDWARE & DATA LOADING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=====================================")
print(f"Using Hardware Device: {device.type.upper()}")
print(f"=====================================")

base_path = r"C:\Users\Sakthi Visahan V\OneDrive\Documents\spyder_stuff\quakity quakity\EQ_SDOF\EQ"

print("Loading MATLAB Text Files...")
gt_data = np.loadtxt(os.path.join(base_path, 'PDF_Ground_Truth.txt'), delimiter=',')
x_gt = gt_data[:, 0]
p_gt = gt_data[:, 1]

raw_peaks = np.loadtxt(os.path.join(base_path, 'Raw_20000_Peaks.txt'), delimiter=',')

# FIX: comment said "350 Sample Challenge" but slice took 425 -- now
# reduced further to 400 per latest request.
N_SAMPLES = 300
np.random.shuffle(raw_peaks)
x_max_samples = torch.tensor(raw_peaks[:N_SAMPLES], dtype=torch.float32, device=device).unsqueeze(1)
print(f"Training on exactly {len(x_max_samples)} earthquake samples!")

# Region of interest for prioritized accuracy (per your request)
SHOULDER_LO, SHOULDER_HI = 0.055, 0.08   # where residual error was concentrated
PEAK_LO, PEAK_HI = 0.02, 0.08            # around the mode of the PDF
TAIL_LO, TAIL_HI = 0.08, 0.13            # the region you specifically need accurate

# ==========================================
# 2. NETWORK ARCHITECTURE (SIREN)
# ==========================================
# FIX #1 (previous round): the old Sine()+siren_init combo initialized
# weights correctly but never scaled by omega_0 in the forward pass.
# FIX #2 (this round): omega_0=30 -- the standard SIREN value for fitting
# static signals (images/audio) where the loss never differentiates the
# network -- breaks derivative-based PINN training. Your physics loss
# backprops through dp/dy and dp/dtau, and the omega_0 factor compounds
# multiplicatively across 4 stacked sine layers during that
# differentiation, which is exactly what caused the gradient explosion
# (PDE loss climbing from 0.03 to 4269, IC loss stuck at 30-40, final
# output pure noise). omega_0=1 keeps the correct init-scaling math
# without the frequency amplification that broke training.
SIREN_OMEGA_0 = 1.0  # was 30.0 -- see note above


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
            SirenLayer(5, 256, is_first=True),
            SirenLayer(256, 256),
            SirenLayer(256, 256),
            SirenLayer(256, 256),
            nn.Linear(256, 1)  # final layer stays linear
        )
        with torch.no_grad():
            bound = math.sqrt(6 / 256) / SIREN_OMEGA_0
            self.net[-1].weight.uniform_(-bound, bound)

    def forward(self, y, tau, x_max):
        # Frequency = 5*pi/2 (kept as in original / "professor's fix")
        sin_feat = torch.sin((5.0 * math.pi / 2.0) * tau)
        cos_feat = torch.cos((5.0 * math.pi / 2.0) * tau)

        inputs = torch.cat([y, tau, x_max, sin_feat, cos_feat], dim=1)
        raw_p = self.net(inputs)
        return torch.nn.functional.softplus(raw_p)


# ==========================================
# 3. PHYSICS LOSS FUNCTIONS (Earthquake Math)
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


def ic_loss(model, y_ic, x_max_ic):
    tau_zero = torch.zeros_like(y_ic)
    p_pred = model(y_ic, tau_zero, x_max_ic)

    sigma = 0.005
    p_exact = (1.0 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * (y_ic / sigma) ** 2)

    return torch.mean((p_pred - p_exact) ** 2)


# ==========================================
# 3b. IMPROVEMENT: REGION-WEIGHTED (IMPORTANCE) SAMPLING
# ==========================================
def sample_y_importance(n, device):
    """
    The original code sampled y uniformly over the full [-0.3, 0.3] domain
    for every loss term. That gives the peak region and the 0.08-0.13 tail
    (the regions you explicitly want prioritized) no more collocation
    density than any other part of the domain. This mixes in extra points
    from those regions without touching your physical data (x_max samples
    are unaffected -- this only changes how densely we probe the PDE/IC
    in y-space, which is "free" in that it costs no extra earthquake data).
    """
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
# 4. TWO-STAGE TRAINING LOOP
# ==========================================
model = PDEM_PINN().to(device)
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=2000, gamma=0.5)

epochs_adam = 12500
batch_size = 6000  # FIX: removed misleading "Reduced from 6,000" comment (no change had occurred)

print(f"Starting Phase 1: Adam Optimization ({epochs_adam} Epochs)...")
start_time = time.time()

for epoch in range(epochs_adam):
    optimizer_adam.zero_grad()

    tau_col = torch.rand(batch_size, 1, device=device)
    y_col = sample_y_importance(batch_size, device)  # IMPROVEMENT: was uniform
    idx = torch.randint(0, len(x_max_samples), (batch_size,))
    x_max_col = x_max_samples[idx]

    tau_bc = torch.rand(batch_size, 1, device=device)
    x_max_bc = x_max_samples[idx]

    tau_int = torch.rand(50, 1, device=device)
    idx_int = torch.randint(0, len(x_max_samples), (50,))
    x_max_int = x_max_samples[idx_int]

    y_ic = sample_y_importance(batch_size, device)  # IMPROVEMENT: was uniform
    idx_ic = torch.randint(0, len(x_max_samples), (batch_size,))
    x_max_ic = x_max_samples[idx_ic]

    loss_p = standard_phy_loss(model, y_col, tau_col, x_max_col)
    loss_b = bc_loss(model, tau_bc, x_max_bc)
    loss_int = integral_loss(model, tau_int, x_max_int)
    loss_i = ic_loss(model, y_ic, x_max_ic)

    loss = (75.0 * loss_p) + (80.0 * loss_i) + (50.0 * loss_b) + (20.0 * loss_int)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer_adam.step()
    scheduler.step()

    if epoch % 1000 == 0:
        elapsed = time.time() - start_time
        print(f"Adam Ep {epoch:04d} | Tot: {loss.item():.3f} | PDE: {loss_p.item():.3f} | IC: {loss_i.item():.3f} | Time: {elapsed:.2f}s")
        start_time = time.time()

# --- PHASE 2: L-BFGS ---
print(f"Starting Phase 2: L-BFGS (Early Stopping)...")

# FIX: comment previously claimed this was "reduced to 2,500" while the
# value was actually 5000 -- kept at 5000 and comment corrected to match.
lbfgs_points = 5000

tau_col_s = torch.rand(lbfgs_points, 1, device=device)
y_col_s = sample_y_importance(lbfgs_points, device)  # IMPROVEMENT: was uniform
idx_s = torch.randint(0, len(x_max_samples), (lbfgs_points,))
x_max_col_s = x_max_samples[idx_s]

tau_bc_s = torch.rand(lbfgs_points, 1, device=device)
x_max_bc_s = x_max_samples[idx_s]

tau_int_s = torch.rand(100, 1, device=device)
idx_int_s = torch.randint(0, len(x_max_samples), (100,))
x_max_int_s = x_max_samples[idx_int_s]

y_ic_s = sample_y_importance(lbfgs_points, device)  # IMPROVEMENT: was uniform
idx_ic_s = torch.randint(0, len(x_max_samples), (lbfgs_points,))
x_max_ic_s = x_max_samples[idx_ic_s]

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=0.003,
    max_iter=10,
    max_eval=15,
    history_size=50,
    line_search_fn="strong_wolfe"
)
epochs_lbfgs = 15

start_time = time.time()

for epoch in range(epochs_lbfgs):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss_p_s = standard_phy_loss(model, y_col_s, tau_col_s, x_max_col_s)
        loss_b_s = bc_loss(model, tau_bc_s, x_max_bc_s)
        loss_int_s = integral_loss(model, tau_int_s, x_max_int_s)
        loss_i_s = ic_loss(model, y_ic_s, x_max_ic_s)

        loss_s = (75.0 * loss_p_s) + (80.0 * loss_i_s) + (50.0 * loss_b_s) + (20.0 * loss_int_s)
        loss_s.backward()
        return loss_s

    optimizer_lbfgs.step(closure)

    current_loss = closure().item()
    elapsed = time.time() - start_time
    print(f"L-BFGS Ep {epoch:04d} | Tot Loss: {current_loss:.4f} | Time: {elapsed:.2f}s")
    start_time = time.time()

print("Two-Stage Training Complete.")

# ==========================================
# 5. PLOTTING THE SHOWDOWN
# ==========================================
print("Plotting Final Comparison...")
model.eval()

y_plot_tensor = torch.tensor(x_gt, dtype=torch.float32, device=device).unsqueeze(1)
tau_final = torch.ones_like(y_plot_tensor)

p_predicted = torch.zeros_like(y_plot_tensor)

with torch.no_grad():
    for x_m in x_max_samples:
        x_m_tensor = torch.full_like(y_plot_tensor, x_m.item())
        p_predicted += model(y_plot_tensor, tau_final, x_m_tensor)

    p_raw = (p_predicted / len(x_max_samples)).cpu().numpy().flatten()

area = np.trapezoid(p_raw, x_gt)
if area > 0:
    p_normalized = p_raw / area
else:
    p_normalized = p_raw

# --- IMPROVEMENT: quick quantitative check on the region you care about ---
mask = (x_gt >= TAIL_LO) & (x_gt <= TAIL_HI)
if mask.sum() > 0:
    tail_l2 = np.sqrt(np.mean((p_normalized[mask] - p_gt[mask]) ** 2))
    peak_err = abs(p_normalized.max() - p_gt.max())
    print(f"Tail region [{TAIL_LO},{TAIL_HI}] RMSE: {tail_l2:.4f}")
    print(f"Peak height error: {peak_err:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(x_gt, p_gt, 'b-', linewidth=3, label='MATLAB 20,000 MCS (Ground Truth)')
plt.plot(x_gt, p_normalized, 'r--', linewidth=2.5, label=f'Base PINN ({len(x_max_samples)} Samples)')

plt.title(f'Base PINN vs Ground Truth ({len(x_max_samples)} Sample Challenge)', fontsize=14)
plt.xlabel('Peak Displacement [m]', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.xlim([-0.02, 0.25])
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
