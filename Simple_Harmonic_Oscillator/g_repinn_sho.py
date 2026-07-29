import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import time

#usual cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=====================================")
print(f"Using Hardware Device: {device.type.upper()}")
print(f"=====================================")

#SIREN activasion: check ai notes
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class PDEM_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 256), 
            Sine(), 
            nn.Linear(256, 256),
            Sine(),
            nn.Linear(256, 256),
            Sine(),
            nn.Linear(256, 256),
            Sine(),
            nn.Linear(256, 1) 
        )

    def forward(self, x, t, theta):
        sin_feat = torch.sin(theta * t)
        cos_feat = torch.cos(theta * t)
        inputs = torch.cat([x, t, theta, sin_feat, cos_feat], dim=1)
        raw_p = self.net(inputs)
        
        # Applies a smooth, differentiable floor so p > 0
        return torch.nn.functional.softplus(raw_p)

#losses
def phy_loss(model, x, t, theta, x0=0.1):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    p = model(x, t, theta)
    
    dp_dt = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    dp_dx = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    
    residual = dp_dt - (x0 * theta * torch.sin(theta * t) * dp_dx)
    
    dres_dx = torch.autograd.grad(residual, x, grad_outputs=torch.ones_like(residual), create_graph=True)[0]
    dres_dt = torch.autograd.grad(residual, t, grad_outputs=torch.ones_like(residual), create_graph=True)[0]
    
    loss_p = torch.mean(residual**2)
    loss_gpinn = torch.mean(dres_dx**2) + torch.mean(dres_dt**2)
    
    return loss_p + (0.005 * loss_gpinn)


def ic_loss(model, x_ic, theta_ic, x0=0.1):
    t_zero = torch.zeros_like(x_ic)
    p_pred = model(x_ic, t_zero, theta_ic)
    SD = 0.01 
    p_exact = torch.exp(-0.5 * ((x_ic - x0) / SD)**2) 
    return torch.mean((p_pred - p_exact)**2)


def bc_loss(model, t_bc, theta_bc):
    x_left = torch.full_like(t_bc, -0.2)
    x_right = torch.full_like(t_bc, 0.2)
    p_left = model(x_left, t_bc, theta_bc)
    p_right = model(x_right, t_bc, theta_bc)
    
    target = torch.zeros_like(p_left)
    return torch.mean((p_left - target)**2) + torch.mean((p_right - target)**2)


def integral_loss(model, t_int, theta_int):
    n_points = 200
    x_min, x_max = -0.2, 0.2
    dx = (x_max - x_min) / (n_points - 1)
    
    # ADDED DEVICE
    x_grid = torch.linspace(x_min, x_max, n_points, device=device).unsqueeze(1)
    batch_size = t_int.shape[0]
    
    x_eval = x_grid.repeat(batch_size, 1) 
    t_eval = t_int.repeat_interleave(n_points, dim=0) 
    theta_eval = theta_int.repeat_interleave(n_points, dim=0)
    
    p_pred = model(x_eval, t_eval, theta_eval)
    p_pred = p_pred.view(batch_size, n_points)
    calculated_area = torch.sum(p_pred, dim=1) * dx
    
    target_area = 0.01 * math.sqrt(2 * math.pi) 
    return torch.mean((calculated_area - target_area)**2)


def forbidden_loss(model, t_fb, theta_fb):
    half_batch = t_fb.shape[0] // 2
    
    x_center = 0.1 * torch.cos(theta_fb * t_fb)
    
    # ADDED DEVICE
    x_left = (torch.rand(half_batch, 1, device=device) * (x_center[:half_batch] - 0.04 - (-0.2))) + (-0.2)
    x_right = (torch.rand(half_batch, 1, device=device) * (0.2 - (x_center[half_batch:] + 0.04))) + (x_center[half_batch:] + 0.04)
    
    x_fb = torch.cat([x_left, x_right], dim=0)
    p_pred = model(x_fb, t_fb, theta_fb)
    
    return torch.mean(torch.abs(p_pred))


def peak_loss(model, t_peak, theta_peak):
    x_center = 0.1 * torch.cos(theta_peak * t_peak)
    p_pred = model(x_center, t_peak, theta_peak)
    target_peak = torch.ones_like(p_pred)
    return torch.mean((p_pred - target_peak)**2)

#adam part
model = PDEM_PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 10000 
batch_size = 2000
loss_history = []

x_min, x_max = -0.2, 0.2
t_min = 0.0
theta_min, theta_max = 1.25 * math.pi, 1.75 * math.pi


print(f"Starting {device.type.upper()} PINN Training (gPINN + Softplus Version) for {epochs} epochs...")

start_time = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()
    
    if epoch < 2000:#spereate training to give more focus in initial stages to accurately map
        current_t_max = 2.0
        phase = "t=2.0"
    elif epoch < 4000:
        current_t_max = 4.0
        phase = "t=4.0"
    elif epoch < 6000:
        current_t_max = 6.0
        phase = "t=6.0"
    elif epoch < 8000:
        current_t_max = 8.0
        phase = "t=8.0"
    else:
        current_t_max = 10.0
        phase = "t=10.0"

    #fix?? it seems tensors also stay in cuda, so move to device
    t_col = (torch.rand(batch_size, 1, device=device) * (current_t_max - t_min)) + t_min
    theta_col = (torch.rand(batch_size, 1, device=device) * (theta_max - theta_min)) + theta_min
    
    half_batch = batch_size // 2
    
    x_col_unif = (torch.rand(half_batch, 1, device=device) * (x_max - x_min)) + x_min
    x_center_col = 0.1 * torch.cos(theta_col[half_batch:] * t_col[half_batch:])
    x_col_focus = x_center_col + (torch.randn(half_batch, 1, device=device) * 0.01) 
    x_col = torch.cat([x_col_unif, x_col_focus], dim=0)
    
    x_ic_unif = (torch.rand(half_batch, 1, device=device) * (x_max - x_min)) + x_min
    x_ic_norm = (torch.randn(half_batch, 1, device=device) * 0.01) + 0.1 
    x_ic = torch.cat([x_ic_unif, x_ic_norm], dim=0)
    theta_ic = (torch.rand(batch_size, 1, device=device) * (theta_max - theta_min)) + theta_min
    
    t_bc = (torch.rand(batch_size, 1, device=device) * (current_t_max - t_min)) + t_min
    theta_bc = (torch.rand(batch_size, 1, device=device) * (theta_max - theta_min)) + theta_min
    
    t_int = (torch.rand(50, 1, device=device) * (current_t_max - t_min)) + t_min
    theta_int = (torch.rand(50, 1, device=device) * (theta_max - theta_min)) + theta_min
    
    t_fb = (torch.rand(batch_size, 1, device=device) * (current_t_max - t_min)) + t_min
    theta_fb = (torch.rand(batch_size, 1, device=device) * (theta_max - theta_min)) + theta_min
    
    t_peak = (torch.rand(batch_size, 1, device=device) * (current_t_max - t_min)) + t_min
    theta_peak = (torch.rand(batch_size, 1, device=device) * (theta_max - theta_min)) + theta_min

    loss_i = ic_loss(model, x_ic, theta_ic)
    loss_b = bc_loss(model, t_bc, theta_bc)
    
    if epoch < 1000:
        loss = (20.0 * loss_i) + (25.0 * loss_b)
        
        loss_p = torch.tensor(0.0, device=device) 
        loss_int = torch.tensor(0.0, device=device)
        loss_fb = torch.tensor(0.0, device=device)
        loss_pk = torch.tensor(0.0, device=device)
    else:
        loss_p = phy_loss(model, x_col, t_col, theta_col)
        loss_int = integral_loss(model, t_int, theta_int)
        loss_fb = forbidden_loss(model, t_fb, theta_fb)
        loss_pk = peak_loss(model, t_peak, theta_peak)
        
        loss = loss_p + (20.0 * loss_i) + (25.0 * loss_b) + (50.0 * loss_int) + (10.0 * loss_fb) + (10.0 * loss_pk)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        elapsed_time = time.time() - start_time
        if epoch < 1000:
            print(f"Burn ({phase}) | Ep {epoch:04d} | IC: {loss_i.item():.5f} | BC: {loss_b.item():.5f} | Time: {elapsed_time:.2f}s")
        else:
            print(f"March ({phase}) | Ep {epoch:04d} | Tot: {loss.item():.4f} | Phy: {loss_p.item():.4f} | BC: {loss_b.item():.4f} | Int: {loss_int.item():.4f} | FB: {loss_fb.item():.4f} | Pk: {loss_pk.item():.4f} | Time: {elapsed_time:.2f}s")
        
        start_time = time.time()
        
    loss_history.append(loss.item())

print(f"Final loss = {loss.item():.5f}")

#full loss details
print("\n" + "="*40)
print("FINAL LOSS AUTOPSY")
print("="*40)
print(f"Total Combined Loss: {loss.item():.5f}\n")
print(f"Physics    (Raw: {loss_p.item():.5f}) x 1.0  = {loss_p.item():.5f}")
print(f"Init Cond  (Raw: {loss_i.item():.5f}) x 20.0 = {20.0 * loss_i.item():.5f}")
print(f"Bound Cond (Raw: {loss_b.item():.5f}) x 25.0 = {25.0 * loss_b.item():.5f}")
print(f"Integral   (Raw: {loss_int.item():.5f}) x 50.0 = {50.0 * loss_int.item():.5f}")
print(f"Forbidden  (Raw: {loss_fb.item():.5f}) x 10.0 = {10.0 * loss_fb.item():.5f}")
print(f"Peak       (Raw: {loss_pk.item():.5f}) x 10.0 = {10.0 * loss_pk.item():.5f}")
print("="*40 + "\n")

#ai woek, hehe
print("Generating gPINN vs FDM Comparison Graphs...")

t_vals = [0.5, 1.0, 1.5]
x_plot = np.linspace(-0.2, 0.2, 500)
x_tensor = torch.tensor(x_plot, dtype=torch.float32, device=device).unsqueeze(1)

prof_omegas = np.linspace(1.25 * np.pi, 1.75 * np.pi, 101)
prof_probs = np.ones(101) * (1.0 / 100.0)
prof_probs[0] = 1.0 / 200.0
prof_probs[-1] = 1.0 / 200.0
fdm_diffusion_sd = 0.0035 

# 1x3 Subplot layout
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

model.eval()
with torch.no_grad():
    for idx, t_val in enumerate(t_vals):
        ax = axes[idx]
        print(f"Evaluating and plotting for t = {t_val}s...")
        
        # 1. MATLAB FDM (Computed and Plotted)
        p_fdm = np.zeros_like(x_plot)
        for w, prob in zip(prof_omegas, prof_probs):
            mu = 0.1 * np.cos(w * t_val)
            p_fdm += prob * stats.norm.pdf(x_plot, loc=mu, scale=fdm_diffusion_sd)
        
        ax.plot(x_plot, p_fdm, color='b', linestyle='-', linewidth=2.5, alpha=0.4, label='MATLAB FDM')

        # 2. gPINN Prediction (Computed and Plotted)
        t_tensor = torch.full_like(x_tensor, t_val)
        p_marginal = torch.zeros_like(x_tensor)
        
        num_thetas = 100
        theta_range = torch.linspace(theta_min, theta_max, num_thetas, device=device)
        
        for theta_val in theta_range:
            theta_tensor = torch.full_like(x_tensor, theta_val.item())
            p_marginal += model(x_tensor, t_tensor, theta_tensor)
            
        p_raw = (p_marginal / num_thetas).cpu().numpy().flatten()
        
        # Domain masking and Trapezoidal Normalization (Standardized with earlier runs)
        valid_zone = (np.abs(x_plot) <= 0.1001).astype(float)
        p_raw = p_raw * valid_zone
        area = np.trapezoid(p_raw, x_plot)
        p_scaled = p_raw / area if area > 0.0001 else p_raw
        
        ax.plot(x_plot, p_scaled, color='m', linestyle='--', linewidth=3.0, label='gPINN (Softplus)')

        # Formatting subplots
        ax.set_xlim([-0.2, 0.2])
        # Dynamically scale the Y-axis based on the highest peak at this time step
        ax.set_ylim([-1, max(np.max(p_fdm), np.max(p_scaled)) * 1.2]) 
        ax.set_xlabel("Displacement [m]", fontsize=14)
        ax.set_ylabel("PDF", fontsize=14)
        ax.set_title(f"t = {t_val}s", fontsize=16)
        ax.legend(loc='upper right', edgecolor='black', facecolor='white', framealpha=1.0, fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()