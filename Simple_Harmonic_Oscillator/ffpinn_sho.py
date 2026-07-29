import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=====================================")
print(f"Using Hardware Device: {device.type.upper()}")
print(f"=====================================")
#above is made by ai, i dont know cuda

#usuall siren
class Sine(nn.Module):
    def forward(self, x): 
        return torch.sin(x)

class FourierFeatureMapping(nn.Module):
    def __init__(self, in_features, mapping_size, scale=1.0): 
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)
        
    def forward(self, x):
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FF_PDEM_PINN(nn.Module):
    def __init__(self, mapping_size=128):
        super().__init__()
        self.ff_layer = FourierFeatureMapping(3, mapping_size, scale=1.0)
        
        self.net = nn.Sequential(
            nn.Linear(2 * mapping_size, 256), Sine(),
            nn.Linear(256, 256), Sine(),
            nn.Linear(256, 256), Sine(),
            nn.Linear(256, 256), Sine(),
            nn.Linear(256, 1) 
        )
        
    def forward(self, x, t, theta):
        inputs = torch.cat([x, t, theta], dim=1)
        ff_inputs = self.ff_layer(inputs)
        raw_output = self.net(ff_inputs)
        # Squaring allows for sharp corners and true zeroes
        return raw_output**2

#usuall losses
def physics_loss(model, batch_size, device):
    t_col = torch.rand(batch_size, 1, device=device) * 10.0
    theta_col = (torch.rand(batch_size, 1, device=device) * (0.5 * math.pi)) + (1.25 * math.pi)
    
    #3 tier, check repinn.py for more details
    n1 = batch_size // 3
    n2 = batch_size // 3
    n3 = batch_size - n1 - n2  
    
    #spread
    x_uniform = (torch.rand(n1, 1, device=device) * 0.4) - 0.2
    
    #cliff
    x_c1 = 0.1 * torch.cos(theta_col[n1:n1+n2] * t_col[n1:n1+n2]).detach()
    x_peak = x_c1 + (torch.randn(n2, 1, device=device) * 0.01)
    
    #middle
    x_c2 = 0.1 * torch.cos(theta_col[n1+n2:] * t_col[n1+n2:]).detach()
    x_plateau = x_c2 + (torch.randn(n3, 1, device=device) * 0.04) 
    
    #combine all points
    x_col = torch.cat([x_uniform, x_peak, x_plateau], dim=0)
    
    x_col.requires_grad_(True)
    t_col.requires_grad_(True)
    
    p = model(x_col, t_col, theta_col)
    
    dp_dt = torch.autograd.grad(p, t_col, torch.ones_like(p), create_graph=True)[0]
    dp_dx = torch.autograd.grad(p, x_col, torch.ones_like(p), create_graph=True)[0]
    
    residual = dp_dt - (0.1 * theta_col * torch.sin(theta_col * t_col) * dp_dx)
    
    target = torch.zeros_like(residual)
    return torch.nn.functional.smooth_l1_loss(residual, target, beta=0.1)

def bc_loss(model, batch_size, device):
    t_bc = torch.rand(batch_size, 1, device=device) * 10.0
    theta_bc = (torch.rand(batch_size, 1, device=device) * (0.5 * math.pi)) + (1.25 * math.pi)
    
    # Moved boundaries inward from +/- 0.2 to +/- 0.11.
    x_edges = torch.tensor([[-0.11], [0.11]], device=device).repeat(batch_size // 2, 1)
    
    p_bc = model(x_edges, t_bc, theta_bc)
    return torch.mean(p_bc**2)

def ic_loss(model, batch_size, device):
    x_ic = (torch.rand(batch_size, 1, device=device) * 0.4) - 0.2
    t_ic = torch.zeros_like(x_ic)
    theta_ic = (torch.rand(batch_size, 1, device=device) * (0.5 * math.pi)) + (1.25 * math.pi)
    
    p_ic = model(x_ic, t_ic, theta_ic)
    p_exact = torch.exp(-0.5 * ((x_ic - 0.1) / 0.01)**2)
    return torch.mean((p_ic - p_exact)**2)

def forbidden_loss(model, batch_size, device):
    t_fb = torch.rand(batch_size, 1, device=device) * 10.0
    theta_fb = (torch.rand(batch_size, 1, device=device) * (0.5 * math.pi)) + (1.25 * math.pi)
    
    x_center = 0.1 * torch.cos(theta_fb * t_fb).detach()
    half_b = batch_size // 2
    
    allowance = 0.04
    
    x_left = (torch.rand(half_b, 1, device=device) * (x_center[:half_b] - allowance - (-0.2))) - 0.2
    x_right = (torch.rand(half_b, 1, device=device) * (0.2 - (x_center[half_b:] + allowance))) + (x_center[half_b:] + allowance)
    x_fb = torch.cat([x_left, x_right], dim=0)
    
    p_fb = model(x_fb, t_fb, theta_fb)
    
    # THE FIX: Smooth L1 Loss. 
    # It softens the wall to prevent the "hump", but rigidly snaps the trailing foot to absolute zero!
    target = torch.zeros_like(p_fb)
    return torch.nn.functional.smooth_l1_loss(p_fb, target, beta=0.05)

def peak_loss(model, batch_size, device):
    t_pk = torch.rand(batch_size, 1, device=device) * 10.0
    theta_pk = (torch.rand(batch_size, 1, device=device) * (0.5 * math.pi)) + (1.25 * math.pi)
    x_center = 0.1 * torch.cos(theta_pk * t_pk).detach()
    
    return torch.mean((model(x_center, t_pk, theta_pk) - 1.0)**2)

def integral_loss(model, device, num_samples=50):
    t_int = torch.rand(num_samples, 1, device=device) * 10.0
    theta_int = (torch.rand(num_samples, 1, device=device) * (0.5 * math.pi)) + (1.25 * math.pi)
    x_grid = torch.linspace(-0.2, 0.2, 200, device=device).unsqueeze(1)
    
    x_eval = x_grid.repeat(num_samples, 1) 
    t_eval = t_int.repeat_interleave(200, dim=0) 
    theta_eval = theta_int.repeat_interleave(200, dim=0)
    
    p_int = model(x_eval, t_eval, theta_eval).view(num_samples, 200)
    calculated_area = torch.sum(p_int, dim=1) * (0.4 / 199.0)
    target_area = 0.01 * math.sqrt(2 * math.pi) 
    
    return torch.mean((calculated_area - target_area)**2)

#adam
model = FF_PDEM_PINN(mapping_size=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Standard Adam, no weight decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

epochs = 10000
batch_size = 2000 

print(f"Starting Stage 1: Adam Training on {device}...")
start_time = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()
    
    loss_p = physics_loss(model, batch_size, device)
    loss_i = ic_loss(model, batch_size, device)
    loss_b = bc_loss(model, batch_size, device)
    loss_fb = forbidden_loss(model, batch_size, device)
    loss_pk = peak_loss(model, batch_size, device)
    loss_int = integral_loss(model, device)
    
    # Changed 20.0 * loss_fb back to 10.0 * loss_fb, pls check here
    loss = loss_p + (20.0 * loss_i) + (25.0 * loss_b) + (50.0 * loss_int) + (10.0 * loss_fb) + (5.0 * loss_pk)
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:04d} | Total Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")
        start_time = time.time()

#L-BFGS
print("\nStarting Stage 2: L-BFGS Smoothing Phase...")

# L-BFGS evaluates exact curvature to iron out high-frequency vibrations
lbfgs_optimizer = torch.optim.LBFGS(
    model.parameters(), 
    max_iter=1000, 
    history_size=50,
    tolerance_grad=1e-5, 
    tolerance_change=1e-9,
    line_search_fn="strong_wolfe" 
)

def closure():
    lbfgs_optimizer.zero_grad()
    
    loss_p = physics_loss(model, batch_size, device)
    loss_i = ic_loss(model, batch_size, device)
    loss_b = bc_loss(model, batch_size, device)
    loss_fb = forbidden_loss(model, batch_size, device)
    loss_pk = peak_loss(model, batch_size, device)
    loss_int = integral_loss(model, device)
    
    # Changed 20.0 * loss_fb back to 10.0 * loss_fb
    loss = loss_p + (20.0 * loss_i) + (25.0 * loss_b) + (50.0 * loss_int) + (10.0 * loss_fb) + (5.0 * loss_pk)
    
    loss.backward()
    return loss


lbfgs_optimizer.step(closure)#needed as it doesnt close on own
final_loss = closure().item()
print(f"L-BFGS Smoothing Complete. Final Loss: {final_loss:.4f}\n")

#plotting by ai
print("Generating MATLAB FDM Graphs...")

t_vals = [0.5, 1.0, 1.5]
x_plot = np.linspace(-0.2, 0.2, 500)

prof_omegas = np.linspace(1.25 * np.pi, 1.75 * np.pi, 101)
prof_probs = np.ones(101) * (1.0 / 100.0)
prof_probs[0] = 1.0 / 200.0
prof_probs[-1] = 1.0 / 200.0
fdm_diffusion_sd = 0.0035 

# 1x3 Subplot layout
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, t_val in enumerate(t_vals):
    ax = axes[idx]
    print(f"Evaluating and plotting FDM for t = {t_val}s...")

    # MATLAB FDM (Computed and Plotted)
    p_fdm = np.zeros_like(x_plot)
    for w, prob in zip(prof_omegas, prof_probs):
        mu = 0.1 * np.cos(w * t_val)
        p_fdm += prob * stats.norm.pdf(x_plot, loc=mu, scale=fdm_diffusion_sd)
    
    ax.plot(x_plot, p_fdm, color='b', linestyle='-', linewidth=2.5, alpha=0.7, label='MATLAB FDM')

    # Formatting subplots
    ax.set_xlim([-0.2, 0.2])
    # Dynamically scale the Y-axis based on the wave's height
    ax.set_ylim([-1, np.max(p_fdm) * 1.2]) 
    ax.set_xlabel("Displacement [m]", fontsize=14)
    ax.set_ylabel("PDF", fontsize=14)
    ax.set_title(f" FDM at t = {t_val}s", fontsize=16)
    ax.legend(loc='upper right', edgecolor='black', facecolor='white', framealpha=1.0, fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()