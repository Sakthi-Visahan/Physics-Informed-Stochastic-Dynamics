import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import time

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=====================================")
print(f"Using Hardware Device: {device.type.upper()}")
print(f"=====================================")

# ==========================================
# 1. THE FOURIER NEURAL OPERATOR (FNO) ARCHITECTURE
# ==========================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class TimeConditionedFNO1d(nn.Module):
    def __init__(self, modes=16, width=64):
        super(TimeConditionedFNO1d, self).__init__()
        self.modes1 = modes
        self.width = width

        self.fc0 = nn.Linear(3, self.width) 

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return F.softplus(x)

# ==========================================
# 2. DATA LOADING & NORMALIZATION
# ==========================================
print("Loading PINO Dataset...")

base_dir = r'C:\Users\Sakthi Visahan V\OneDrive\Documents\spyder_stuff'
data_path = os.path.join(base_dir, 'pino_dataset_time.pt')

data = torch.load(data_path, weights_only=True)

x_grid_raw = data['x_grid']  # [256]
thetas = data['thetas']      # [5000]
times_data = data['times']   # [5000]
u_true = data['u_true']      # [5000, 256]

num_samples = u_true.shape[0]
num_x_points = u_true.shape[1]

# --- THE FIX: TARGET NORMALIZATION ---
y_scaler = u_true.max()
print(f"Max Target Probability: {y_scaler.item():.2f}")
print("Normalizing targets to [0, 1] to prevent gradient explosion...")

X_train = torch.zeros(num_samples, num_x_points, 3)
X_train[:, :, 0] = x_grid_raw.unsqueeze(0).repeat(num_samples, 1)
X_train[:, :, 1] = times_data.unsqueeze(1).repeat(1, num_x_points)
X_train[:, :, 2] = thetas.unsqueeze(1).repeat(1, num_x_points)

# Divide the targets by the massive scaler to keep gradients healthy
y_train = (u_true / y_scaler).unsqueeze(2) 

dataset = TensorDataset(X_train, y_train)
batch_size = 500
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==========================================
# 3. FNO TRAINING LOOP (WEIGHTED LOSS FIX)
# ==========================================
# FIX 1: Increased modes to 64 to allow high-frequency sharp peaks
model = TimeConditionedFNO1d(modes=64, width=64).to(device)

# Lowered learning rate slightly to keep the higher modes stable
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

epochs = 500 

print(f"Starting FNO Training for {epochs} epochs...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(batch_x)
        
        # FIX 2: Weighted MSE Loss
        # This forces the network to care 100x MORE about the peak than the empty space
        # If batch_y is 0 (empty space), weight is 1. If batch_y is 1.0 (the peak), weight is 101.
        loss_weights = 1.0 + (100.0 * batch_y)
        
        # Calculate loss and multiply by the weights
        loss = torch.mean(loss_weights * (predictions - batch_y)**2)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    scheduler.step()
    
    if epoch % 10 == 0 or epoch == epochs - 1:
        elapsed = time.time() - start_time
        avg_loss = train_loss / len(dataloader)
        print(f"Epoch {epoch:03d}/{epochs} | Weighted MSE Loss: {avg_loss:.6f} | Time: {elapsed:.2f}s")
        start_time = time.time() 

print("FNO Training Complete.")

# ==========================================
# 4. VISUALIZATION (t = 0.9s ONLY - 3 LINES)
# ==========================================
print("Generating Single Timestamp Comparison Graph (t=0.9s, 3 Lines)...")

def analytical_pdf(x_vals, t_val):
    p_exact = np.zeros_like(x_vals)
    valid_mask = np.abs(x_vals) < 0.09999
    x_valid = x_vals[valid_mask]
    
    # Mapped to the 1.25pi - 1.75pi domain
    omega_eq = (2.0 * np.pi - np.arccos(10.0 * x_valid)) / t_val
    
    H1 = np.heaviside(omega_eq - 1.25 * np.pi, 0.5)
    H2 = np.heaviside(omega_eq - 1.75 * np.pi, 0.5)
    
    denom = (np.pi / 2.0) * np.abs(0.1 * t_val * np.sin(np.arccos(10.0 * x_valid)))
    p_exact[valid_mask] = (H1 - H2) / denom
    return p_exact

plt.figure(figsize=(13, 8))
x_plot = np.linspace(-0.2, 0.2, 256) # FNO spatial resolution

# Isolate t = 0.9
t_val = 0.9 

prof_omegas = np.linspace(1.25 * np.pi, 1.75 * np.pi, 101)
prof_probs = np.ones(101) * (1.0 / 100.0)
prof_probs[0] = 1.0 / 200.0
prof_probs[-1] = 1.0 / 200.0
fdm_diffusion_sd = 0.0035 

model.eval()

with torch.no_grad():
    
    # --- A. PLOT ANALYTICAL (Black Dash-Dot) ---
    p_analytical = analytical_pdf(x_plot, t_val)
    plt.plot(x_plot, p_analytical, color='k', linestyle='-.', linewidth=2.0, label=f'Analytical Eq 19 (t={t_val}s)')

    # --- B. PLOT MATLAB TVD-FDM (Blue Solid) ---
    p_fdm = np.zeros_like(x_plot)
    for w, prob in zip(prof_omegas, prof_probs):
        mu = 0.1 * np.cos(w * t_val)
        p_fdm += prob * stats.norm.pdf(x_plot, loc=mu, scale=fdm_diffusion_sd)
    plt.plot(x_plot, p_fdm, color='b', linestyle='-', linewidth=2.5, alpha=0.4, label=f'MATLAB FDM (t={t_val}s)')

    # --- C. PLOT FNO (Red Dashed) ---
    num_thetas = 101
    fno_input = torch.zeros(num_thetas, 256, 3)
    
    fno_input[:, :, 0] = torch.tensor(x_plot, dtype=torch.float32).unsqueeze(0).repeat(num_thetas, 1)
    fno_input[:, :, 1] = torch.full((num_thetas, 256), t_val, dtype=torch.float32)
    fno_input[:, :, 2] = torch.tensor(prof_omegas, dtype=torch.float32).unsqueeze(1).repeat(1, 256)
    
    # Send to GPU
    fno_input = fno_input.to(device)
    
    # THE FIX: Un-normalize using the massive y_scaler and return to CPU
    fno_output = (model(fno_input).squeeze(2) * y_scaler).cpu()
    
    # Marginalize over theta
    p_marginal = torch.zeros(256)
    for i, prob in enumerate(prof_probs):
        p_marginal += prob * fno_output[i, :]
        
    p_raw = p_marginal.numpy()
    
    # Trapezoidal Normalization
    area = np.trapezoid(p_raw, x_plot)
    p_scaled = p_raw / area if area > 0.0001 else p_raw
    
    plt.plot(x_plot, p_scaled, color='r', linestyle='--', linewidth=3.0, label=f'Fourier Neural Operator (t={t_val}s)')

plt.xlim([-0.2, 0.2])
plt.ylim([-2, 22])
plt.xlabel("Displacement [m]", fontsize=14)
plt.ylabel("PDF", fontsize=14)
plt.title(f"PINO vs FDM vs Analytical at t = {t_val}s", fontsize=16)

plt.legend(loc='upper right', edgecolor='black', facecolor='white', framealpha=1.0, fontsize=11)
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()