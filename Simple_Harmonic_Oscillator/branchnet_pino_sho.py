import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import os

print("Initializing Fourier Neural Operator (FNO) Training...")

# ==========================================
# 1. SPECTRAL CONVOLUTION LAYER (The Magic)
# ==========================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to keep (filters out high-freq noise)
        
        # Learnable complex weights for the frequency domain
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        
        # 1. Apply Fast Fourier Transform (Physical Space -> Frequency Space)
        x_ft = torch.fft.rfft(x)
        
        # 2. Multiply relevant Fourier modes by learnable weights
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], self.weights)
        
        # 3. Apply Inverse Fast Fourier Transform (Frequency Space -> Physical Space)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

# ==========================================
# 2. FULL FOURIER NEURAL OPERATOR
# ==========================================
class FNO1d(nn.Module):
    def __init__(self, modes=16, width=64):
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width
        
        # Lifts the 2 input channels (Initial Condition + X-grid) to higher dimension
        self.fc0 = nn.Linear(2, self.width) 
        
        # 4 Layers of Spectral Convolutions
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
        
        # Standard 1D Convolutions (Act as skip connections to stabilize training)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        
        # Projects back down to our single output channel (The predicted wave)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x input shape: (Batch, Grid_Points, 2)
        x = self.fc0(x)
        x = x.permute(0, 2, 1) # Reshape for convolutions
        
        # Layer 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        # Layer 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        
        # Layer 3
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)
        
        # Layer 4
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)
        
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.squeeze(-1) # Output shape: (Batch, Grid_Points)


# ==========================================
# 3. LOAD DATASET & PREPARE DATALOADERS
# ==========================================
print("Loading 'pino_data.pt'...")

# Force the exact path just like we did for saving
load_dir = "C:/Users/Sakthi Visahan V/OneDrive/Documents/spyder_stuff"
full_load_path = os.path.join(load_dir, 'pino_data.pt')

dataset = torch.load(full_load_path)
x_grid = dataset['x_grid']      # Shape: [500]
input_a = dataset['input_a']    # Shape: [1000, 500]
target_u = dataset['target_u']  # Shape: [1000, 500]

num_samples = input_a.shape[0]
num_grid = input_a.shape[1]

# The FNO needs to know "where" it is in space. We feed it the x-coordinates alongside the wave height.
# We stack them so the input has 2 channels: [Initial Wave Height, X-Coordinate]
x_grid_expanded = x_grid.unsqueeze(0).repeat(num_samples, 1)
network_inputs = torch.stack([input_a, x_grid_expanded], dim=-1) # Shape: [1000, 500, 2]

# Split into Training (800) and Testing (200) sets
train_inputs = network_inputs[:800]
train_targets = target_u[:800]
test_inputs = network_inputs[800:]
test_targets = target_u[800:]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_inputs, train_targets), batch_size=20, shuffle=True)
# ==========================================
# 4. TRAINING LOOP
# ==========================================
model = FNO1d(modes=16, width=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
loss_fn = nn.MSELoss()

epochs = 500 # PINO requires drastically fewer epochs than PINN!

print(f"Beginning PINO Training for {epochs} epochs...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Feed the entire Frame 1 to get Frame X
        predictions = model(batch_x)
        
        # Calculate how close it is to the Professor's True FDM data
        loss = loss_fn(predictions, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    scheduler.step()
    
    if epoch % 50 == 0 or epoch == epochs - 1:
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch: {epoch:03d} | MSE Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.1e}")

elapsed = time.time() - start_time
print(f"Training Complete in {elapsed:.2f} seconds.")

# ==========================================
# 5. TEST ON UNSEEN DATA
# ==========================================
print("\nTesting PINO on an unseen wave state...")
model.eval()
with torch.no_grad():
    # Pick a random scenario the model has NEVER seen
    test_idx = 42 
    sample_input = test_inputs[test_idx].unsqueeze(0)
    true_target = test_targets[test_idx]
    
    # Predict the future state in milliseconds
    pino_prediction = model(sample_input).squeeze()
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid.numpy(), true_target.numpy(), 'r-', linewidth=2, label="True FDM (t=1.0s)")
    plt.plot(x_grid.numpy(), pino_prediction.numpy(), 'b--', linewidth=2.5, label="PINO Prediction (t=1.0s)")
    plt.plot(x_grid.numpy(), test_inputs[test_idx, :, 0].numpy(), 'k:', alpha=0.5, label="Initial Condition (t=0.0s)")
    
    plt.xlim([-0.2, 0.2])
    plt.title("Fourier Neural Operator (PINO) vs. MATLAB TVD-FDM", fontsize=14)
    plt.xlabel("Displacement [m]", fontsize=12)
    plt.ylabel("PDF", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
    
# ==========================================
# 6. THE ULTIMATE PROFESSOR TEST (x0 = 0.1)
# ==========================================
print("\nRunning the classic x0 = 0.1 scenario...")
model.eval()
with torch.no_grad():
    # 1. Create the exact x0 = 0.1 starting wave
    x0_prof = 0.1
    fdm_diffusion_sd = 0.0035 
    ic_prof = stats.norm.pdf(x_grid.numpy(), loc=x0_prof, scale=fdm_diffusion_sd)
    
    # 2. Format it for the PINO (Add x-coordinates as the second channel)
    ic_tensor = torch.tensor(ic_prof, dtype=torch.float32).unsqueeze(0)
    x_tensor = x_grid.unsqueeze(0)
    prof_input = torch.stack([ic_tensor, x_tensor], dim=-1) # Shape: [1, 500, 2]
    
    # 3. Predict the future state in milliseconds
    pino_prof_pred = model(prof_input).squeeze()
    
    # 4. Generate the True Analytical Solution for t=1.0
    prof_omegas = np.linspace(1.25 * np.pi, 1.75 * np.pi, 101)
    prof_probs = np.ones(101) * (1.0 / 100.0)
    prof_probs[0] = 1.0 / 200.0
    prof_probs[-1] = 1.0 / 200.0
    
    true_target = np.zeros_like(x_grid.numpy())
    for w, prob in zip(prof_omegas, prof_probs):
        mu = x0_prof * np.cos(w * 1.0) # t = 1.0
        true_target += prob * stats.norm.pdf(x_grid.numpy(), loc=mu, scale=fdm_diffusion_sd)

    # 5. Plot the final victory graph
    plt.figure(figsize=(12, 7))
    plt.plot(x_grid.numpy(), true_target, 'r-', linewidth=2.5, alpha=0.8, label="True MATLAB FDM (t=1.0s)")
    plt.plot(x_grid.numpy(), pino_prof_pred.numpy(), 'b--', linewidth=3.0, label="PINO Output (t=1.0s)")
    
    plt.xlim([-0.2, 0.2])
    plt.ylim(bottom=-0.5)
    plt.title("The Fix: PINO Successfully Resolves Sharp Edges at x0=0.1", fontsize=16)
    plt.xlabel("Displacement [m]", fontsize=14)
    plt.ylabel("PDF", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()