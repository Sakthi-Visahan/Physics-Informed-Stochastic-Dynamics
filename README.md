# Physics-Informed Stochastic Dynamics

### Overview
This repository provides a codebase for Gradient-Enhanced Physics-Informed Neural Networks (gPINNs) and Physics-Informed Neural Operators (PINO) designed to resolve partial differential equations (PDEs) in structural mechanics and stochastic dynamic systems.

### Repository Structure
* **`Simple_Harmonic_Oscillator/`**: Implementations of baseline PINNs, gPINNs, and operator architectures applied to canonical oscillatory systems and differential constraints.
* **`Earthquake_Dynamics/`**: Models and data pipelines analyzing structural peak responses and time-history dynamics subjected to stochastic earthquake excitations.

### Key Technical Focus
* **Physics-Informed Loss Formulations:** Enforces differential equation constraints directly within the neural network loss function.
* **Gradient Enhancement (gPINN):** Incorporates spatial/temporal derivatives of PDE residuals into training passes to accelerate optimization convergence.
* **Neural Operators (PINO):** Evaluates solution operators across varying initial conditions and parametric forcing inputs.

### Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **Scientific Computing:** NumPy, SciPy, Pandas
* **Visualization:** Matplotlib

### How to Run
Ensure the core scientific dependencies are installed:
```bash
pip install torch numpy scipy pandas matplotlib
