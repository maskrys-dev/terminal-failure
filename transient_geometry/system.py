"""
Complex-valued dynamical system for transient geometry experiments.

Core recurrence:
    z_{t+1} = modReLU(W @ z_t + U @ x + b + noise)

The system is NOT trained. It is a fixed dynamical substrate.
Learning happens only in the downstream linear probes.
"""

import torch
import torch.nn as nn
import numpy as np


class ComplexDynamicalSystem(nn.Module):
    """
    Discrete-time complex-valued recurrent dynamical system.
    
    Parameters
    ----------
    input_dim : int
        Dimension of real-valued input (e.g. 784 for MNIST).
    hidden_dim : int
        Dimension of complex state z ∈ C^d.
    spectral_radius : float
        Target spectral radius of recurrent weight W.
        < 1.0 → contractive, ≈ 1.0 → edge of stability, > 1.0 → unstable.
    noise_std : float
        Standard deviation of complex Gaussian noise injected per step.
    modrelu_bias : float
        Bias parameter b for modReLU. Negative values create a dead zone
        that helps prevent unbounded growth.
    seed : int
        Random seed for weight initialisation. Same seed + different
        spectral_radius produces the same base W structure, just rescaled.
    """

    def __init__(self, input_dim, hidden_dim, spectral_radius=0.9,
                 noise_std=0.0, modrelu_bias=-0.5, seed=42):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.modrelu_bias = modrelu_bias

        rng = torch.Generator()
        rng.manual_seed(seed)

        # --- Recurrent weight W ∈ C^{d×d} ---
        W_real = torch.randn(hidden_dim, hidden_dim, generator=rng) / np.sqrt(hidden_dim)
        W_imag = torch.randn(hidden_dim, hidden_dim, generator=rng) / np.sqrt(hidden_dim)
        W = torch.complex(W_real, W_imag)

        # Rescale to target spectral radius
        eigvals = torch.linalg.eigvals(W)
        current_sr = torch.max(torch.abs(eigvals)).item()
        if current_sr > 1e-8:
            W = W * (spectral_radius / current_sr)

        self.register_buffer('W', W)

        # --- Input projection U ∈ C^{d×input_dim} ---
        U_real = torch.randn(hidden_dim, input_dim, generator=rng) / np.sqrt(input_dim)
        U_imag = torch.randn(hidden_dim, input_dim, generator=rng) / np.sqrt(input_dim)
        self.register_buffer('U', torch.complex(U_real, U_imag))

        # --- Bias b ∈ C^d ---
        self.register_buffer('b', torch.zeros(hidden_dim, dtype=torch.complex64))

        # Store nominal spectral radius for logging
        self.nominal_spectral_radius = spectral_radius

    def modrelu(self, z):
        """
        modReLU(z; b) = max(|z| + b, 0) · z / |z|
        
        Phase-preserving nonlinearity. When b < 0, creates a dead zone
        for small-magnitude activations while preserving phase structure
        for large-magnitude activations.
        """
        mag = torch.abs(z)
        safe_mag = torch.clamp(mag, min=1e-8)
        activated = torch.clamp(mag + self.modrelu_bias, min=0.0)
        return activated * (z / safe_mag)

    @torch.no_grad()
    def forward(self, x, T=40):
        """
        Roll out the dynamical system for T steps.
        
        Parameters
        ----------
        x : Tensor [batch, input_dim]
            Real-valued input (e.g. flattened MNIST image).
        T : int
            Number of rollout timesteps.
            
        Returns
        -------
        trajectory : Tensor [T, batch, hidden_dim]
            Complex-valued state trajectory.
        """
        batch = x.shape[0]
        device = x.device

        # Constant input drive: U @ x (same at every timestep)
        x_complex = x.to(torch.complex64)
        Ux = torch.matmul(x_complex, self.U.T)  # [batch, hidden_dim]

        # Initial state z_0 = 0
        z = torch.zeros(batch, self.hidden_dim, dtype=torch.complex64, device=device)

        trajectory = torch.empty(T, batch, self.hidden_dim, dtype=torch.complex64, device=device)
        for t in range(T):
            pre = torch.matmul(z, self.W.T) + Ux + self.b

            if self.noise_std > 0:
                noise = torch.complex(
                    torch.randn_like(pre.real) * self.noise_std,
                    torch.randn_like(pre.imag) * self.noise_std
                )
                pre = pre + noise

            z = self.modrelu(pre)
            # Magnitude clip for numerical safety (preserves phase)
            mag = torch.abs(z)
            mask = mag > 50.0
            if mask.any():
                z = torch.where(mask, z * (50.0 / mag.clamp(min=1e-8)), z)
            trajectory[t] = z

        return trajectory

    @torch.no_grad()
    def forward_features(self, x, T=40):
        """
        Fast path: compute final-state and transient-mean features directly
        without storing the full trajectory tensor. Much faster and lighter.
        
        Returns
        -------
        final_features : Tensor [batch, 2*hidden_dim] (real, on CPU)
        transient_mean_features : Tensor [batch, 2*hidden_dim] (real, on CPU)
        """
        batch = x.shape[0]
        device = x.device

        x_complex = x.to(torch.complex64)
        Ux = torch.matmul(x_complex, self.U.T)

        z = torch.zeros(batch, self.hidden_dim, dtype=torch.complex64, device=device)
        z_sum = torch.zeros_like(z)  # Running sum for mean

        for t in range(T):
            pre = torch.matmul(z, self.W.T) + Ux + self.b
            if self.noise_std > 0:
                noise = torch.complex(
                    torch.randn_like(pre.real) * self.noise_std,
                    torch.randn_like(pre.imag) * self.noise_std
                )
                pre = pre + noise
            z = self.modrelu(pre)
            # Magnitude clip for numerical safety (preserves phase)
            mag = torch.abs(z)
            mask = mag > 50.0
            if mask.any():
                z = torch.where(mask, z * (50.0 / mag.clamp(min=1e-8)), z)
            z_sum += z

        z_mean = z_sum / T

        # Convert to real features: [real, imag] concatenated
        final_features = torch.cat([z.real, z.imag], dim=-1).cpu()
        transient_features = torch.cat([z_mean.real, z_mean.imag], dim=-1).cpu()

        return final_features.numpy(), transient_features.numpy()

    @torch.no_grad()
    def forward_all_readouts(self, x, T, early_K=10):
        """
        Single forward pass producing ALL readout features + norm trajectory.
        
        Returns
        -------
        result : dict with keys:
            'final' : ndarray [batch, 2d] — z_T features
            'transient_mean' : ndarray [batch, 2d] — mean(z_1..z_T)
            'early_window' : ndarray [batch, 2d] — mean(z_1..z_K)
            'norms' : ndarray [T] — mean batch norm at each timestep
            'per_timestep' : list of ndarray [batch, 2d] — features at each t
        """
        batch = x.shape[0]
        device = x.device
        
        x_complex = x.to(torch.complex64)
        Ux = torch.matmul(x_complex, self.U.T)
        
        z = torch.zeros(batch, self.hidden_dim, dtype=torch.complex64, device=device)
        z_sum = torch.zeros_like(z)
        z_early_sum = torch.zeros_like(z)
        norms = np.zeros(T)
        per_timestep = []
        
        for t in range(T):
            pre = torch.matmul(z, self.W.T) + Ux + self.b
            if self.noise_std > 0:
                noise = torch.complex(
                    torch.randn_like(pre.real) * self.noise_std,
                    torch.randn_like(pre.imag) * self.noise_std
                )
                pre = pre + noise
            z = self.modrelu(pre)
            mag = torch.abs(z)
            mask = mag > 50.0
            if mask.any():
                z = torch.where(mask, z * (50.0 / mag.clamp(min=1e-8)), z)
            
            z_sum += z
            if t < early_K:
                z_early_sum += z
            norms[t] = torch.abs(z).mean().item()
            
            feat = torch.cat([z.real, z.imag], dim=-1).cpu().numpy()
            per_timestep.append(feat)
        
        z_mean = z_sum / T
        z_early_mean = z_early_sum / early_K
        
        return {
            'final': torch.cat([z.real, z.imag], dim=-1).cpu().numpy(),
            'transient_mean': torch.cat([z_mean.real, z_mean.imag], dim=-1).cpu().numpy(),
            'early_window': torch.cat([z_early_mean.real, z_early_mean.imag], dim=-1).cpu().numpy(),
            'norms': norms,
            'per_timestep': per_timestep,
        }


class RealESN(nn.Module):
    """
    Real-valued Echo State Network with tanh nonlinearity.
    
    Mirrors ComplexDynamicalSystem interface for cross-system comparison.
    
    Recurrence:
        h_{t+1} = tanh(W @ h_t + U @ x + b)
    
    Feature readout: h_t ∈ R^d (no real/imag split needed).
    """

    def __init__(self, input_dim, hidden_dim, spectral_radius=0.9,
                 noise_std=0.0, leak_rate=1.0, seed=42):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.leak_rate = leak_rate

        rng = torch.Generator()
        rng.manual_seed(seed)

        # Recurrent weight W ∈ R^{d×d}
        W = torch.randn(hidden_dim, hidden_dim, generator=rng) / np.sqrt(hidden_dim)
        eigvals = torch.linalg.eigvals(W)
        current_sr = torch.max(torch.abs(eigvals)).item()
        if current_sr > 1e-8:
            W = W * (spectral_radius / current_sr)
        self.register_buffer('W', W)

        # Input projection U ∈ R^{d×input_dim}
        U = torch.randn(hidden_dim, input_dim, generator=rng) / np.sqrt(input_dim)
        self.register_buffer('U', U)

        # Bias
        self.register_buffer('b', torch.zeros(hidden_dim))

        self.nominal_spectral_radius = spectral_radius

    @torch.no_grad()
    def forward(self, x, T=40):
        """
        Roll out ESN for T steps.
        
        Returns trajectory [T, batch, hidden_dim] (real-valued).
        """
        batch = x.shape[0]
        device = x.device

        Ux = torch.matmul(x, self.U.T)  # [batch, hidden_dim]
        h = torch.zeros(batch, self.hidden_dim, device=device)

        trajectory = torch.empty(T, batch, self.hidden_dim, device=device)
        for t in range(T):
            pre = torch.matmul(h, self.W.T) + Ux + self.b
            if self.noise_std > 0:
                pre = pre + torch.randn_like(pre) * self.noise_std
            h_new = torch.tanh(pre)
            h = self.leak_rate * h_new + (1 - self.leak_rate) * h
            trajectory[t] = h

        return trajectory

    @torch.no_grad()
    def forward_all_readouts(self, x, T, early_K=10):
        """Same interface as ComplexDynamicalSystem.forward_all_readouts."""
        batch = x.shape[0]
        device = x.device

        Ux = torch.matmul(x, self.U.T)
        h = torch.zeros(batch, self.hidden_dim, device=device)
        h_sum = torch.zeros_like(h)
        h_early_sum = torch.zeros_like(h)
        norms = np.zeros(T)
        per_timestep = []

        for t in range(T):
            pre = torch.matmul(h, self.W.T) + Ux + self.b
            if self.noise_std > 0:
                pre = pre + torch.randn_like(pre) * self.noise_std
            h_new = torch.tanh(pre)
            h = self.leak_rate * h_new + (1 - self.leak_rate) * h

            h_sum += h
            if t < early_K:
                h_early_sum += h
            norms[t] = h.norm(dim=-1).mean().item()
            per_timestep.append(h.cpu().numpy())

        h_mean = h_sum / T
        h_early = h_early_sum / early_K

        return {
            'final': h.cpu().numpy(),
            'transient_mean': h_mean.cpu().numpy(),
            'early_window': h_early.cpu().numpy(),
            'norms': norms,
            'per_timestep': per_timestep,
        }


class LinearDynamicalSystem(nn.Module):
    """
    Pure linear dynamical system (no nonlinearity).
    
    Recurrence:
        z_{t+1} = W @ z_t + U @ x
    
    Tests whether the grace period arises from spectral instability
    alone, independent of nonlinear expressivity.
    """

    def __init__(self, input_dim, hidden_dim, spectral_radius=0.9,
                 noise_std=0.0, seed=42):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std

        rng = torch.Generator()
        rng.manual_seed(seed)

        W = torch.randn(hidden_dim, hidden_dim, generator=rng) / np.sqrt(hidden_dim)
        eigvals = torch.linalg.eigvals(W)
        current_sr = torch.max(torch.abs(eigvals)).item()
        if current_sr > 1e-8:
            W = W * (spectral_radius / current_sr)
        self.register_buffer('W', W)

        U = torch.randn(hidden_dim, input_dim, generator=rng) / np.sqrt(input_dim)
        self.register_buffer('U', U)

        self.nominal_spectral_radius = spectral_radius

    @torch.no_grad()
    def forward(self, x, T=40):
        batch = x.shape[0]
        device = x.device

        Ux = torch.matmul(x, self.U.T)
        z = torch.zeros(batch, self.hidden_dim, device=device)

        trajectory = torch.empty(T, batch, self.hidden_dim, device=device)
        for t in range(T):
            z = torch.matmul(z, self.W.T) + Ux
            if self.noise_std > 0:
                z = z + torch.randn_like(z) * self.noise_std
            # Magnitude clip for numerical safety
            znorm = z.norm(dim=-1, keepdim=True)
            mask = znorm > 100.0
            if mask.any():
                z = torch.where(mask, z * (100.0 / znorm.clamp(min=1e-8)), z)
            trajectory[t] = z

        return trajectory

    @torch.no_grad()
    def forward_all_readouts(self, x, T, early_K=10):
        batch = x.shape[0]
        device = x.device

        Ux = torch.matmul(x, self.U.T)
        z = torch.zeros(batch, self.hidden_dim, device=device)
        z_sum = torch.zeros_like(z)
        z_early_sum = torch.zeros_like(z)
        norms = np.zeros(T)
        per_timestep = []

        for t in range(T):
            z = torch.matmul(z, self.W.T) + Ux
            if self.noise_std > 0:
                z = z + torch.randn_like(z) * self.noise_std
            znorm = z.norm(dim=-1, keepdim=True)
            mask = znorm > 100.0
            if mask.any():
                z = torch.where(mask, z * (100.0 / znorm.clamp(min=1e-8)), z)

            z_sum += z
            if t < early_K:
                z_early_sum += z
            norms[t] = z.norm(dim=-1).mean().item()
            per_timestep.append(z.cpu().numpy())

        z_mean = z_sum / T
        z_early = z_early_sum / early_K

        return {
            'final': z.cpu().numpy(),
            'transient_mean': z_mean.cpu().numpy(),
            'early_window': z_early.cpu().numpy(),
            'norms': norms,
            'per_timestep': per_timestep,
        }


def compute_empirical_regime_descriptors(system, x_batch, T=40, epsilon=0.01):
    """
    Compute empirical regime descriptors for a batch of inputs.
    
    These measure the *actual* dynamical behaviour of the system, not just
    the nominal spectral radius.
    
    Parameters
    ----------
    system : ComplexDynamicalSystem
    x_batch : Tensor [batch, input_dim]
    T : int
    epsilon : float
        Perturbation magnitude for amplification measurement.
    
    Returns
    -------
    dict with keys:
        'perturbation_amplification' : float
            Mean ratio of output divergence to input perturbation.
        'mean_final_norm' : float
            Mean |z_T| across batch and hidden dims.
        'norm_trajectory' : ndarray [T]
            Mean state norm at each timestep.
        'phase_coherence' : float
            Mean inter-unit phase coherence at final timestep.
    """
    device = x_batch.device

    # --- Original trajectory ---
    traj = system(x_batch, T)  # [T, batch, d]

    # --- Perturbed trajectory ---
    perturbation = torch.randn_like(x_batch) * epsilon
    traj_pert = system(x_batch + perturbation, T)  # [T, batch, d]

    # --- Perturbation amplification ---
    # Divergence in state space at final time
    state_diff = traj[-1] - traj_pert[-1]  # [batch, d]
    state_divergence = torch.abs(state_diff).mean(dim=-1)  # [batch]
    # Input perturbation magnitude
    input_pert_mag = torch.abs(perturbation).mean(dim=-1)  # [batch]
    amp_factor = (state_divergence / input_pert_mag.clamp(min=1e-8)).mean().item()

    # --- State norm trajectory ---
    norms = torch.abs(traj).mean(dim=(1, 2))  # [T] mean over batch and hidden dim
    norm_trajectory = norms.cpu().numpy()
    mean_final_norm = torch.abs(traj[-1]).mean().item()

    # --- Phase coherence ---
    # Mean resultant length of phases across hidden units at final time
    # High coherence → units have aligned phases
    # Low coherence → phases are scattered
    phases = torch.angle(traj[-1])  # [batch, d]
    # For each sample, compute coherence across hidden units
    coherence_per_sample = torch.abs(
        torch.mean(torch.exp(1j * phases), dim=-1)
    )  # [batch]
    phase_coherence = coherence_per_sample.mean().item()

    return {
        'perturbation_amplification': amp_factor,
        'mean_final_norm': mean_final_norm,
        'norm_trajectory': norm_trajectory,
        'phase_coherence': phase_coherence,
    }
