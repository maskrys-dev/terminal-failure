"""
Linear readout probes for the transient geometry experiments.

Two readout strategies:
    1. Final-state: classify using only z_T
    2. Transient-mean: classify using mean(z_1, ..., z_T)

Both produce matched-dimensionality feature vectors (2*hidden_dim)
to ensure fair comparison — no dimension cheating.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Feature extraction
# =============================================================================

def extract_final_state_features(trajectory):
    """
    Extract features from the final state z_T.
    
    Parameters
    ----------
    trajectory : Tensor [T, batch, hidden_dim] complex
    
    Returns
    -------
    features : ndarray [batch, 2 * hidden_dim]
        Real and imaginary parts concatenated.
    """
    z_T = trajectory[-1]  # [batch, d]
    return torch.cat([z_T.real, z_T.imag], dim=-1).cpu().numpy()


def extract_transient_mean_features(trajectory):
    """
    Extract features from the temporal mean of the trajectory.
    
    Dimensionality matches final-state features exactly (2 * hidden_dim),
    ensuring fair comparison.
    
    Parameters
    ----------
    trajectory : Tensor [T, batch, hidden_dim] complex
    
    Returns
    -------
    features : ndarray [batch, 2 * hidden_dim]
    """
    z_mean = trajectory.mean(dim=0)  # [batch, d]
    return torch.cat([z_mean.real, z_mean.imag], dim=-1).cpu().numpy()


def extract_amplitude_only_features(trajectory):
    """
    Extract amplitude-only features (discard phase).
    Mean amplitude over time per hidden unit.
    
    Returns
    -------
    features : ndarray [batch, hidden_dim]
        Note: lower dimensionality than full features.
    """
    amplitudes = torch.abs(trajectory)  # [T, batch, d]
    return amplitudes.mean(dim=0).cpu().numpy()  # [batch, d]


def extract_phase_only_features(trajectory):
    """
    Extract phase-only features (discard amplitude).
    Circular mean of phases over time, encoded as (cos, sin).
    
    Returns
    -------
    features : ndarray [batch, 2 * hidden_dim]
        Matched dimensionality with full features.
    """
    phases = torch.angle(trajectory)  # [T, batch, d]
    mean_cos = torch.cos(phases).mean(dim=0)  # [batch, d]
    mean_sin = torch.sin(phases).mean(dim=0)  # [batch, d]
    return torch.cat([mean_cos, mean_sin], dim=-1).cpu().numpy()


def extract_timestep_features(trajectory, t):
    """
    Extract features from a single timestep z_t.
    For time-resolved separability analysis.
    
    Parameters
    ----------
    trajectory : Tensor [T, batch, hidden_dim] complex
    t : int
        Timestep index.
    
    Returns
    -------
    features : ndarray [batch, 2 * hidden_dim]
    """
    z_t = trajectory[t]  # [batch, d]
    return torch.cat([z_t.real, z_t.imag], dim=-1).cpu().numpy()


# =============================================================================
# Linear probe
# =============================================================================

class LinearProbe:
    """
    Linear probe using RidgeClassifierCV (closed-form, extremely fast).
    
    RidgeClassifier is equivalent to least-squares classification with
    L2 regularisation. It's a standard linear probe in representation
    learning literature and runs in seconds vs minutes for LogisticRegression.
    """

    def __init__(self, **kwargs):
        from sklearn.linear_model import RidgeClassifierCV
        self.scaler = StandardScaler()
        self.model = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])

    def fit(self, features, labels):
        """
        Parameters
        ----------
        features : ndarray [N, D]
        labels : ndarray [N]
        """
        X = self.scaler.fit_transform(features)
        self.model.fit(X, labels)

    def score(self, features, labels):
        """Returns accuracy on test data."""
        X = self.scaler.transform(features)
        return self.model.score(X, labels)

