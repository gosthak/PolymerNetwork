"""
analysis/non_gaussian.py — Non-Gaussian parameter α₂(t).

Paper 1, eq.(9):
    α₂(t) = (3/5) * <r⁴(t)> / <r²(t)>² − 1

α₂ = 0 for a Gaussian process; large α₂ signals heterogeneous dynamics.
Paper 1 shows α₂ comparable to supercooled liquids for C ≳ 3.
"""

import numpy as np


def non_gaussian_parameter(positions, L):
    """
    Compute α₂(t) for all lag times up to n_frames//2.

    Parameters
    ----------
    positions : np.ndarray (n_frames, N_E, 3) — enzyme positions (wrapped OK)
    L : float

    Returns
    -------
    lag_idx : np.ndarray (max_lag,)
    alpha2  : np.ndarray (max_lag,)
    """
    from src.utils import _unwrap
    n_frames, N_E, _ = positions.shape
    max_lag = n_frames // 2
    unwrapped = _unwrap(positions, L)

    alpha2 = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        dr = unwrapped[lag:] - unwrapped[:-lag]      # (n_frames-lag, N_E, 3)
        r2 = np.sum(dr ** 2, axis=-1)                # (n_frames-lag, N_E)
        r4 = r2 ** 2

        mean_r2 = np.mean(r2)
        mean_r4 = np.mean(r4)

        if mean_r2 > 0:
            alpha2[lag - 1] = (3.0 / 5.0) * mean_r4 / (mean_r2 ** 2) - 1.0
        else:
            alpha2[lag - 1] = 0.0

    return np.arange(1, max_lag + 1), alpha2


def alpha2_peak(lag_indices, alpha2):
    """Return (peak_value, peak_lag_index)."""
    idx_max = np.argmax(alpha2)
    return alpha2[idx_max], lag_indices[idx_max]
