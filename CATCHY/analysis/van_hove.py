"""
analysis/van_hove.py — Van Hove correlation functions.

Reproduces Paper 1 Figures 10–11.

G_s(r, t) = (1/N_E) Σ_i <δ(r - |r_i(t) - r_i(0)|)>
G_d(r, t) = distinct part (cross-correlations)

Hopping is identified when G_s shows a secondary peak at r ~ ξ (mesh size).
"""

import numpy as np


def self_van_hove(positions, L, lag, r_bins):
    """
    Compute the self part of the van Hove function G_s(r, t=lag).

    Parameters
    ----------
    positions : np.ndarray (n_frames, N_E, 3) — enzyme positions (unwrapped)
    L : float  box length (for display only; positions should be unwrapped)
    lag : int  time lag in frames
    r_bins : np.ndarray  bin edges for r

    Returns
    -------
    r_centers : np.ndarray
    Gs : np.ndarray  (normalized)
    """
    n_frames, N_E, _ = positions.shape
    n_origins = n_frames - lag

    dr_all = []
    for t0 in range(n_origins):
        dr = positions[t0 + lag] - positions[t0]
        r = np.linalg.norm(dr, axis=-1)   # (N_E,)
        dr_all.append(r)

    dr_all = np.concatenate(dr_all)   # (n_origins * N_E,)

    counts, edges = np.histogram(dr_all, bins=r_bins)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    dr_bin = edges[1:] - edges[:-1]

    # Normalize: G_s(r) = count / (4π r² dr * n_total)
    vol = 4.0 * np.pi * r_centers ** 2 * dr_bin
    n_total = len(dr_all)
    Gs = counts / (vol * n_total + 1e-30)

    # Gaussian reference: G_s^Gauss(r,t) = (4π σ²_r t)^(-3/2) exp(-r²/4σ²_r t)
    # σ²_r = MSD(t)/6 from the data
    msd_lag = np.mean(dr_all ** 2)
    sigma2 = msd_lag / 6.0
    if sigma2 > 0:
        Gs_gauss = (
            (4.0 * np.pi * sigma2) ** (-1.5)
            * np.exp(-r_centers ** 2 / (4.0 * sigma2))
        )
    else:
        Gs_gauss = np.zeros_like(r_centers)

    return r_centers, Gs, Gs_gauss


def distinct_van_hove(mono_positions, enzyme_positions, L, lag, r_bins):
    """
    Distinct part G_d(r, t): enzyme-monomer cross-correlations.
    G_d(r, t) = (1/(N_E N_m)) Σ_{i≠j} <δ(r - |r_i(t) - r_j(0)|)>
    """
    n_frames = mono_positions.shape[0]
    N_m = mono_positions.shape[1]
    N_E = enzyme_positions.shape[1]

    n_origins = min(20, n_frames - lag)   # limit for performance
    origins = np.linspace(0, n_frames - lag - 1, n_origins, dtype=int)

    dr_all = []
    for t0 in origins:
        # Enzyme positions at t0 + lag, monomer positions at t0
        ep = enzyme_positions[t0 + lag]   # (N_E, 3)
        mp = mono_positions[t0]           # (N_m, 3)
        # Pairwise distances with PBC
        for ie in range(N_E):
            dr = mp - ep[ie]
            dr -= L * np.round(dr / L)
            r = np.linalg.norm(dr, axis=-1)
            dr_all.append(r)

    dr_all = np.concatenate(dr_all)

    counts, edges = np.histogram(dr_all, bins=r_bins)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    dr_bin = edges[1:] - edges[:-1]
    vol = 4.0 * np.pi * r_centers ** 2 * dr_bin
    n_total = len(dr_all)
    Gd = counts / (vol * n_total + 1e-30)

    return r_centers, Gd


def delta_gs(positions, L, lag_list, r_bins):
    """
    Compute the displacement field Δ(r, t) = G_s(r,t) - G_s^Gauss(r,t)
    for multiple lag times. Non-zero Δ indicates non-Gaussian dynamics (hopping).
    """
    from src.utils import _unwrap
    n_frames, N_E, _ = positions.shape
    unwrapped = _unwrap(positions, L)

    results = {}
    for lag in lag_list:
        r, Gs, Gs_gauss = self_van_hove(unwrapped, L, lag, r_bins)
        results[lag] = {"r": r, "Gs": Gs, "Gs_gauss": Gs_gauss, "delta": Gs - Gs_gauss}

    return results
