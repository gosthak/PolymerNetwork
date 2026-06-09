"""
analysis/msd.py — MSD, diffusion coefficient D(C), subdiffusion exponent β(t).

Reproduces Paper 1 Figures 5, 6, 7.
Also provides Cai-Rubinstein and Dell-Schweizer theoretical curves (Paper 1 eqs 15-16).
"""

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Theory: Paper 1 eq.(15) — Cai, Panyukov & Rubinstein (2011)
# ---------------------------------------------------------------------------

def D_cai_rubinstein(C, D0, b, nu=0.588):
    """
    D_N/D0 = exp(−b * C^(3nu/(3nu-1)))  with nu=0.588 (self-avoiding)
    b is a non-universal prefactor fitted to simulation data.
    """
    exponent = 3.0 * nu / (3.0 * nu - 1.0)   # ≈ 1.28
    return D0 * np.exp(-b * C ** exponent)


# ---------------------------------------------------------------------------
# Theory: Paper 1 eq.(16) — Dell & Schweizer (2014)
# ---------------------------------------------------------------------------

def D_dell_schweizer(C, D0, A):
    """
    D_N/D0 = A * C^(-2) * exp(−C^2 / 2)   (simplified Gaussian form)
    """
    return D0 * A * C ** (-2) * np.exp(-C ** 2 / 2.0)


# ---------------------------------------------------------------------------
# Local subdiffusion exponent β(t) = d log MSD / d log t
# ---------------------------------------------------------------------------

def compute_beta(time, msd, window=5):
    """
    β(t) = d[log MSD] / d[log t]  — local slope in log-log space.
    Smoothed with a rolling window derivative.
    """
    log_t = np.log(time)
    log_msd = np.log(np.maximum(msd, 1e-20))
    beta = np.gradient(log_msd, log_t)
    # Smooth with rolling mean
    from numpy import convolve, ones
    kernel = ones(window) / window
    beta_smooth = convolve(beta, kernel, mode="same")
    return beta_smooth


# ---------------------------------------------------------------------------
# Fit D(C) to both theories
# ---------------------------------------------------------------------------

def fit_theories(C_vals, D_vals):
    """
    Fit Cai-Rubinstein and Dell-Schweizer to simulation D(C) data.

    Returns dict with fitted parameters and chi² for each model.
    """
    results = {}

    # Cai-Rubinstein
    try:
        mask = np.isfinite(D_vals) & (D_vals > 0) & (C_vals > 0)
        popt_cr, _ = curve_fit(
            D_cai_rubinstein, C_vals[mask], D_vals[mask],
            p0=[D_vals[mask].max(), 1.0], maxfev=5000
        )
        D_cr = D_cai_rubinstein(C_vals[mask], *popt_cr)
        chi2_cr = np.mean((np.log(D_cr) - np.log(D_vals[mask])) ** 2)
        results["cai_rubinstein"] = {
            "D0": popt_cr[0], "b": popt_cr[1], "chi2": chi2_cr
        }
    except Exception as e:
        results["cai_rubinstein"] = {"error": str(e)}

    # Dell-Schweizer
    try:
        mask = np.isfinite(D_vals) & (D_vals > 0) & (C_vals > 1.0)
        popt_ds, _ = curve_fit(
            D_dell_schweizer, C_vals[mask], D_vals[mask],
            p0=[D_vals[mask].max() * 4, 1.0], maxfev=5000
        )
        D_ds = D_dell_schweizer(C_vals[mask], *popt_ds)
        chi2_ds = np.mean((np.log(D_ds) - np.log(D_vals[mask])) ** 2)
        results["dell_schweizer"] = {
            "D0": popt_ds[0], "A": popt_ds[1], "chi2": chi2_ds
        }
    except Exception as e:
        results["dell_schweizer"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Load all MSD results and assemble D(C) table
# ---------------------------------------------------------------------------

def load_results_table(out_dir, sigma_list, rho, interaction, modes=("active", "passive")):
    """
    Load all msd_*.npz files and return a dict:
        results[mode][sigma_E] = {'time', 'msd', 'D', 'C'}
    """
    import os
    from src.utils import confinement_parameter

    results = {m: {} for m in modes}
    for mode in modes:
        for sigma_E in sigma_list:
            label = f"sigma{sigma_E:.1f}_{interaction}_{mode}"
            path = os.path.join(out_dir, f"msd_{label}.npz")
            if not os.path.exists(path):
                continue
            data = np.load(path)
            C, _ = confinement_parameter(sigma_E, rho)
            results[mode][sigma_E] = {
                "time": data["time"],
                "msd": data["msd"],
                "D": float(data["D"]),
                "C": C,
            }
    return results
