"""
analysis/degradation.py — Enzymatic degradation observables.

New CATCHY-specific analysis (not in Paper 1):
  - Bond survival fraction S(t)
  - Enzymatic mobility enhancement D_active(C) / D_passive(C)
  - Cleavage rate  dS/dt
"""

import os
import numpy as np


def load_survival(out_dir, sigma_E, interaction):
    """Load S(t) for active run."""
    label = f"sigma{sigma_E:.1f}_{interaction}_active"
    path = os.path.join(out_dir, f"survival_{label}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data["time"], data["survival"]


def load_D_table(out_dir, sigma_list, rho, interaction):
    """
    Return arrays C_vals, D_active, D_passive for plotting enhancement.
    """
    from src.utils import confinement_parameter

    C_vals, D_act, D_pass = [], [], []
    for sigma_E in sigma_list:
        C, _ = confinement_parameter(sigma_E, rho)
        D_a = _load_D(out_dir, sigma_E, interaction, "active")
        D_p = _load_D(out_dir, sigma_E, interaction, "passive")
        if D_a is not None and D_p is not None and D_p > 0:
            C_vals.append(C)
            D_act.append(D_a)
            D_pass.append(D_p)

    return np.array(C_vals), np.array(D_act), np.array(D_pass)


def _load_D(out_dir, sigma_E, interaction, mode):
    label = f"sigma{sigma_E:.1f}_{interaction}_{mode}"
    path = os.path.join(out_dir, f"msd_{label}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return float(data["D"])


def cleavage_rate(time, survival):
    """
    dS/dt estimated by finite differences.
    Returns (t_mid, rate) where rate = -d[S]/dt >= 0.
    """
    t_mid = 0.5 * (time[:-1] + time[1:])
    dt = time[1:] - time[:-1]
    rate = -(survival[1:] - survival[:-1]) / (dt + 1e-30)
    return t_mid, rate
