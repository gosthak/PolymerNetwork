#!/usr/bin/env python3
"""
plot_all.py — Generate all CATCHY figures.

Figures produced (saved to {out_dir}/figs/):
    fig_A_msd.pdf         — MSD(t) for all sigma_E, passive (Paper 1 Fig 5)
    fig_B_diffusion.pdf   — D(C): passive + active, theory fits (Paper 1 Fig 6)
    fig_C_beta.pdf        — β(t) subdiffusion exponent (Paper 1 Fig 7)
    fig_D_vanhove.pdf     — G_s(r,t) self van Hove (Paper 1 Fig 10)
    fig_E_nongaussian.pdf — α₂(t) (Paper 1 Fig 12)
    fig_F_survival.pdf    — Bond survival S(t) (new CATCHY)
    fig_G_enhancement.pdf — D_active/D_passive vs C (new CATCHY, key result)

Usage:
    python plot_all.py --config ../configs/default.yaml
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, confinement_parameter, LAMBDA_REF

# Paper-quality style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "lines.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, 8))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/default.yaml")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fig A — MSD(t), passive mode
# ---------------------------------------------------------------------------

def fig_A_msd(out_dir, sigma_list, rho, interaction, dt, save_interval, fig_dir):
    fig, ax = plt.subplots(figsize=(5, 4))

    for i, sigma_E in enumerate(sigma_list):
        C, _ = confinement_parameter(sigma_E, rho)
        label = f"sigma{sigma_E:.1f}_{interaction}_passive"
        path = os.path.join(out_dir, f"msd_{label}.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        t, msd = data["time"], data["msd"]
        ax.loglog(t, msd, color=COLORS[i % len(COLORS)],
                  label=f"C={C:.2f}")

    # Ballistic t² and diffusive t references
    t_ref = np.logspace(-1, 4, 100)
    ax.loglog(t_ref, 0.5 * t_ref ** 2, "k--", lw=0.8, alpha=0.5, label=r"$\sim t^2$")
    ax.loglog(t_ref, 3.0 * t_ref, "k:", lw=0.8, alpha=0.5, label=r"$\sim t$")

    ax.set_xlabel(r"$t$ [LJ units]")
    ax.set_ylabel(r"MSD$(t)$ [$\sigma_m^2$]")
    ax.set_title("Enzyme MSD — passive (Paper 1 Fig. 5 equivalent)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_A_msd.pdf"))
    plt.close(fig)
    print("  Saved fig_A_msd.pdf")


# ---------------------------------------------------------------------------
# Fig B — D(C): passive + active + theory fits
# ---------------------------------------------------------------------------

def fig_B_diffusion(out_dir, sigma_list, rho, interaction, fig_dir):
    from analysis.msd import fit_theories, D_cai_rubinstein, D_dell_schweizer
    from analysis.degradation import load_D_table

    C_all, D_act, D_pass = load_D_table(out_dir, sigma_list, rho, interaction)

    fig, ax = plt.subplots(figsize=(5, 4))

    if len(C_all) > 0:
        ax.semilogy(C_all, D_pass, "o-", color="steelblue",
                    label="Passive (k_cat=0)", markerfacecolor="white")
        ax.semilogy(C_all, D_act, "s-", color="firebrick",
                    label="Active (enzymatic)")

        # Theory fits — passive only
        fits = fit_theories(C_all, D_pass)
        C_th = np.linspace(C_all.min(), C_all.max() * 1.1, 200)
        if "cai_rubinstein" in fits and "error" not in fits["cai_rubinstein"]:
            cr = fits["cai_rubinstein"]
            ax.semilogy(C_th, D_cai_rubinstein(C_th, cr["D0"], cr["b"]),
                        "b--", lw=1, label="Cai–Rubinstein (eq.15)")
        if "dell_schweizer" in fits and "error" not in fits["dell_schweizer"]:
            ds = fits["dell_schweizer"]
            ax.semilogy(C_th, D_dell_schweizer(C_th, ds["D0"], ds["A"]),
                        "g:", lw=1, label="Dell–Schweizer (eq.16)")

    ax.set_xlabel(r"Confinement $C = \sigma_E / \lambda$")
    ax.set_ylabel(r"$D_E$ [LJ units]")
    ax.set_title(r"$D(C)$ — Paper 1 Fig. 6 + enzymatic enhancement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_B_diffusion.pdf"))
    plt.close(fig)
    print("  Saved fig_B_diffusion.pdf")


# ---------------------------------------------------------------------------
# Fig C — β(t) subdiffusion exponent
# ---------------------------------------------------------------------------

def fig_C_beta(out_dir, sigma_list, rho, interaction, dt, save_interval, fig_dir):
    from analysis.msd import compute_beta

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, sigma_E in enumerate(sigma_list):
        C, _ = confinement_parameter(sigma_E, rho)
        label = f"sigma{sigma_E:.1f}_{interaction}_passive"
        path = os.path.join(out_dir, f"msd_{label}.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        t, msd = data["time"], data["msd"]
        beta = compute_beta(t, msd)
        ax.semilogx(t, beta, color=COLORS[i % len(COLORS)], label=f"C={C:.2f}")

    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6, label="β=1 (diffusive)")
    ax.axhline(2.0, color="gray", lw=0.8, ls=":", alpha=0.6, label="β=2 (ballistic)")
    ax.set_xlabel(r"$t$ [LJ units]")
    ax.set_ylabel(r"$\beta(t) = d\log\mathrm{MSD}/d\log t$")
    ax.set_title("Subdiffusion exponent β(t)")
    ax.set_ylim(0, 2.5)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_C_beta.pdf"))
    plt.close(fig)
    print("  Saved fig_C_beta.pdf")


# ---------------------------------------------------------------------------
# Fig D — Self van Hove G_s(r, t)
# ---------------------------------------------------------------------------

def fig_D_vanhove(out_dir, sigma_list, rho, interaction, fig_dir):
    import h5py
    from analysis.van_hove import self_van_hove
    from src.utils import _unwrap

    fig, axes = plt.subplots(1, min(3, len(sigma_list)),
                             figsize=(4 * min(3, len(sigma_list)), 4),
                             sharey=False)
    if len(sigma_list) == 1:
        axes = [axes]

    r_bins = np.linspace(0, 20, 120)
    lag_fracs = [0.05, 0.2, 0.5]   # fractions of trajectory length

    for ix, sigma_E in enumerate(sigma_list[:3]):
        C, _ = confinement_parameter(sigma_E, rho)
        label = f"sigma{sigma_E:.1f}_{interaction}_passive"
        traj_path = os.path.join(out_dir, f"traj_{label}.h5")
        if not os.path.exists(traj_path):
            continue
        ax = axes[ix]
        with h5py.File(traj_path, "r") as f:
            N_m = int(f.attrs["N_m"])
            N_E = int(f.attrs["N_E"])
            L = float(f.attrs["L"])
            epos = f["enzyme_positions"][:]   # (n_frames, N_E, 3)

        n_frames = epos.shape[0]
        unwrapped = _unwrap(epos, L)
        colors_lag = plt.cm.plasma(np.linspace(0.1, 0.9, len(lag_fracs)))

        for il, frac in enumerate(lag_fracs):
            lag = max(1, int(frac * n_frames))
            r, Gs, Gs_gauss = self_van_hove(unwrapped, L, lag, r_bins)
            t_label = f"lag={lag}"
            ax.semilogy(r, Gs, color=colors_lag[il], label=t_label, lw=1.2)
            ax.semilogy(r, Gs_gauss, "--", color=colors_lag[il], alpha=0.5, lw=0.8)

        ax.set_xlabel(r"$r$ [$\sigma_m$]")
        ax.set_ylabel(r"$G_s(r,t)$")
        ax.set_title(f"C={C:.2f}")
        ax.legend(fontsize=7)

    fig.suptitle("Self van Hove — dashed: Gaussian reference")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_D_vanhove.pdf"))
    plt.close(fig)
    print("  Saved fig_D_vanhove.pdf")


# ---------------------------------------------------------------------------
# Fig E — Non-Gaussian parameter α₂(t)
# ---------------------------------------------------------------------------

def fig_E_nongaussian(out_dir, sigma_list, rho, interaction, fig_dir):
    import h5py
    from analysis.non_gaussian import non_gaussian_parameter, alpha2_peak
    from src.utils import _unwrap

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, sigma_E in enumerate(sigma_list):
        C, _ = confinement_parameter(sigma_E, rho)
        label = f"sigma{sigma_E:.1f}_{interaction}_passive"
        traj_path = os.path.join(out_dir, f"traj_{label}.h5")
        if not os.path.exists(traj_path):
            continue
        with h5py.File(traj_path, "r") as f:
            epos = f["enzyme_positions"][:]
            L = float(f.attrs["L"])
            t_arr = f["time"][:]

        n_frames = epos.shape[0]
        lag_idx, alpha2 = non_gaussian_parameter(epos, L)
        t_axis = t_arr[:len(lag_idx)]

        ax.semilogx(t_axis, alpha2, color=COLORS[i % len(COLORS)],
                    label=f"C={C:.2f}")

    ax.set_xlabel(r"$t$ [LJ units]")
    ax.set_ylabel(r"$\alpha_2(t)$")
    ax.set_title(r"Non-Gaussian parameter $\alpha_2(t)$ — Paper 1 Fig. 12")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_E_nongaussian.pdf"))
    plt.close(fig)
    print("  Saved fig_E_nongaussian.pdf")


# ---------------------------------------------------------------------------
# Fig F — Bond survival S(t)  [new CATCHY]
# ---------------------------------------------------------------------------

def fig_F_survival(out_dir, sigma_list, rho, interaction, fig_dir):
    from analysis.degradation import load_survival

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, sigma_E in enumerate(sigma_list):
        C, _ = confinement_parameter(sigma_E, rho)
        result = load_survival(out_dir, sigma_E, interaction)
        if result is None:
            continue
        t, S = result
        ax.plot(t, S, color=COLORS[i % len(COLORS)], label=f"C={C:.2f}")

    ax.set_xlabel(r"$t$ [LJ units]")
    ax.set_ylabel(r"Bond survival $S(t)$")
    ax.set_title("Enzymatic degradation — bond survival")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_F_survival.pdf"))
    plt.close(fig)
    print("  Saved fig_F_survival.pdf")


# ---------------------------------------------------------------------------
# Fig G — D_active / D_passive vs C  [new CATCHY, key result]
# ---------------------------------------------------------------------------

def fig_G_enhancement(out_dir, sigma_list, rho, interaction, fig_dir):
    from analysis.degradation import load_D_table

    C_all, D_act, D_pass = load_D_table(out_dir, sigma_list, rho, interaction)

    fig, ax = plt.subplots(figsize=(5, 4))
    if len(C_all) > 0:
        enhancement = D_act / D_pass
        ax.semilogy(C_all, enhancement, "D-", color="firebrick",
                    markerfacecolor="white", markersize=8, lw=2,
                    label=r"$D_\mathrm{active}/D_\mathrm{passive}$")
        ax.axhline(1.0, color="gray", lw=1, ls="--", label="no enhancement")

        # Annotate regimes
        ax.axvspan(0, 1, alpha=0.05, color="green", label="free diffusion")
        ax.axvspan(1, 3, alpha=0.05, color="orange", label="hopping")
        ax.axvspan(3, C_all.max() + 0.5, alpha=0.05, color="red",
                   label="extreme confinement")

    ax.set_xlabel(r"Confinement $C = \sigma_E / \lambda$")
    ax.set_ylabel(r"$D_\mathrm{active} / D_\mathrm{passive}$")
    ax.set_title("Enzymatic mobility enhancement — key CATCHY result")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_G_enhancement.pdf"))
    plt.close(fig)
    print("  Saved fig_G_enhancement.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)

    sys_cfg = cfg["system"]
    enz_cfg = cfg["enzyme"]
    sim_cfg = cfg["simulation"]
    out_dir = cfg["output"]["dir"]
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    rho = sys_cfg["rho_m0"]
    sigma_list = enz_cfg["sigma_list"]
    interaction = enz_cfg["interaction"]
    dt = sim_cfg["dt"]
    save_interval = sim_cfg["save_interval"]

    print("Generating figures ...")
    fig_A_msd(out_dir, sigma_list, rho, interaction, dt, save_interval, fig_dir)
    fig_B_diffusion(out_dir, sigma_list, rho, interaction, fig_dir)
    fig_C_beta(out_dir, sigma_list, rho, interaction, dt, save_interval, fig_dir)
    fig_D_vanhove(out_dir, sigma_list, rho, interaction, fig_dir)
    fig_E_nongaussian(out_dir, sigma_list, rho, interaction, fig_dir)
    fig_F_survival(out_dir, sigma_list, rho, interaction, fig_dir)
    fig_G_enhancement(out_dir, sigma_list, rho, interaction, fig_dir)

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
