"""
Pair potentials — Sorichetti et al. 2021 (Paper 1).

  eq.(2)  WCA repulsion between monomers
  eq.(3)  FENE bonded potential
  eq.(4)  Expanded LJ for NP-monomer and NP-NP interactions
"""
from openmm import CustomNonbondedForce, CustomBondForce

# LJ reduced units: ε_m = σ_m = m_m = 1
SIGMA_M   = 1.0
EPSILON_M = 1.0
FENE_K    = 30.0
FENE_R0   = 1.5
WCA_RC    = 2.0 ** (1.0 / 6.0) * SIGMA_M   # ≈ 1.122 σ_m


def add_wca_monomers_only(system, n_monomers, sigma_m=SIGMA_M,
                          epsilon_m=EPSILON_M, n_total=None):
    """
    WCA (eq.2) between monomer pairs only.
    n_total: total particles in system — OpenMM requires addParticle() n_total times.
    """
    if n_total is None:
        n_total = n_monomers

    rc  = WCA_RC
    eps = epsilon_m
    energy_expr = (
        f"step({rc:.10f} - r) * "
        f"(4*{eps}*((({sigma_m}/r)^12) - (({sigma_m}/r)^6)) + {eps})"
    )
    wca = CustomNonbondedForce(energy_expr)
    wca.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    wca.setCutoffDistance(rc * 1.05)

    for _ in range(n_total):
        wca.addParticle([])

    wca.addInteractionGroup(set(range(n_monomers)), set(range(n_monomers)))
    system.addForce(wca)
    return wca


def add_fene(system, bonds, k=FENE_K, r0=FENE_R0):
    """
    FENE bond potential (eq.3):
        U = -0.5 k R0² ln(1 - (r/R0)²)
    k_bond is per-bond so CleavageManager can zero cleaved bonds.
    """
    r0sq = r0 * r0
    # select() avoids computing log when k_bond=0 (cleaved bond, r may exceed R0)
    energy_expr = (
        f"select(k_bond, "
        f"k_bond * (-0.5 * {r0sq:.6f} * log(1 - (r/{r0:.6f})^2)), "
        f"0)"
    )
    fene = CustomBondForce(energy_expr)
    fene.setUsesPeriodicBoundaryConditions(True)   # minimum image for bonded pairs
    # Or no PBC for FENE — positions must be unwrapped so bonded pairs?
    # are close in Cartesian space (minimizer works without PBC)
    fene.addPerBondParameter("k_bond")
    for (i, j) in bonds:
        fene.addBond(i, j, [k])
    system.addForce(fene)
    return fene


def add_expanded_lj(system, enzyme_indices, monomer_indices, sigma_E,
                    sigma_m=SIGMA_M, attractive=True):
    """
    Expanded LJ for NP-monomer interaction (eq.4):
        Δ_Nm = (σ_E - σ_m) / 2
        r_c  = 2.5 σ_m (attractive) or 2^(1/6) σ_m (repulsive)
        U(r) = 4ε[(σ_m/(r-Δ))^12 - (σ_m/(r-Δ))^6] - E_c,  r ≤ Δ + r_c
    E_c ensures U(Δ + r_c) = 0.
    """
    delta  = 0.5 * (sigma_E - sigma_m)
    eps    = 2.0 if attractive else 1.0
    rc_lj  = 2.5 * sigma_m if attractive else WCA_RC
    rcd    = rc_lj + delta          # total cutoff in r-space
    cutoff = rcd + 0.05

    # E_c = -U_LJ(r_c) so that U(Δ + r_c) = 0
    eps_shift = 4.0 * eps * ((sigma_m / rc_lj) ** 12 - (sigma_m / rc_lj) ** 6)

    energy_expr = (
        f"step({rcd:.6f} - r) * "
        f"(4*{eps}*(({sigma_m:.6f}/(r-{delta:.6f}))^12"
        f" - ({sigma_m:.6f}/(r-{delta:.6f}))^6) - {eps_shift:.8f})"
    )
    force = CustomNonbondedForce(energy_expr)
    force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff)

    n_all = max(max(enzyme_indices), max(monomer_indices)) + 1
    for _ in range(n_all):
        force.addParticle([])

    force.addInteractionGroup(set(enzyme_indices), set(monomer_indices))
    system.addForce(force)
    return force


def add_enzyme_enzyme_wca(system, enzyme_indices, sigma_E,
                          sigma_m=SIGMA_M, epsilon_m=EPSILON_M):
    """
    NP-NP interaction (eq.4):
        Δ_NN = σ_E - σ_m
        r_c  = 2^(1/6) σ_m  (purely repulsive)
        U(r) = 4ε[(σ_m/(r-Δ))^12 - (σ_m/(r-Δ))^6] - E_c,  r ≤ Δ + r_c
    """
    delta  = sigma_E - sigma_m
    rc     = WCA_RC
    rcd    = delta + rc
    cutoff = rcd + 0.05

    eps_shift = 4.0 * epsilon_m * ((sigma_m / rc) ** 12 - (sigma_m / rc) ** 6)

    energy_expr = (
        f"step({rcd:.6f} - r) * "
        f"(4*{epsilon_m}*(({sigma_m:.6f}/(r-{delta:.6f}))^12"
        f" - ({sigma_m:.6f}/(r-{delta:.6f}))^6) - {eps_shift:.8f})"
    )
    force = CustomNonbondedForce(energy_expr)
    force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff)

    n_all = max(enzyme_indices) + 1
    for _ in range(n_all):
        force.addParticle([])

    force.addInteractionGroup(set(enzyme_indices), set(enzyme_indices))
    system.addForce(force)
    return force
