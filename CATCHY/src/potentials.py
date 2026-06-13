"""
Pair potentials used in Sorichetti et al. 2021 (Paper 1).

Equations referenced:
  eq.(2)  WCA repulsion between all monomers
  eq.(3)  FENE bonded potential (backbone + cross-links)
  eq.(4)  Expanded LJ for enzyme-monomer interactions
"""
import math
from openmm import CustomNonbondedForce, CustomBondForce


# ---------------------------------------------------------------------------
# Constants (Paper 1 reduced LJ units: ε_m = σ_m = m_m = 1)
# ---------------------------------------------------------------------------
SIGMA_M = 1.0          # monomer diameter
EPSILON_M = 1.0        # monomer LJ energy scale
FENE_K = 30.0          # FENE spring constant (Paper 1)
FENE_R0 = 1.5          # FENE maximum extension  (Paper 1)
WCA_RC = 2.0 ** (1.0 / 6.0) * SIGMA_M   # WCA cutoff ≈ 1.122 σ_m


def add_wca(system, particle_types, sigma_m=SIGMA_M, epsilon_m=EPSILON_M):
    """
    Add WCA (purely repulsive) interaction between ALL particle pairs.

    This implements Paper 1 eq.(2):
        U_WCA(r) = 4ε[(σ/r)^12 − (σ/r)^6] + ε ,  r < 2^(1/6)σ
                 = 0                              ,  r ≥ 2^(1/6)σ

    For mixed pairs (monomer–enzyme) we use the WCA form with σ_ij = (σ_i+σ_j)/2
    (Lorentz–Berthelot). The expanded-LJ for enzyme-monomer is added separately
    via add_expanded_lj, which REPLACES the WCA for those pairs.

    Parameters
    ----------
    system : openmm.System
    particle_types : list[str]  'monomer', 'crosslink', or 'enzyme'
    sigma_m, epsilon_m : LJ parameters for monomers

    Returns
    -------
    wca_force : CustomNonbondedForce  (already added to system)
    """
    # Per-particle parameters: sigma, epsilon
    wca = CustomNonbondedForce(
        "U_wca;"
        "U_wca = step(rc - r) * (4*eps*((s/r)^12 - (s/r)^6) + eps);"
        "eps = sqrt(eps1*eps2);"
        "s = 0.5*(sigma1+sigma2);"
        "rc = 2^(1.0/6.0)*s;"
    )
    wca.addPerParticleParameter("sigma")
    wca.addPerParticleParameter("eps")
    wca.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    wca.setCutoffDistance(3.0)   # large enough for all σ_E up to 7

    for ptype in particle_types:
        if ptype in ("monomer", "crosslink"):
            wca.addParticle([sigma_m, epsilon_m])
        else:
            # enzyme sigma is stored externally; placeholder — overridden by
            # add_expanded_lj which is added as a separate force between
            # enzyme-monomer pairs only.  Here enzyme-enzyme WCA uses σ_E.
            raise ValueError(
                "Enzyme WCA should be handled via add_expanded_lj. "
                "Call add_wca only for the monomer subsystem."
            )

    system.addForce(wca)
    return wca


def add_wca_monomers_only(system, n_monomers, sigma_m=SIGMA_M,
                          epsilon_m=EPSILON_M, n_total=None):
    """
    WCA between monomer pairs only (indices 0..n_monomers-1).
    Enzyme pairs are excluded and handled by add_expanded_lj.

    n_total : total number of particles in the system (monomers + enzymes).
              OpenMM requires addParticle() called exactly n_total times.
              If None, assumes n_total == n_monomers (network-only system).
    """
    if n_total is None:
        n_total = n_monomers

    rc  = WCA_RC
    eps = epsilon_m

    energy_expr = (
        f"step({rc:.10f} - r) * "
        f"(4*{eps}*(pow({sigma_m}/r,12) - pow({sigma_m}/r,6)) + {eps})"
    )

    wca = CustomNonbondedForce(energy_expr)
    wca.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    wca.setCutoffDistance(rc * 1.05)

    # Must add exactly n_total particles (one per system particle)
    for i in range(n_total):
        wca.addParticle([])

    # Only compute interactions between monomer pairs
    monomer_set = set(range(n_monomers))
    wca.addInteractionGroup(monomer_set, monomer_set)
    system.addForce(wca)
    return wca


def add_fene(system, bonds, k=FENE_K, r0=FENE_R0):
    """
    FENE bond potential — Paper 1 eq.(3):
        U_FENE(r) = -(k R₀²/2) ln[1 - (r/R₀)²]

    Applied to backbone bonds AND cross-link bonds.
    Returns a CustomBondForce that uses per-bond parameters so that
    CleavageManager can set k→0 for cleaved bonds via updateParametersInContext.

    Parameters
    ----------
    bonds : list[tuple(int,int)]  particle index pairs
    """
    # Energy: U = k_bond * (-0.5 * R0^2 * log(1 - (r/R0)^2))
    # k_bond is per-bond (set to 0 for cleaved bonds by CleavageManager)
    # R0 is fixed — inlined as a number to avoid parser issues
    r0sq = r0 * r0
    # Note: using pow() instead of ^ for CustomBondForce compatibility
    energy_expr = f"k_bond * (-0.5 * {r0sq:.6f} * log(1 - pow(r/{r0:.6f}, 2)))"
    fene = CustomBondForce(energy_expr)
    fene.addPerBondParameter("k_bond")

    for (i, j) in bonds:
        fene.addBond(i, j, [k])

    system.addForce(fene)
    return fene


def add_expanded_lj(system, enzyme_indices, monomer_indices, sigma_E,
                    sigma_m=SIGMA_M, attractive=True):
    """
    Expanded Lennard-Jones between enzymes and monomers — Paper 1 eq.(4):

        σ_ij = (σ_E + σ_m)/2
        Δ    = (σ_E − σ_m)/2
        U(r) = 4ε[(σ_ij/(r−Δ))^12 − (σ_ij/(r−Δ))^6] + ε,  r < r_c + Δ
             = 0                                             ,  r ≥ r_c + Δ

    Repulsive (RNP-like):  ε=1, r_c = 2^(1/6) σ_ij
    Attractive (ANP-like): ε=2, r_c = 2.5 σ_ij

    Parameters
    ----------
    enzyme_indices, monomer_indices : list[int]
    sigma_E : float   enzyme diameter
    attractive : bool

    Returns
    -------
    force : CustomNonbondedForce
    """
    sigma_ij = 0.5 * (sigma_E + sigma_m)
    delta = 0.5 * (sigma_E - sigma_m)
    eps = 2.0 if attractive else 1.0
    rc_lj = 2.5 * sigma_ij if attractive else (2.0 ** (1.0 / 6.0)) * sigma_ij
    cutoff = rc_lj + delta + 0.05   # slightly larger for safety

    eps_shift = 4.0 * eps * ((sigma_ij / rc_lj) ** 12 - (sigma_ij / rc_lj) ** 6)

    sij = sigma_ij
    dlt = delta
    rcd = rc_lj + delta

    energy_expr = (
        f"step({rcd:.6f} - r) * "
        f"(4*{eps}*(pow({sij:.6f}/(r-{dlt:.6f}),12) - pow({sij:.6f}/(r-{dlt:.6f}),6)) - {eps_shift:.8f})"
    )

    force = CustomNonbondedForce(energy_expr)
    force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff)

    # We need placeholders for all particles (OpenMM requires addParticle for each)
    # but interaction is restricted via addInteractionGroup
    n_all = max(max(enzyme_indices), max(monomer_indices)) + 1
    for _ in range(n_all):
        force.addParticle([])

    force.addInteractionGroup(set(enzyme_indices), set(monomer_indices))
    system.addForce(force)
    return force


def add_enzyme_enzyme_wca(system, enzyme_indices, sigma_E, epsilon_m=EPSILON_M):
    """
    WCA between enzyme pairs (enzyme-enzyme repulsion, always on).
    Uses same expanded form with sigma_ij = sigma_E, delta = 0.
    """
    rc = (2.0 ** (1.0 / 6.0)) * sigma_E + 0.02

    energy_expr = (
        f"step({rc:.6f} - r) * "
        f"(4*{epsilon_m}*(pow({sigma_E:.6f}/r,12) - pow({sigma_E:.6f}/r,6)) + {epsilon_m})"
    )
    force = CustomNonbondedForce(energy_expr)
    force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(rc)

    n_all = max(enzyme_indices) + 1
    for _ in range(n_all):
        force.addParticle([])

    eset = set(enzyme_indices)
    force.addInteractionGroup(eset, eset)
    system.addForce(force)
    return force
