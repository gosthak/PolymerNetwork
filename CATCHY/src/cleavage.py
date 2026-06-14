"""
CleavageManager — stochastic enzymatic bond cleavage for CATCHY.

k_cat : cleavage rate in inverse reduced time units [1/τ*]
        Physical meaning: at saturating substrate, one bond per 1/k_cat time units.

Cleavage probability per check interval (Poisson process):
    p = 1 - exp(-k_cat * dt * check_interval)

Only CROSS-LINK bonds are cleaved.
Backbone bonds are permanent chain connections.
The enzyme cleaves the cross-links, liberating polymer chains.
"""

import numpy as np


class CleavageManager:
    """
    Parameters
    ----------
    fene_force : openmm.CustomBondForce
        FENE force with per-bond k_bond parameter.
    crosslink_bond_indices : list[int]
        Indices within fene_force of backbone bonds only.
    N_m : int
        Number of monomers (enzyme indices start at N_m).
    L : float
        Box side length (LJ units).
    k_cat : float
        Cleavage rate [1/τ*] — NOT probability per step.
    r_cleave : float
        Cleavage radius in units of sigma_E.
    sigma_E : float
        Enzyme diameter.
    dt : float
        Integration timestep [τ*].
    check_interval : int
        Steps between cleavage checks.
    """

    def __init__(self, fene_force, crosslink_bond_indices,
                 N_m, L, k_cat=0.005, r_cleave=1.5,
                 sigma_E=1.0, dt=0.006, check_interval=1000):
        self.fene_force      = fene_force
        self.cl_indices      = list(crosslink_bond_indices)
        self.N_m             = N_m
        self.L               = L
        self.k_cat           = k_cat
        self.r_cleave_abs    = r_cleave * sigma_E
        self.dt              = dt
        self.check_interval  = check_interval

        # Cleavage probability per check interval (Poisson)
        # k_cat is a rate [1/τ*], dt*check_interval is the time window
        self.p_cleave = 1.0 - np.exp(-k_cat * dt * check_interval)

        # State
        self.active_bonds  = set(self.cl_indices)
        self.n_cleaved     = 0
        self.cleavage_log  = []   # list of (step, bond_force_idx, i, j)

        # Cache bond atom indices
        self._bond_atoms = {}
        for fi in self.cl_indices:
            p1, p2, _ = fene_force.getBondParameters(fi)
            self._bond_atoms[fi] = (p1, p2)

    # ------------------------------------------------------------------

    def attempt_cleavage(self, context, step):
        """
        Check geometry and attempt cleavage.
        Returns number of bonds cleaved this call.
        """
        if not self.active_bonds:
            return 0

        import openmm.unit as unit
        state = context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        enzyme_pos = pos[self.N_m:]   # (N_E, 3)

        rng = np.random.default_rng()
        bonds_to_cleave = []
        n_contacts = 0

        for fi in list(self.active_bonds):
            i, j = self._bond_atoms[fi]
            # Bond midpoint (PBC)
            dr = pos[j] - pos[i]
            dr -= self.L * np.round(dr / self.L)
            midpoint = pos[i] + 0.5 * dr

            # Check if any enzyme is within cleavage radius
            for ep in enzyme_pos:
                d = ep - midpoint
                d -= self.L * np.round(d / self.L)
                dist = np.linalg.norm(d)
                if dist < self.r_cleave_abs:
                    n_contacts += 1
                    if rng.random() < self.p_cleave:
                        bonds_to_cleave.append(fi)
                        self.cleavage_log.append((step, fi, i, j))
                    break

        if step % 5000 == 0:
            print(f"  [cleavage debug] step={step} "
                  f"active_bonds={len(self.active_bonds)} "
                  f"n_enzymes={len(enzyme_pos)} "
                  f"r_cleave={self.r_cleave_abs:.2f} "
                  f"contacts={n_contacts} "
                  f"p_cleave={self.p_cleave:.4f}")   # one enzyme enough per bond per check

        if bonds_to_cleave:
            for fi in bonds_to_cleave:
                p1, p2, _ = self.fene_force.getBondParameters(fi)
                self.fene_force.setBondParameters(fi, p1, p2, [0.0])
                self.active_bonds.discard(fi)
            self.fene_force.updateParametersInContext(context)
            self.n_cleaved += len(bonds_to_cleave)

        return len(bonds_to_cleave)

    @property
    def survival_fraction(self):
        n_total = len(self.cl_indices)
        return len(self.active_bonds) / n_total if n_total > 0 else 1.0

    def get_bond_status(self):
        """Return int8 array: 1=intact, 0=cleaved, for each backbone bond."""
        return np.array(
            [1 if fi in self.active_bonds else 0 for fi in self.cl_indices],
            dtype=np.int8
        )

    def summary(self):
        print(f"CleavageManager: {self.n_cleaved}/{len(self.cl_indices)} "
              f"backbone bonds cleaved (S={self.survival_fraction:.3f})")
        print(f"  k_cat={self.k_cat} [1/τ*], p_cleave={self.p_cleave:.4f} per check")
