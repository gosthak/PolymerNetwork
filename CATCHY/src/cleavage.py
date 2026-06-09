"""
CleavageManager — stochastic enzymatic bond cleavage for CATCHY.

Algorithm (per save interval):
    For each intact FENE bond (i, j):
        midpoint = (r_i + r_j) / 2
        For each enzyme e:
            d = |r_e - midpoint|  (PBC-aware)
            if d < r_cleave * sigma_E:
                draw u ~ Uniform(0,1)
                if u < k_cat * n_steps:
                    cleave bond: set k_bond = 0 via updateParametersInContext

The FENE CustomBondForce uses per-bond parameter k_bond.
Setting k_bond = 0 effectively removes the bond force without
removing the bond from the topology (which OpenMM does not support).
"""

import numpy as np


class CleavageManager:
    """
    Manages enzymatic bond cleavage during production runs.

    Parameters
    ----------
    fene_force : openmm.CustomBondForce
        The FENE force returned by add_fene(). Must use per-bond
        parameter 'k_bond'.
    crosslink_bond_indices : list[int]
        Indices (within fene_force) of cross-link bonds only.
        Backbone bonds are NOT cleaved (enzyme acts on cross-links).
    N_m : int
        Number of monomers (enzyme start index = N_m).
    L : float
        Box side length for PBC minimum-image.
    k_cat : float
        Cleavage probability per timestep per enzyme-bond contact.
    r_cleave : float
        Cleavage radius in units of sigma_E.
    sigma_E : float
        Enzyme diameter.
    dt : float
        Integration timestep.
    """

    def __init__(self, fene_force, crosslink_bond_indices,
                 N_m, L, k_cat=0.005, r_cleave=1.5,
                 sigma_E=1.0, dt=0.006):
        self.fene_force = fene_force
        self.cl_indices = list(crosslink_bond_indices)
        self.N_m = N_m
        self.L = L
        self.k_cat = k_cat
        self.r_cleave_abs = r_cleave * sigma_E
        self.dt = dt
        self.sigma_E = sigma_E

        # State tracking
        self.active_bonds = set(self.cl_indices)   # bond force indices still intact
        self.n_cleaved = 0
        self.cleavage_log = []   # list of (step, bond_force_index, i, j)

        # Cache bond atom indices for quick lookup
        self._bond_atoms = {}   # force_idx -> (i, j)
        for fi in self.cl_indices:
            p1, p2, params = fene_force.getBondParameters(fi)
            self._bond_atoms[fi] = (p1, p2)

    # ------------------------------------------------------------------
    # Main update — call once per save interval
    # ------------------------------------------------------------------

    def attempt_cleavage(self, context, step):
        """
        Attempt bond cleavage given current context state.

        Parameters
        ----------
        context : openmm.Context
        step : int  current simulation step (for logging)

        Returns
        -------
        n_new : int  number of bonds cleaved in this call
        """
        if not self.active_bonds:
            return 0

        state = context.getState(getPositions=True)
        pos = np.array(state.getPositions(asNumpy=True))   # shape (N_total, 3)

        enzyme_pos = pos[self.N_m:]   # (N_E, 3)
        N_E = len(enzyme_pos)
        rng = np.random.default_rng()

        bonds_to_cleave = []
        for fi in list(self.active_bonds):
            i, j = self._bond_atoms[fi]
            midpoint = 0.5 * (pos[i] + pos[j])

            for ep in enzyme_pos:
                dr = ep - midpoint
                # Minimum image
                dr -= self.L * np.round(dr / self.L)
                dist = np.linalg.norm(dr)

                if dist < self.r_cleave_abs:
                    # Cleavage probability accumulated over save_interval steps
                    p_cleave = 1.0 - (1.0 - self.k_cat * self.dt) ** 1   # per-step prob
                    if rng.random() < p_cleave:
                        bonds_to_cleave.append(fi)
                        self.cleavage_log.append((step, fi, i, j))
                        break   # one enzyme is enough to cleave this bond

        if bonds_to_cleave:
            for fi in bonds_to_cleave:
                p1, p2, params = self.fene_force.getBondParameters(fi)
                # Set k_bond = 0 (disable force without removing bond)
                self.fene_force.setBondParameters(fi, p1, p2, [0.0, params[1]])
                self.active_bonds.discard(fi)

            self.fene_force.updateParametersInContext(context)
            self.n_cleaved += len(bonds_to_cleave)

        return len(bonds_to_cleave)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n_intact(self):
        return len(self.active_bonds)

    @property
    def survival_fraction(self):
        n_total = len(self.cl_indices)
        return self.n_intact / n_total if n_total > 0 else 1.0

    def get_bond_status(self):
        """
        Return array of shape (N_cl,) with 1=intact, 0=cleaved,
        in the same order as cl_indices.
        """
        return np.array([1 if fi in self.active_bonds else 0
                         for fi in self.cl_indices], dtype=np.int8)

    def summary(self):
        n_total = len(self.cl_indices)
        print(f"CleavageManager: {self.n_cleaved}/{n_total} bonds cleaved "
              f"(S = {self.survival_fraction:.3f})")
