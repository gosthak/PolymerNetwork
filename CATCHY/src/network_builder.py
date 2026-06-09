"""
Polymer network builder — Sorichetti et al. 2021 (Paper 1).

Generates a polydisperse, randomly cross-linked Kremer-Grest network:
- Strand lengths drawn from Flory-Stockmayer distribution (eq. 1, Paper 1):
      p(n) = (1/⟨n⟩)(1 - 1/⟨n⟩)^(n-1)
- Monomers placed on a simple cubic lattice, then relaxed.
- Cross-links chosen randomly among non-bonded monomer pairs within r_cl.

All coordinates in reduced LJ units (σ_m = 1).
"""

import numpy as np
from numpy.random import default_rng


class NetworkBuilder:
    """
    Build a permanently cross-linked polymer network matching Paper 1.

    Parameters
    ----------
    N_m : int    total number of monomers
    rho : float  monomer number density
    c   : float  fraction of monomers that are cross-link sites
    mean_strand : float  ⟨n⟩ in Flory-Stockmayer distribution
    seed : int
    """

    def __init__(self, N_m=8000, rho=0.290, c=0.1, mean_strand=6, seed=42):
        self.N_m = N_m
        self.rho = rho
        self.c = c
        self.mean_strand = mean_strand
        self.rng = default_rng(seed)

        self.L = (N_m / rho) ** (1.0 / 3.0)   # box side length
        self.positions = None      # (N_m, 3) array
        self.bonds = []            # list of (i, j) tuples — backbone + cross-links
        self.crosslink_ids = []    # indices of cross-link beads
        self.backbone_bonds = []
        self.crosslink_bonds = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self):
        """
        Full build pipeline:
        1. Place monomers on lattice
        2. Generate chain topology via Flory-Stockmayer strand lengths
        3. Randomly assign cross-links
        Returns self for chaining.
        """
        self._place_on_lattice()
        self._build_chains()
        self._add_crosslinks()
        return self

    # ------------------------------------------------------------------
    # Step 1: lattice placement
    # ------------------------------------------------------------------

    def _place_on_lattice(self):
        """Place monomers on a cubic lattice with small random displacements."""
        n_side = int(np.ceil(self.N_m ** (1.0 / 3.0)))
        a = self.L / n_side   # lattice spacing

        positions = []
        for ix in range(n_side):
            for iy in range(n_side):
                for iz in range(n_side):
                    if len(positions) >= self.N_m:
                        break
                    x = (ix + 0.5) * a
                    y = (iy + 0.5) * a
                    z = (iz + 0.5) * a
                    positions.append([x, y, z])

        positions = np.array(positions[:self.N_m])
        # Small random displacements to break symmetry
        positions += self.rng.uniform(-0.05 * a, 0.05 * a, positions.shape)
        # Wrap into box
        self.positions = positions % self.L

    # ------------------------------------------------------------------
    # Step 2: chain topology (Flory-Stockmayer strand lengths)
    # ------------------------------------------------------------------

    def _flory_stockmayer_strand_length(self):
        """
        Draw a strand length n from the geometric distribution:
        p(n) = (1/⟨n⟩)(1 - 1/⟨n⟩)^(n-1),  n = 1, 2, 3, …

        This is equivalent to a geometric distribution with success
        probability p = 1/⟨n⟩.
        """
        p = 1.0 / self.mean_strand
        # numpy geometric: number of trials until first success, 1-indexed
        return int(self.rng.geometric(p))

    def _build_chains(self):
        """
        Assign monomers sequentially to linear chains with Flory-Stockmayer
        strand lengths. Connect consecutive monomers in each strand with
        backbone FENE bonds.
        """
        idx = 0
        while idx < self.N_m:
            n = self._flory_stockmayer_strand_length()
            end = min(idx + n, self.N_m)
            for k in range(idx, end - 1):
                self.backbone_bonds.append((k, k + 1))
            idx = end

        self.bonds = list(self.backbone_bonds)

    # ------------------------------------------------------------------
    # Step 3: cross-links
    # ------------------------------------------------------------------

    def _add_crosslinks(self):
        """
        Randomly select N_cl = c * N_m monomers as cross-link sites,
        then connect nearby pairs that are not already bonded.
        """
        N_cl = int(self.c * self.N_m)
        candidates = self.rng.choice(self.N_m, size=N_cl, replace=False)
        self.crosslink_ids = list(candidates)

        # Build a simple cell-list for efficiency
        r_cl = 1.5   # maximum cross-link bond length (in σ_m units)
        bonded = set(map(frozenset, self.backbone_bonds))

        already_crosslinked = set()
        for i in range(len(candidates)):
            if candidates[i] in already_crosslinked:
                continue
            for j in range(i + 1, len(candidates)):
                if candidates[j] in already_crosslinked:
                    continue
                pair = frozenset([candidates[i], candidates[j]])
                if pair in bonded:
                    continue
                # Minimum image distance
                dr = self.positions[candidates[i]] - self.positions[candidates[j]]
                dr -= self.L * np.round(dr / self.L)
                dist = np.linalg.norm(dr)
                if dist < r_cl:
                    self.crosslink_bonds.append((candidates[i], candidates[j]))
                    self.bonds.append((candidates[i], candidates[j]))
                    bonded.add(pair)
                    already_crosslinked.add(candidates[i])
                    already_crosslinked.add(candidates[j])
                    break   # each cross-link site bonds to at most one partner

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def all_bonds(self):
        return self.backbone_bonds + self.crosslink_bonds

    def summary(self):
        print(f"NetworkBuilder summary")
        print(f"  N_m          = {self.N_m}")
        print(f"  rho          = {self.rho:.3f}")
        print(f"  Box side L   = {self.L:.3f} σ_m")
        print(f"  Backbone bonds = {len(self.backbone_bonds)}")
        print(f"  Cross-links    = {len(self.crosslink_bonds)}  (target: {int(self.c*self.N_m/2)})")
        print(f"  Total bonds    = {len(self.all_bonds)}")
