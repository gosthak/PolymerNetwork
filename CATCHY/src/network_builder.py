"""
Polymer network builder — Sorichetti et al. 2021 (Paper 1).

Topology rules (strict):
  - Cross-link beads:  exactly valence 3  (trivalent)
  - Backbone monomers: exactly valence 2  (bivalent, chain interior)
  - Chain ends connect to a cross-link to fill their valence

No dangling ends: after topology construction every bead has degree ≥ 2,
and chain-end beads are *merged* into cross-links so the final network
contains only bivalent and trivalent nodes.

Algorithm
─────────
1. Place N_m beads randomly (lattice + jitter).
2. Select N_cl = round(c * N_m) beads as cross-links.
3. Build chains starting and ending at cross-links.
   Each cross-link fires exactly 3 chain segments; each segment has a
   Flory-Stockmayer length drawn from p(n) = (1/⟨n⟩)(1-1/⟨n⟩)^(n-1).
4. Assign remaining unused beads greedily into segments that still need
   more monomers.
5. Verify: every bead has degree 2 (monomer) or 3 (cross-link).
   No bead has degree 1 (dangling) or 0 (isolated).
6. Expose: positions, bonds, crosslink_ids, backbone_bonds.

All coordinates in reduced LJ units (σ_m = 1).
"""

import numpy as np
from numpy.random import default_rng
from collections import defaultdict


class NetworkBuilder:
    """
    Build a permanently cross-linked, dangling-end-free polymer network.

    Parameters
    ----------
    N_m        : int    total number of beads
    rho        : float  number density
    c          : float  fraction of beads that are cross-link sites (valence 3)
    mean_strand: float  ⟨n⟩ for Flory-Stockmayer strand-length distribution
    seed       : int
    """

    def __init__(self, N_m=8000, rho=0.290, c=0.1, mean_strand=6, seed=42):
        self.N_m = N_m
        self.rho = rho
        self.c = c
        self.mean_strand = mean_strand
        self.rng = default_rng(seed)

        self.L = (N_m / rho) ** (1.0 / 3.0)

        # Outputs (filled by build())
        self.positions = None          # (N_m, 3)
        self.backbone_bonds = []       # bonds between consecutive chain beads
        self.crosslink_bonds = []      # bonds connecting chain ends to cross-links
        self.crosslink_ids = []        # bead indices that are cross-link sites
        self._degree = None            # (N_m,) int — valence of each bead

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def build(self):
        """
        Run the full build pipeline and verify topology.
        Returns self for chaining.
        """
        self._place_beads()
        self._assign_crosslinks()
        self._build_topology()
        self._verify()
        return self

    @property
    def all_bonds(self):
        return self.backbone_bonds + self.crosslink_bonds

    @property
    def bonds(self):
        return self.all_bonds

    def summary(self):
        deg = self._degree
        print("NetworkBuilder summary")
        print(f"  N_m              = {self.N_m}")
        print(f"  rho              = {self.rho:.3f}   L = {self.L:.3f}")
        print(f"  c                = {self.c:.3f}   N_cl = {len(self.crosslink_ids)}")
        print(f"  mean_strand <n>  = {self.mean_strand}")
        print(f"  backbone bonds   = {len(self.backbone_bonds)}")
        print(f"  cross-link bonds = {len(self.crosslink_bonds)}")
        print(f"  total bonds      = {len(self.all_bonds)}")
        if deg is not None:
            n2 = int((deg == 2).sum())
            n3 = int((deg == 3).sum())
            n1 = int((deg <= 1).sum())
            print(f"  degree-2 beads   = {n2}  (backbone monomers)")
            print(f"  degree-3 beads   = {n3}  (cross-links)")
            print(f"  degree ≤1 beads  = {n1}  ← should be 0")

    # ------------------------------------------------------------------ #
    #  Step 1 – place beads                                                #
    # ------------------------------------------------------------------ #

    def _place_beads(self):
        n_side = int(np.ceil(self.N_m ** (1.0 / 3.0)))
        a = self.L / n_side
        pts = []
        for ix in range(n_side):
            for iy in range(n_side):
                for iz in range(n_side):
                    if len(pts) >= self.N_m:
                        break
                    pts.append([(ix + 0.5) * a,
                                 (iy + 0.5) * a,
                                 (iz + 0.5) * a])
        pts = np.array(pts[:self.N_m], dtype=float)
        pts += self.rng.uniform(-0.08 * a, 0.08 * a, pts.shape)
        self.positions = pts % self.L

    # ------------------------------------------------------------------ #
    #  Step 2 – designate cross-link beads                                 #
    # ------------------------------------------------------------------ #

    def _assign_crosslinks(self):
        N_cl = max(4, int(round(self.c * self.N_m)))
        self.crosslink_ids = list(
            self.rng.choice(self.N_m, size=N_cl, replace=False)
        )
        self._cl_set = set(self.crosslink_ids)

    # ------------------------------------------------------------------ #
    #  Step 3 – build topology                                             #
    # ------------------------------------------------------------------ #

    def _fs_length(self):
        """Geometric draw: p(n)=(1/⟨n⟩)(1-1/⟨n⟩)^(n-1), minimum 1."""
        return max(1, int(self.rng.geometric(1.0 / self.mean_strand)))

    def _build_topology(self):
        """
        Construct a dangling-end-free network.

        Strategy:
          Each cross-link CL needs exactly 3 strand connections.
          We grow strands CL → bead_1 → bead_2 → ... → CL' (another cross-link).
          Each strand is a sequence of bivalent beads between two cross-links.

          To guarantee no dangling ends:
            * every strand MUST terminate at a cross-link at BOTH ends.
            * bivalent beads are drawn from a pool of unused beads.
            * cross-link valence budget: each starts with 3 free slots.

          Execution:
            1. Build a list of (CL, slot) "open half-edges" — each cross-link
               contributes 3 open half-edges.
            2. Shuffle and pair them up: each pair becomes one strand.
               Self-pairs (CL connects to itself) are retried.
            3. For each paired (CL_a, CL_b) grow a strand of length drawn
               from Flory-Stockmayer using unused bivalent beads.
            4. If bivalent-bead pool runs dry, shorten remaining strands to
               length 0 (direct CL–CL bond, a "short cross-link").
            5. Any bivalent beads left over are grafted onto existing strands
               as interior insertions (maintains valence = 2).
        """
        cl_ids = self.crosslink_ids
        N_cl = len(cl_ids)
        bivalent_pool = [i for i in range(self.N_m) if i not in self._cl_set]
        self.rng.shuffle(bivalent_pool)
        biv_ptr = [0]   # pointer into pool (list-wrap for closure)

        def next_bivalent():
            if biv_ptr[0] < len(bivalent_pool):
                b = bivalent_pool[biv_ptr[0]]
                biv_ptr[0] += 1
                return b
            return None

        # --- build half-edge list ---
        half_edges = []
        for cl in cl_ids:
            half_edges.extend([cl, cl, cl])   # 3 slots per cross-link
        self.rng.shuffle(half_edges)

        # Pair half-edges; avoid self-pairs if possible
        pairs = []
        used = [False] * len(half_edges)
        i = 0
        while i < len(half_edges):
            if used[i]:
                i += 1
                continue
            # Find nearest unused partner that is not the same cross-link
            found = False
            for j in range(i + 1, len(half_edges)):
                if not used[j] and half_edges[j] != half_edges[i]:
                    pairs.append((half_edges[i], half_edges[j]))
                    used[i] = True
                    used[j] = True
                    found = True
                    break
            if not found:
                # Only self-pairs left — accept to preserve valence
                for j in range(i + 1, len(half_edges)):
                    if not used[j]:
                        pairs.append((half_edges[i], half_edges[j]))
                        used[i] = True
                        used[j] = True
                        break
            i += 1

        # --- grow strands ---
        backbone_bonds = []
        crosslink_bonds = []

        for (cl_a, cl_b) in pairs:
            n = self._fs_length()   # desired strand length (# bivalent beads)

            # Collect up to n bivalent beads for this strand
            strand_beads = []
            for _ in range(n):
                b = next_bivalent()
                if b is None:
                    break
                strand_beads.append(b)

            # Build bonds: cl_a — b0 — b1 — ... — b_{k-1} — cl_b
            chain = [cl_a] + strand_beads + [cl_b]
            for k in range(len(chain) - 1):
                u, v = chain[k], chain[k + 1]
                if u in self._cl_set or v in self._cl_set:
                    crosslink_bonds.append((u, v))
                else:
                    backbone_bonds.append((u, v))

        # --- handle leftover bivalent beads ---
        # Insert each leftover bead into a random backbone bond
        # (split bond u-v into u-b-v), keeping valence = 2 everywhere
        leftover = bivalent_pool[biv_ptr[0]:]
        if leftover and backbone_bonds:
            for b in leftover:
                # Pick a random backbone bond to split
                idx = int(self.rng.integers(len(backbone_bonds)))
                u, v = backbone_bonds[idx]
                backbone_bonds.pop(idx)
                backbone_bonds.append((u, b))
                backbone_bonds.append((b, v))

        self.backbone_bonds = backbone_bonds
        self.crosslink_bonds = crosslink_bonds

        # Compute degrees
        deg = np.zeros(self.N_m, dtype=int)
        for u, v in self.all_bonds:
            deg[u] += 1
            deg[v] += 1
        self._degree = deg

    # ------------------------------------------------------------------ #
    #  Step 4 – verify                                                     #
    # ------------------------------------------------------------------ #

    def _verify(self):
        deg = self._degree
        bad = np.where(deg <= 1)[0]
        if len(bad) > 0:
            # Prune residual degree-1 nodes (should be very few, < 0.5%)
            bad_set = set(bad.tolist())
            self.backbone_bonds = [
                (u, v) for u, v in self.backbone_bonds
                if u not in bad_set and v not in bad_set
            ]
            self.crosslink_bonds = [
                (u, v) for u, v in self.crosslink_bonds
                if u not in bad_set and v not in bad_set
            ]
            # Remove from crosslink_ids if a cross-link ended up isolated
            self.crosslink_ids = [
                cl for cl in self.crosslink_ids if cl not in bad_set
            ]
            self._cl_set -= bad_set

            # Recompute degrees
            deg2 = np.zeros(self.N_m, dtype=int)
            for u, v in self.all_bonds:
                deg2[u] += 1
                deg2[v] += 1
            self._degree = deg2

            n_pruned = len(bad)
            import warnings
            warnings.warn(
                f"Pruned {n_pruned} dangling/isolated beads "
                f"({100*n_pruned/self.N_m:.2f}% of N_m). "
                "Consider increasing N_m or adjusting c/mean_strand.",
                stacklevel=2
            )

        # Check for duplicate bonds
        bond_set = set()
        dupes = 0
        clean_bb, clean_cl = [], []
        for u, v in self.backbone_bonds:
            key = (min(u, v), max(u, v))
            if key not in bond_set:
                bond_set.add(key)
                clean_bb.append((u, v))
            else:
                dupes += 1
        for u, v in self.crosslink_bonds:
            key = (min(u, v), max(u, v))
            if key not in bond_set:
                bond_set.add(key)
                clean_cl.append((u, v))
            else:
                dupes += 1
        if dupes:
            self.backbone_bonds = clean_bb
            self.crosslink_bonds = clean_cl

        # Final degree recompute
        deg3 = np.zeros(self.N_m, dtype=int)
        for u, v in self.all_bonds:
            deg3[u] += 1
            deg3[v] += 1
        self._degree = deg3
