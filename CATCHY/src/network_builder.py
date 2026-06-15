"""
Polymer network builder — Sorichetti et al. 2021 (Paper 1).

Strategy: place beads on a cubic lattice with spacing a < R0.
At rho=0.29, a_natural = (1/rho)^(1/3) = 1.435 σ < R0 = 1.5 σ.
So nearest-neighbour lattice bonds are always valid FENE bonds.

Topology:
  1. Place N_m beads on cubic lattice, spacing a = L/n_side < R0.
  2. Randomly select N_cl = c*N_m cross-link beads.
  3. Build chains by walking lattice neighbours according to
     Flory-Stockmayer strand lengths. Each chain segment connects
     beads that are nearest neighbours on the lattice.
  4. Prune dangling ends and isolated beads.

No unwrapping needed — all bonds are between lattice neighbours
with r < a < R0 by construction.
"""

import numpy as np
from numpy.random import default_rng
from collections import defaultdict, deque

FENE_R0 = 1.5


class NetworkBuilder:
    def __init__(self, N_m=8000, rho=0.290, c=0.1, mean_strand=6, seed=42):
        self.N_m = N_m
        self.rho = rho
        self.c = c
        self.mean_strand = mean_strand
        self.rng = default_rng(seed)
        self.L = (N_m / rho) ** (1.0 / 3.0)

        self.positions = None
        self.backbone_bonds = []
        self.crosslink_bonds = []
        self.crosslink_ids = []
        self._degree = None
        self._cl_set = set()

    def build(self):
        self._place_beads()
        self._assign_crosslinks()
        self._build_neighbour_list()
        self._build_topology()
        self._prune_dangling()
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
        bl = self._bond_lengths()
        print("NetworkBuilder summary")
        print(f"  N_m              = {self.N_m}")
        print(f"  rho              = {self.rho:.3f}   L = {self.L:.3f}")
        print(f"  c                = {self.c:.3f}   N_cl = {len(self.crosslink_ids)}")
        print(f"  mean_strand <n>  = {self.mean_strand}")
        print(f"  backbone bonds   = {len(self.backbone_bonds)}")
        print(f"  cross-link bonds = {len(self.crosslink_bonds)}")
        print(f"  total bonds      = {len(self.all_bonds)}")
        if deg is not None:
            print(f"  degree-2 beads   = {int((deg==2).sum())}  (backbone monomers)")
            print(f"  degree-3 beads   = {int((deg==3).sum())}  (cross-links)")
            print(f"  degree ≤1 beads  = {int((deg<=1).sum())}  ← should be 0")
        if bl is not None and len(bl) > 0:
            print(f"  bond lengths     = min={bl.min():.3f}  "
                  f"mean={bl.mean():.3f}  max={bl.max():.3f}  (R0={FENE_R0})")
            print(f"  bonds >= R0      = {int((bl >= FENE_R0).sum())}  ← should be 0")
        sl = self._strand_lengths()
        if sl is not None and len(sl) > 0:
            print(f"  strand lengths   = mean={np.mean(sl):.2f}  "
                  f"std={np.std(sl):.2f}  (target <n>={self.mean_strand})")

    # ------------------------------------------------------------------ #
    #  Step 1: cubic lattice placement                                     #
    # ------------------------------------------------------------------ #

    def _place_beads(self):
        """
        Place N_m beads on a cubic lattice with spacing a = L/n_side.
        At rho=0.29: a ≈ 1.435 σ < R0 = 1.5 σ — all nearest-neighbour
        bonds are valid FENE bonds by construction.
        """
        n_side = int(np.ceil(self.N_m ** (1.0 / 3.0)))
        a = self.L / n_side

        if a >= FENE_R0:
            import warnings
            warnings.warn(
                f"Lattice spacing a={a:.3f} >= R0={FENE_R0}. "
                f"Bonds may exceed R0. Consider increasing rho.",
                stacklevel=3
            )

        pts = []
        for ix in range(n_side):
            for iy in range(n_side):
                for iz in range(n_side):
                    if len(pts) >= self.N_m:
                        break
                    pts.append([ix * a, iy * a, iz * a])

        pts = np.array(pts[:self.N_m], dtype=float)
        # Tiny jitter to break symmetry (< 2% of spacing)
        pts += self.rng.uniform(-0.02 * a, 0.02 * a, pts.shape)
        pts = pts % self.L
        self.positions = pts
        self._n_side = n_side
        self._a = a

    # ------------------------------------------------------------------ #
    #  Step 2: assign cross-links                                          #
    # ------------------------------------------------------------------ #

    def _assign_crosslinks(self):
        N_cl = max(4, int(round(self.c * self.N_m)))
        self.crosslink_ids = list(
            self.rng.choice(self.N_m, size=N_cl, replace=False)
        )
        self._cl_set = set(self.crosslink_ids)

    # ------------------------------------------------------------------ #
    #  Step 3: build neighbour list (lattice neighbours only)              #
    # ------------------------------------------------------------------ #

    def _build_neighbour_list(self):
        """
        Find nearest neighbours on the cubic lattice (6 neighbours per bead).
        Only bonds between lattice neighbours are < a < R0.
        """
        n_side = self._n_side
        L = self.L
        a = self._a

        # Map lattice indices to bead index
        # bead k = ix*n_side^2 + iy*n_side + iz  (for beads placed in order)
        # But we only placed N_m beads, so cap at N_m
        N = self.N_m

        # Use cell list for robustness (handles jitter)
        r_cut = a * 1.2   # slightly larger than a to catch jittered neighbours
        self._nbrs = defaultdict(list)

        # Cell list
        n_cells = max(1, int(L / r_cut))
        cs = L / n_cells
        pos = self.positions
        cell_idx = (pos / cs).astype(int) % n_cells
        cells = defaultdict(list)
        for i in range(N):
            cells[tuple(cell_idx[i])].append(i)

        r_cut2 = r_cut ** 2
        for i in range(N):
            cx, cy, cz = cell_idx[i]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in cells[((cx+dx)%n_cells,
                                        (cy+dy)%n_cells,
                                        (cz+dz)%n_cells)]:
                            if j <= i:
                                continue
                            dr = pos[i] - pos[j]
                            dr -= L * np.round(dr / L)
                            if np.dot(dr, dr) < r_cut2:
                                self._nbrs[i].append(j)
                                self._nbrs[j].append(i)

    # ------------------------------------------------------------------ #
    #  Step 4: build topology                                              #
    # ------------------------------------------------------------------ #

    def _build_topology(self):
        """
        Build chains by walking lattice neighbours.
        Strand lengths drawn from Flory-Stockmayer distribution.
        Cross-links cannot bond to each other.
        Valence: backbone=2, crosslink=3.
        """
        L = self.L
        N = self.N_m
        cl_set = self._cl_set
        nbrs = self._nbrs
        pos = self.positions

        valence_max = np.array([3 if i in cl_set else 2 for i in range(N)])
        valence_cur = np.zeros(N, dtype=int)
        bond_set = set()
        backbone_bonds = []
        crosslink_bonds = []

        def try_bond(i, j):
            if valence_cur[i] >= valence_max[i]: return False
            if valence_cur[j] >= valence_max[j]: return False
            if i in cl_set and j in cl_set:      return False
            key = (min(i,j), max(i,j))
            if key in bond_set:                  return False
            # Verify bond length via PBC
            dr = pos[i] - pos[j]
            dr -= L * np.round(dr / L)
            if np.linalg.norm(dr) >= FENE_R0:
                return False
            bond_set.add(key)
            valence_cur[i] += 1
            valence_cur[j] += 1
            if i in cl_set or j in cl_set:
                crosslink_bonds.append((i, j))
            else:
                backbone_bonds.append((i, j))
            return True

        def nearest_free_nbr(i, exclude=None):
            candidates = []
            for j in nbrs[i]:
                if j == (exclude or -1): continue
                if valence_cur[j] >= valence_max[j]: continue
                if i in cl_set and j in cl_set: continue
                key = (min(i,j), max(i,j))
                if key in bond_set: continue
                dr = pos[i] - pos[j]
                dr -= L * np.round(dr / L)
                candidates.append((np.linalg.norm(dr), j))
            if not candidates:
                return None
            return min(candidates)[1]

        cl_shuffled = list(self.crosslink_ids)
        self.rng.shuffle(cl_shuffled)

        for cl in cl_shuffled:
            while valence_cur[cl] < valence_max[cl]:
                n_target = max(1, int(self.rng.geometric(1.0 / self.mean_strand)))
                prev = cl
                cur = nearest_free_nbr(cl)
                if cur is None or not try_bond(cl, cur):
                    break
                for _ in range(n_target - 1):
                    if cur in cl_set:
                        break
                    nxt = nearest_free_nbr(cur, exclude=prev)
                    if nxt is None or not try_bond(cur, nxt):
                        break
                    prev, cur = cur, nxt
                if cur not in cl_set:
                    for j in nbrs[cur]:
                        if j in cl_set and valence_cur[j] < valence_max[j]:
                            try_bond(cur, j)
                            break

        # Fill remaining valence
        for i in self.rng.permutation(N):
            while valence_cur[i] < valence_max[i]:
                j = nearest_free_nbr(i)
                if j is None or not try_bond(i, j):
                    break

        self.backbone_bonds = backbone_bonds
        self.crosslink_bonds = crosslink_bonds
        self._update_degree()

    # ------------------------------------------------------------------ #
    #  Step 5: prune dangling ends and isolated beads                      #
    # ------------------------------------------------------------------ #

    def _prune_dangling(self):
        changed = True
        while changed:
            changed = False
            self._update_degree()
            dangling = set(np.where(self._degree == 1)[0].tolist())
            if dangling:
                self.backbone_bonds = [
                    (u,v) for u,v in self.backbone_bonds
                    if u not in dangling and v not in dangling
                ]
                self.crosslink_bonds = [
                    (u,v) for u,v in self.crosslink_bonds
                    if u not in dangling and v not in dangling
                ]
                self.crosslink_ids = [
                    cl for cl in self.crosslink_ids if cl not in dangling
                ]
                self._cl_set -= dangling
                changed = True

        # Remove isolated beads (degree 0) — reindex
        self._update_degree()
        isolated = set(np.where(self._degree == 0)[0].tolist())
        if isolated:
            old2new = {}
            new_idx = 0
            for i in range(self.N_m):
                if i not in isolated:
                    old2new[i] = new_idx
                    new_idx += 1
            self.positions = self.positions[
                [i for i in range(self.N_m) if i not in isolated]
            ]
            self.backbone_bonds = [
                (old2new[u], old2new[v]) for u,v in self.backbone_bonds
            ]
            self.crosslink_bonds = [
                (old2new[u], old2new[v]) for u,v in self.crosslink_bonds
            ]
            self.crosslink_ids = [
                old2new[cl] for cl in self.crosslink_ids if cl not in isolated
            ]
            self._cl_set = set(self.crosslink_ids)
            self.N_m = new_idx
        self._update_degree()

    # ------------------------------------------------------------------ #
    #  Step 6: verify                                                      #
    # ------------------------------------------------------------------ #

    def _verify(self):
        self._update_degree()
        bl = self._bond_lengths()
        if bl is not None and len(bl) > 0:
            n_bad = int((bl >= FENE_R0).sum())
            if n_bad > 0:
                import warnings
                warnings.warn(
                    f"{n_bad} bonds >= R0={FENE_R0} (max={bl.max():.3f}). "
                    "These will cause NaN in FENE. Check lattice spacing.",
                    stacklevel=2
                )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _update_degree(self):
        deg = np.zeros(self.N_m, dtype=int)
        for u, v in self.all_bonds:
            deg[u] += 1; deg[v] += 1
        self._degree = deg

    def _bond_lengths(self):
        if not self.all_bonds or self.positions is None:
            return None
        pos = self.positions; L = self.L
        lengths = []
        for u, v in self.all_bonds:
            dr = pos[u] - pos[v]
            dr -= L * np.round(dr / L)
            lengths.append(np.linalg.norm(dr))
        return np.array(lengths)

    def _strand_lengths(self):
        from collections import defaultdict
        if not self.backbone_bonds:
            return None
        adj = defaultdict(list)
        for u, v in self.backbone_bonds:
            adj[u].append(v); adj[v].append(u)
        cl_set = self._cl_set
        visited = set()
        strand_lengths = []
        for cl in self.crosslink_ids:
            for nbr in adj[cl]:
                if nbr in cl_set: continue
                key = (min(cl,nbr), max(cl,nbr))
                if key in visited: continue
                visited.add(key)
                length = 1
                prev, cur = cl, nbr
                while cur not in cl_set:
                    nexts = [n for n in adj[cur] if n != prev]
                    if not nexts: break
                    prev, cur = cur, nexts[0]
                    length += 1
                strand_lengths.append(length)
        return np.array(strand_lengths) if strand_lengths else None
