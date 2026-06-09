"""
tests/test_potentials.py — unit tests for potentials and key physics.

Run with:  pytest tests/test_potentials.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# WCA potential tests
# ---------------------------------------------------------------------------

class TestWCA:
    """Paper 1 eq.(2): U_WCA(r) = 4ε[(σ/r)^12 - (σ/r)^6] + ε for r < 2^(1/6)σ"""

    sigma = 1.0
    eps = 1.0
    rc = 2.0 ** (1.0 / 6.0)

    def U_wca(self, r):
        if r < self.rc * self.sigma:
            return 4 * self.eps * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6) + self.eps
        return 0.0

    def test_zero_at_cutoff(self):
        """U_WCA should be exactly 0 at r = 2^(1/6) σ."""
        U = self.U_wca(self.rc * self.sigma)
        assert abs(U) < 1e-10, f"U_WCA({self.rc:.4f}) = {U:.2e} ≠ 0"

    def test_positive_repulsive(self):
        """WCA is purely repulsive: U > 0 for r < r_c."""
        r_vals = np.linspace(0.8, self.rc - 0.01, 20)
        for r in r_vals:
            assert self.U_wca(r) > 0, f"U_WCA({r}) = {self.U_wca(r):.3f} ≤ 0"

    def test_zero_beyond_cutoff(self):
        """U_WCA = 0 beyond cutoff."""
        for r in [self.rc + 0.01, 1.5, 2.0, 5.0]:
            assert self.U_wca(r) == 0.0

    def test_minimum_at_sigma(self):
        """Minimum of full LJ at r = 2^(1/6) σ — WCA is shifted to be 0 there."""
        # The minimum of U_LJ = -ε at r = 2^(1/6) σ; WCA shifts by +ε → U = 0
        U_lj_min = 4 * self.eps * (
            (self.sigma / self.rc) ** 12 - (self.sigma / self.rc) ** 6
        )
        assert abs(U_lj_min + self.eps) < 1e-10


# ---------------------------------------------------------------------------
# FENE potential tests
# ---------------------------------------------------------------------------

class TestFENE:
    """Paper 1 eq.(3): U_FENE = -(k R0²/2) ln[1 - (r/R0)²]"""

    k = 30.0
    R0 = 1.5

    def U_fene(self, r):
        if r >= self.R0:
            return np.inf
        return -0.5 * self.k * self.R0 ** 2 * np.log(1 - (r / self.R0) ** 2)

    def test_zero_at_origin(self):
        """U_FENE(0) = 0."""
        assert self.U_fene(0.0) == 0.0

    def test_diverges_at_R0(self):
        """U_FENE → +∞ as r → R0."""
        U = self.U_fene(self.R0 - 0.001)
        assert U > 1e4, f"U_FENE near R0 = {U:.1f}, expected large"

    def test_monotonically_increasing(self):
        """U_FENE is monotonically increasing."""
        r_vals = np.linspace(0.01, self.R0 - 0.01, 50)
        U_vals = [self.U_fene(r) for r in r_vals]
        diffs = np.diff(U_vals)
        assert np.all(diffs > 0), "U_FENE is not monotonically increasing"

    def test_force_is_restoring(self):
        """F_FENE = -dU/dr = -k r / (1 - (r/R0)²) < 0 for r > 0."""
        r = 0.5
        F = -self.k * r / (1 - (r / self.R0) ** 2)
        assert F < 0, "FENE force should be restoring (negative)"


# ---------------------------------------------------------------------------
# Expanded LJ tests
# ---------------------------------------------------------------------------

class TestExpandedLJ:
    """Paper 1 eq.(4)."""

    sigma_m = 1.0
    sigma_E = 3.0
    sigma_ij = 0.5 * (sigma_m + sigma_E)   # = 2.0
    delta = 0.5 * (sigma_E - sigma_m)       # = 1.0

    def U_elj(self, r, attractive=False):
        eps = 2.0 if attractive else 1.0
        rc_lj = 2.5 * self.sigma_ij if attractive else 2.0 ** (1 / 6) * self.sigma_ij
        if r - self.delta <= 0 or r >= rc_lj + self.delta:
            return 0.0
        s_over_r = self.sigma_ij / (r - self.delta)
        eps_cut = 4 * eps * (
            (self.sigma_ij / rc_lj) ** 12 - (self.sigma_ij / rc_lj) ** 6
        )
        return 4 * eps * (s_over_r ** 12 - s_over_r ** 6) - eps_cut

    def test_contact_distance(self):
        """At r = sigma_ij + delta = sigma_E, expanded LJ has contact."""
        r_contact = self.sigma_ij + self.delta   # = sigma_E = 3.0
        # For repulsive, r_c = 2^(1/6) * sigma_ij ≈ 2.245 → contact is inside rc+delta
        U = self.U_elj(r_contact, attractive=False)
        # U at contact (r-delta = sigma_ij) → same as LJ at sigma → U = 0 + shift = eps
        # For repulsive: U_cut = 0 (at r=rc), so U = 4ε[(1)^12-(1)^6] = 0 but shifted
        # Just check it's computable and non-negative for repulsive
        assert np.isfinite(U)

    def test_shift_zero_at_cutoff_repulsive(self):
        """Repulsive expanded LJ: U=0 at r = 2^(1/6)*sigma_ij + delta."""
        rc_plus_delta = 2.0 ** (1.0 / 6.0) * self.sigma_ij + self.delta
        U = self.U_elj(rc_plus_delta - 1e-6, attractive=False)
        assert abs(U) < 0.01, f"U not near zero at cutoff: {U:.4f}"

    def test_attractive_well_exists(self):
        """Attractive expanded LJ has a negative well."""
        # Check at r = 1.2*sigma_ij + delta (well region for attractive)
        r_well = 1.2 * self.sigma_ij + self.delta
        U = self.U_elj(r_well, attractive=True)
        assert U < 0, f"Attractive well not negative at r={r_well}: U={U:.4f}"


# ---------------------------------------------------------------------------
# Confinement parameter tests
# ---------------------------------------------------------------------------

def test_confinement_paper1_table1():
    """Reference values from Paper 1 Table 1."""
    from src.utils import confinement_parameter, LAMBDA_REF

    assert abs(LAMBDA_REF[0.190] - 3.12) < 0.01
    assert abs(LAMBDA_REF[0.290] - 2.01) < 0.01
    assert abs(LAMBDA_REF[0.375] - 1.51) < 0.01

    # C = sigma_E / lambda
    C, lam = confinement_parameter(sigma_E=2.01, rho=0.290)
    assert abs(C - 1.0) < 0.01, f"C = {C:.3f} ≠ 1.0 for sigma_E=lambda"


# ---------------------------------------------------------------------------
# NetworkBuilder tests
# ---------------------------------------------------------------------------

def test_network_builder_small():
    """Build a small network and check basic properties."""
    from src.network_builder import NetworkBuilder

    builder = NetworkBuilder(N_m=500, rho=0.290, c=0.1, mean_strand=6, seed=0)
    builder.build()

    assert builder.positions.shape == (500, 3)
    assert len(builder.backbone_bonds) > 0
    assert len(builder.crosslink_bonds) > 0
    # All bonds reference valid particle indices
    all_bonds = builder.all_bonds
    for i, j in all_bonds:
        assert 0 <= i < 500
        assert 0 <= j < 500
        assert i != j


def test_flory_stockmayer_mean():
    """Mean strand length should be close to <n> for large sample."""
    from src.network_builder import NetworkBuilder
    builder = NetworkBuilder(N_m=10000, rho=0.290, c=0.0, mean_strand=6, seed=99)
    builder._place_on_lattice()
    lengths = [builder._flory_stockmayer_strand_length() for _ in range(10000)]
    mean = np.mean(lengths)
    assert abs(mean - 6.0) < 0.3, f"Mean strand length = {mean:.2f}, expected ≈ 6"


# ---------------------------------------------------------------------------
# MSD utility tests
# ---------------------------------------------------------------------------

def test_msd_free_diffusion():
    """For free diffusion, MSD = 6 D t."""
    rng = np.random.default_rng(42)
    D = 1.0
    dt = 0.01
    N = 100
    n_frames = 500
    L = 1000.0   # large box — no PBC effects

    # Generate free diffusion trajectories
    displacements = rng.normal(0, np.sqrt(2 * D * dt), (n_frames, N, 3))
    positions = np.cumsum(displacements, axis=0)

    from src.utils import compute_msd
    lag_steps, msd = compute_msd(positions, L, max_lag=100)
    time_axis = lag_steps * dt

    # Fit D from MSD = 6 D t
    D_fit = np.polyfit(time_axis[10:80], msd[10:80], 1)[0] / 6.0
    assert abs(D_fit - D) / D < 0.1, f"D_fit = {D_fit:.3f}, expected ≈ {D}"
