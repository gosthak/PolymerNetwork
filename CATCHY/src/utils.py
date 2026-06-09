"""
Utilities for CATCHY simulations.
"""

import os
import yaml
import numpy as np
import h5py


# ---------------------------------------------------------------------------
# Paper 1 (Sorichetti et al. 2021) reference values — Table 1
# ---------------------------------------------------------------------------

# Cross-link dynamic localization length λ for each monomer density
LAMBDA_REF = {
    0.190: 3.12,
    0.290: 2.01,
    0.375: 1.51,
}

def confinement_parameter(sigma_E, rho):
    """C = sigma_E / lambda(rho)."""
    rho_keys = np.array(list(LAMBDA_REF.keys()))
    closest = rho_keys[np.argmin(np.abs(rho_keys - rho))]
    lam = LAMBDA_REF[closest]
    return sigma_E / lam, lam


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# HDF5 trajectory writer
# ---------------------------------------------------------------------------

class HDF5Writer:
    """
    Write trajectory frames to an HDF5 file.

    File structure:
        /positions          (n_frames, N_total, 3)  float32
        /enzyme_positions   (n_frames, N_E, 3)       float32
        /bond_status        (n_frames, N_cl)          int8
        /step               (n_frames,)               int64
        /time               (n_frames,)               float64
        /metadata           attributes: sigma_E, rho, k_cat, N_m, N_E, L
    """

    def __init__(self, path, N_total, N_m, N_E, N_cl,
                 sigma_E, rho, k_cat, L,
                 chunk_size=100, compression="gzip"):
        self.path = path
        self.N_total = N_total
        self.N_m = N_m
        self.N_E = N_E
        self.N_cl = N_cl
        self.chunk_size = chunk_size
        self.frame = 0

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = h5py.File(path, "w")

        # Create resizable datasets
        kw = dict(compression=compression, compression_opts=4)
        self.ds_pos = self.f.create_dataset(
            "positions", shape=(0, N_total, 3), maxshape=(None, N_total, 3),
            dtype="float32", chunks=(chunk_size, N_total, 3), **kw
        )
        self.ds_epos = self.f.create_dataset(
            "enzyme_positions", shape=(0, N_E, 3), maxshape=(None, N_E, 3),
            dtype="float32", chunks=(chunk_size, N_E, 3), **kw
        )
        self.ds_bonds = self.f.create_dataset(
            "bond_status", shape=(0, N_cl), maxshape=(None, N_cl),
            dtype="int8", chunks=(chunk_size, N_cl), **kw
        )
        self.ds_step = self.f.create_dataset(
            "step", shape=(0,), maxshape=(None,), dtype="int64"
        )
        self.ds_time = self.f.create_dataset(
            "time", shape=(0,), maxshape=(None,), dtype="float64"
        )

        # Metadata
        self.f.attrs["sigma_E"] = sigma_E
        self.f.attrs["rho"] = rho
        self.f.attrs["k_cat"] = k_cat
        self.f.attrs["N_m"] = N_m
        self.f.attrs["N_E"] = N_E
        self.f.attrs["L"] = L

    def write_frame(self, positions, bond_status, step, time):
        """
        Parameters
        ----------
        positions : np.ndarray (N_total, 3)
        bond_status : np.ndarray (N_cl,) int8
        step : int
        time : float
        """
        n = self.frame + 1
        self.ds_pos.resize(n, axis=0)
        self.ds_epos.resize(n, axis=0)
        self.ds_bonds.resize(n, axis=0)
        self.ds_step.resize(n, axis=0)
        self.ds_time.resize(n, axis=0)

        self.ds_pos[self.frame] = positions.astype("float32")
        self.ds_epos[self.frame] = positions[self.N_m:self.N_m+self.N_E].astype("float32")
        self.ds_bonds[self.frame] = bond_status
        self.ds_step[self.frame] = step
        self.ds_time[self.frame] = time

        self.frame += 1
        if self.frame % 50 == 0:
            self.f.flush()

    def close(self):
        self.f.flush()
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# MSD computation (used in analysis scripts)
# ---------------------------------------------------------------------------

def compute_msd(positions, L, max_lag=None):
    """
    Compute ensemble-averaged MSD from unwrapped positions.

    Parameters
    ----------
    positions : np.ndarray (n_frames, N, 3)  WRAPPED coordinates
    L : float   box side length

    Returns
    -------
    lag_steps : np.ndarray (max_lag,)
    msd : np.ndarray (max_lag,)
    """
    n_frames, N, _ = positions.shape
    if max_lag is None:
        max_lag = n_frames // 2

    # Unwrap positions
    unwrapped = _unwrap(positions, L)

    msd = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        dr = unwrapped[lag:] - unwrapped[:-lag]
        msd[lag - 1] = np.mean(np.sum(dr ** 2, axis=-1))

    return np.arange(1, max_lag + 1), msd


def _unwrap(positions, L):
    """Unwrap periodic coordinates."""
    unwrapped = positions.copy().astype(float)
    for t in range(1, len(positions)):
        delta = unwrapped[t] - unwrapped[t - 1]
        delta -= L * np.round(delta / L)
        unwrapped[t] = unwrapped[t - 1] + delta
    return unwrapped
