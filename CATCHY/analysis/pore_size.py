"""
analysis/pore_size.py — Pore-size distribution P(r) and mesh size ξ(t).

P(r) dr = probability that a random point in the network has nearest
monomer at distance in [r, r+dr].

ξ = <r> from P(r)  (mean mesh size / pore size).

During degradation, ξ(t) increases as bonds are cleaved.
"""

import numpy as np


def pore_size_distribution(mono_positions, L, n_probe=5000, r_bins=None,
                           rng=None):
    """
    Monte Carlo estimate of P(r).

    Parameters
    ----------
    mono_positions : np.ndarray (N_m, 3)
    L : float
    n_probe : int  number of random probe points
    r_bins : np.ndarray or None  (auto if None)

    Returns
    -------
    r_centers, P_r : np.ndarray
    xi : float  mean pore radius
    """
    if rng is None:
        rng = np.random.default_rng()

    probes = rng.uniform(0, L, (n_probe, 3))

    min_dists = np.zeros(n_probe)
    for ip, probe in enumerate(probes):
        dr = mono_positions - probe
        dr -= L * np.round(dr / L)
        r = np.linalg.norm(dr, axis=-1)
        min_dists[ip] = r.min()

    if r_bins is None:
        r_bins = np.linspace(0, L / 2, 100)

    counts, edges = np.histogram(min_dists, bins=r_bins)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    P_r = counts / (counts.sum() * (edges[1:] - edges[:-1]) + 1e-30)

    xi = np.average(r_centers, weights=counts + 1e-30)
    return r_centers, P_r, xi


def mesh_size_trajectory(traj_path, frame_indices, L, n_probe=2000):
    """
    Compute ξ(t) from a subset of trajectory frames.

    Parameters
    ----------
    traj_path : str  path to HDF5 trajectory
    frame_indices : list[int]  which frames to analyse
    L : float
    n_probe : int

    Returns
    -------
    times : np.ndarray
    xi_t  : np.ndarray
    """
    import h5py
    rng = np.random.default_rng(0)
    xi_t = []
    times_out = []

    with h5py.File(traj_path, "r") as f:
        N_m = int(f.attrs["N_m"])
        time_arr = f["time"][:]

        for fi in frame_indices:
            pos_frame = f["positions"][fi][:N_m]   # (N_m, 3)
            _, _, xi = pore_size_distribution(pos_frame, L,
                                              n_probe=n_probe, rng=rng)
            xi_t.append(xi)
            times_out.append(time_arr[fi])

    return np.array(times_out), np.array(xi_t)
