import os
import MDAnalysis as mda

os.makedirs("xyz_frames", exist_ok=True)

u = mda.Universe(
    "topology_sigma1.0_attractive.pdb",
    "traj_sigma1.0_attractive_active.dcd"
)

for ts in u.trajectory[::1000]:
    u.atoms.write(f"xyz_frames/frame_{ts.frame:06d}.xyz")
