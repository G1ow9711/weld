"""Readers for the audited ASCII XYZ PCD inputs, including Unicode paths."""
from pathlib import Path
import numpy as np


def read_xyz_pcd(path):
    path = Path(path)
    header = {}
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(" ")
            header[key] = value.strip()
            if key == "DATA":
                break
        if header.get("DATA") != "ascii" or header.get("FIELDS", "").split() != ["x", "y", "z"]:
            raise ValueError(f"Expected audited ASCII XYZ PCD, got {header.get('FIELDS')} / {header.get('DATA')}")
        xyz = np.loadtxt(stream, dtype=np.float64, ndmin=2)
    declared = int(header["POINTS"])
    if xyz.shape != (declared, 3) or not np.isfinite(xyz).all():
        raise ValueError(f"Point count/shape/finite-value mismatch in {path}")
    return xyz
