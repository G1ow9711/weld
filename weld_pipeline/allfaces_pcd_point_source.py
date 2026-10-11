"""One original ASCII XYZ PCD in metres -> fresh millimetre encoder input.

The path is caller supplied. Runtime contains no source inventory, label,
STEP, candidate bank, workpiece-name rule or prepared feature-cache argument.
The metre-to-millimetre conversion is shared by every input.
"""
from pathlib import Path

from training.dataio import read_xyz_pcd
from .fresh_xyz_point_source import source_from_original_xyz_mm
from .predict_boundary_incidence_pcd_v1 import runtime_read_guard


def source_from_original_allfaces_pcd_m(pcd_path):
    path = Path(pcd_path).resolve()
    if path.suffix.lower() != ".pcd" or not path.is_file():
        raise ValueError("One original ASCII XYZ PCD in metres required")
    with runtime_read_guard(path) as ledger:
        xyz_mm = read_xyz_pcd(path) * 1000.0
        source = source_from_original_xyz_mm(xyz_mm)
        if ledger["rejected_reads"] or not ledger["reads"]:
            raise RuntimeError("The runtime must read exactly its original PCD")
        if set(ledger["reads"]) != {str(path)}:
            raise RuntimeError("Runtime accessed data other than its input PCD")
    return source
