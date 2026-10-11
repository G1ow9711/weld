"""Pure XYZ spatial grouping; all original rows retain their memberships."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .live_json_snapshot_282 import write_json_snapshot as dump
from .raw_triangle_neural_features_279 import file_sha

GRID = np.array([16, 16, 8], dtype=np.int64)
TOKEN_CAPACITY = int(np.prod(GRID))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_source_groups(root, entry, output, chunk=65536):
    root, output = Path(root), Path(output)
    output.mkdir(exist_ok=False)
    raw_path = root / entry["raw_xyz_mm_path"]
    assert file_sha(raw_path) == entry["raw_xyz_file_sha256"]
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    assert raw.dtype == np.float64 and raw.shape == (entry["original_points"], 3)
    minimum, maximum, total = np.full(3, np.inf), np.full(3, -np.inf), np.zeros(3, np.float64)
    for first in range(0, len(raw), chunk):
        block = np.asarray(raw[first:first + chunk])
        if not np.isfinite(block).all():
            raise ValueError("Original source has nonfinite coordinates")
        minimum = np.minimum(minimum, block.min(axis=0))
        maximum = np.maximum(maximum, block.max(axis=0))
        total += block.sum(axis=0, dtype=np.float64)
    center = total / len(raw)
    extent = maximum - minimum
    # Degenerate source axes use one cell, without inventing spatial extent.
    divisions = np.divide(GRID, extent, out=np.zeros(3, np.float64), where=extent > 0)
    member_path = output / "original_row_group_ids.npy"
    members = np.lib.format.open_memmap(member_path, mode="w+", dtype=np.int16, shape=(len(raw),))
    counts = np.zeros(TOKEN_CAPACITY, np.int64)
    sums = np.zeros((TOKEN_CAPACITY, 3), np.float64)
    consumed = 0
    for first in range(0, len(raw), chunk):
        block = np.asarray(raw[first:first + chunk])
        cells = np.floor((block - minimum) * divisions).astype(np.int64)
        cells = np.minimum(np.maximum(cells, 0), GRID - 1)
        group = (cells[:, 0] * GRID[1] + cells[:, 1]) * GRID[2] + cells[:, 2]
        members[first:first + len(block)] = group
        counts += np.bincount(group, minlength=TOKEN_CAPACITY)
        np.add.at(sums, group, block)
        consumed += len(block)
    members.flush()
    del members
    assert consumed == len(raw) == int(counts.sum())
    means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    statistics = output / "original_source_group_statistics.npz"
    np.savez(statistics, counts=counts, mean_xyz_mm=means, center_mm=center,
             minimum_mm=minimum, maximum_mm=maximum)
    receipt = dict(status="COMPLETE_ORIGINAL_XYZ_ONLY_SPATIAL_GROUPS_285_v1",
        utc=datetime.now(timezone.utc).isoformat(), source_index_metadata_only=entry["source_index_metadata_only"],
        original_pcd_sha256=entry["original_pcd_sha256"], raw_xyz_mm_path=entry["raw_xyz_mm_path"],
        raw_xyz_file_sha256=entry["raw_xyz_file_sha256"], original_points=len(raw), original_rows_consumed=consumed,
        grid_shape=GRID.tolist(), global_token_capacity=TOKEN_CAPACITY, nonempty_groups=int((counts > 0).sum()),
        grouping_depends_on_original_XYZ_only=True, original_row_membership_preserved=True,
        group_centers_are_final_output_geometry=False, label_reads=0, STEP_reads=0,
        file_sha256={member_path.name:file_sha(member_path), statistics.name:file_sha(statistics)},
        actual_optimizer_updates=0, new_model_accuracy=False, full_contract_pass=False)
    dump(output / "receipt.json", receipt)
    return receipt


class RawPointSource:
    """No target, reference weld, workpiece identity or material truth is exposed."""
    def __init__(self, root, group_directory):
        root, directory = Path(root), Path(group_directory)
        receipt = read(directory / "receipt.json")
        if receipt["status"] != "COMPLETE_ORIGINAL_XYZ_ONLY_SPATIAL_GROUPS_285_v1":
            raise ValueError("Complete pure XYZ grouping required")
        if receipt["label_reads"] or receipt["STEP_reads"]:
            raise ValueError("Runtime source cannot read supervision")
        for name, expected in receipt["file_sha256"].items():
            if file_sha(directory / name) != expected:
                raise ValueError("Original source membership/statistics changed")
        path = root / receipt["raw_xyz_mm_path"]
        if file_sha(path) != receipt["raw_xyz_file_sha256"]:
            raise ValueError("Original XYZ file changed")
        self.xyz_mm = np.load(path, mmap_mode="r", allow_pickle=False)
        self.group_ids = np.load(directory / "original_row_group_ids.npy", mmap_mode="r", allow_pickle=False)
        with np.load(directory / "original_source_group_statistics.npz", allow_pickle=False) as values:
            self.counts = values["counts"].copy()
            self.mean_xyz_mm = values["mean_xyz_mm"].copy()
            self.center_mm = values["center_mm"].copy()
        if self.xyz_mm.shape != (receipt["original_points"], 3) or self.group_ids.shape != (len(self.xyz_mm),):
            raise ValueError("Runtime grouping omits original source rows")
        if int(self.counts.sum()) != len(self.xyz_mm) or self.counts.shape != (TOKEN_CAPACITY,):
            raise ValueError("Whole original point denominator required")

    def blocks(self, chunk):
        if type(chunk) is not int or chunk < 1:
            raise ValueError("Explicit original point block size required")
        for first in range(0, len(self.xyz_mm), chunk):
            last = min(first + chunk, len(self.xyz_mm))
            # Original float64 XYZ remain authoritative; centered metre features
            # are new float32 copies for the encoder alone.
            features = np.array((self.xyz_mm[first:last] - self.center_mm) / 1000., dtype=np.float32, copy=True)
            groups = np.array(self.group_ids[first:last], dtype=np.int64, copy=True)
            yield first, features, groups
