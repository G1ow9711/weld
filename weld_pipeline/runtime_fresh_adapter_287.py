"""Fresh ASCII XYZ PCD adapter for the unchanged B neural runtime.

Copied into the runtime package as raw_point_groups_285.py. All original
rows contribute. Coordinates, group statistics and ordering match training.
Input units and the transform are global configuration, never file lookup.
"""
from hashlib import sha256
from pathlib import Path
import numpy as np

GRID = np.array([16, 16, 8], dtype=np.int64)
TOKEN_CAPACITY = int(np.prod(GRID))


def array_sha(values):
    if not values.flags.c_contiguous:
        raise ValueError("Authoritative coordinate array must be contiguous")
    return sha256(memoryview(values).cast("B")).hexdigest()


def file_sha(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


class RawPointSource:
    def __init__(self, path, configuration, chunk=65536):
        path = Path(path)
        input_sha = file_sha(path)
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
                raise ValueError("INPUT_FORMAT_UNSUPPORTED: expected ASCII PCD with exactly x y z")
            xyz = np.loadtxt(stream, dtype=np.float64, ndmin=2)
        declared = int(header["POINTS"])
        if declared < 1 or xyz.shape != (declared, 3) or not np.isfinite(xyz).all():
            raise ValueError("INPUT_POINT_COUNT_OR_VALUES_INVALID")
        if file_sha(path) != input_sha:
            raise ValueError("INPUT_CHANGED_DURING_FRESH_READ")
        units = configuration["original_pcd_units"]
        if units not in ("m", "mm") or configuration["output_units"] != "mm":
            raise ValueError("GLOBAL_UNIT_CONFIGURATION_INVALID")
        xyz *= 1000. if units == "m" else 1.
        transform = np.asarray(configuration["after_unit_conversion_to_output_transform_4x4"], dtype=np.float64)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.array_equal(transform[3], [0., 0., 0., 1.])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-10, rtol=0)
                or abs(float(np.linalg.det(transform[:3, :3]))-1.) > 1e-10):
            raise ValueError("GLOBAL_RIGHT_HANDED_RIGID_TRANSFORM_INVALID")
        if not np.array_equal(transform, np.eye(4)):
            xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        self.xyz_mm = np.ascontiguousarray(xyz)
        minimum, maximum, total = np.full(3, np.inf), np.full(3, -np.inf), np.zeros(3, np.float64)
        for first in range(0, len(xyz), chunk):
            block = xyz[first:first+chunk]
            minimum = np.minimum(minimum, block.min(axis=0))
            maximum = np.maximum(maximum, block.max(axis=0))
            total += block.sum(axis=0, dtype=np.float64)
        self.center_mm = total / len(xyz)
        extent = maximum-minimum
        divisions = np.divide(GRID, extent, out=np.zeros(3, np.float64), where=extent > 0)
        self.group_ids = np.empty(len(xyz), dtype=np.int16)
        self.counts = np.zeros(TOKEN_CAPACITY, np.int64)
        sums = np.zeros((TOKEN_CAPACITY, 3), np.float64)
        consumed = 0
        for first in range(0, len(xyz), chunk):
            block = xyz[first:first+chunk]
            cells = np.floor((block-minimum)*divisions).astype(np.int64)
            cells = np.minimum(np.maximum(cells, 0), GRID-1)
            groups = (cells[:, 0]*GRID[1]+cells[:, 1])*GRID[2]+cells[:, 2]
            self.group_ids[first:first+len(block)] = groups
            self.counts += np.bincount(groups, minlength=TOKEN_CAPACITY)
            np.add.at(sums, groups, block)
            consumed += len(block)
        if consumed != declared or int(self.counts.sum()) != declared:
            raise ValueError("FRESH_INPUT_GROUPING_OMITTED_ORIGINAL_ROWS")
        self.mean_xyz_mm = np.divide(sums, self.counts[:, None], out=np.zeros_like(sums), where=self.counts[:, None] > 0)
        self.fresh_read_receipt = dict(original_pcd_sha256=input_sha, original_points=declared,
            original_rows_grouped=consumed, original_pcd_units=units, output_units="mm",
            unit_multiplier=1000. if units == "m" else 1.,
            transform_4x4=transform.tolist(), transform_direction="AFTER_UNIT_CONVERSION_TO_OUTPUT_MM",
            multiplication_convention="COLUMN_HOMOGENEOUS_OUT=T_IN; ROW_ARRAY_OUT=IN@R.T+t",
            handedness="RIGHT", original_XYZ_mm_array_sha256=array_sha(self.xyz_mm),
            original_row_group_ids_sha256=array_sha(self.group_ids),
            nonempty_groups=int((self.counts > 0).sum()), group_capacity=TOKEN_CAPACITY,
            grouping_grid=GRID.tolist(), label_reads=0, STEP_reads=0,
            input_name_is_model_feature=False, input_preprocessing_cache_reused=False,
            predicted_cache_reused=False, attribute_truth_or_count_read=False)

    def blocks(self, chunk):
        if type(chunk) is not int or chunk < 1:
            raise ValueError("Explicit original point block size required")
        for first in range(0, len(self.xyz_mm), chunk):
            last = min(first+chunk, len(self.xyz_mm))
            features = np.array((self.xyz_mm[first:last]-self.center_mm)/1000., dtype=np.float32, copy=True)
            groups = np.array(self.group_ids[first:last], dtype=np.int64, copy=True)
            yield first, features, groups
