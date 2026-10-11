"""Build the existing encoder input directly from complete original XYZ.

This factory has no file, workpiece-name, label, STEP or candidate-bank argument.
Its grouping arithmetic preserves raw_point_groups_285's fixed grid and row
order. It is preparation for a future runtime; existing registered runs do not
import this module.
"""
import numpy as np

from .raw_point_groups_285 import RawPointSource, TOKEN_CAPACITY


_GRID = np.array([16, 16, 8], dtype=np.int64)
_GROUPING_BLOCK = 65536


def source_from_original_xyz_mm(xyz_mm):
    """Return RawPointSource with all inputs freshly derived from an XYZ array.

    The deployed encoder requires this exact class. Allocate that class without
    its disk-cache-loading initializer, then populate and validate its five
    ordinary data attributes. The existing class and encoder remain unchanged.
    """
    if type(xyz_mm) not in (np.ndarray, np.memmap):
        raise TypeError("Complete ordinary NumPy XYZ array required")
    if xyz_mm.dtype != np.float64 or xyz_mm.ndim != 2 or xyz_mm.shape[1] != 3:
        raise ValueError("Original float64 XYZ in millimetres required")
    if len(xyz_mm) == 0:
        raise ValueError("Original XYZ must contain at least one row")
    raw = xyz_mm.view(np.ndarray)
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    total = np.zeros(3, np.float64)
    for first in range(0, len(raw), _GROUPING_BLOCK):
        block = raw[first:first + _GROUPING_BLOCK]
        if not np.isfinite(block).all():
            raise ValueError("Original XYZ contains nonfinite coordinates")
        minimum = np.minimum(minimum, block.min(axis=0))
        maximum = np.maximum(maximum, block.max(axis=0))
        total += block.sum(axis=0, dtype=np.float64)
    center = total / len(raw)
    extent = maximum - minimum
    if not np.isfinite(center).all() or not np.isfinite(extent).all():
        raise ValueError("Original XYZ range or sum exceeds float64 capacity")
    divisions = np.divide(_GRID, extent, out=np.zeros(3, np.float64), where=extent > 0)
    members = np.empty(len(raw), np.int16)
    counts = np.zeros(TOKEN_CAPACITY, np.int64)
    sums = np.zeros((TOKEN_CAPACITY, 3), np.float64)
    consumed = 0
    for first in range(0, len(raw), _GROUPING_BLOCK):
        block = raw[first:first + _GROUPING_BLOCK]
        cells = np.floor((block - minimum) * divisions).astype(np.int64)
        cells = np.minimum(np.maximum(cells, 0), _GRID - 1)
        groups = (cells[:, 0] * _GRID[1] + cells[:, 1]) * _GRID[2] + cells[:, 2]
        members[first:first + len(block)] = groups
        counts += np.bincount(groups, minlength=TOKEN_CAPACITY)
        np.add.at(sums, groups, block)
        consumed += len(block)
    means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    if consumed != len(raw) or int(counts.sum()) != len(raw):
        raise RuntimeError("Complete original row denominator was not preserved")
    if not np.isfinite(means).all():
        raise ValueError("Original XYZ group sums exceed float64 capacity")
    source = object.__new__(RawPointSource)
    source.xyz_mm, source.group_ids = raw, members
    source.counts, source.mean_xyz_mm, source.center_mm = counts, means, center
    for value in (raw, members, counts, means, center):
        value.flags.writeable = False
    return source
