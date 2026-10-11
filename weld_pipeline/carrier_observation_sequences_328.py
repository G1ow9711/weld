"""Every finite carrier cell receives original XYZ local observations.

There is no supervision in this module. The layout includes TWO independent
slots per <=1 mm cell plus the frozen finite/event slots. Every row is produced
from the same source-only procedure; no GT window or requested output count.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import cKDTree
from .bounded_analytic_residuals_327 import line_transverse_basis
from .raw_point_groups_285 import RawPointSource

VERSION = "PURE_XYZ_ALL_FINITE_CELL_LOCAL_OBSERVATIONS_328_v1"
FEATURE_WIDTH = 64
LOCAL_NEIGHBORS = 8
FEATURE_BLOCK_ROWS = 4096
FOURIER_PERIODS_MM = np.array([2., 4., 8., 16., 32., 64., 128., 256., 512., 1024.])


@dataclass
class SequenceLayout:
    carrier: object
    lower_mm: object
    upper_mm: object
    positions_mm: object
    slot_ids: object
    row_kind: object
    cell_to_row: object
    extra_to_row: object

    @classmethod
    def from_carrier(cls, carrier):
        count = carrier.cell_count
        lower = np.arange(count, dtype=np.float64) * carrier.cell_width_mm
        upper = np.r_[lower[1:], carrier.length_mm]
        position = (lower + upper) / 2.
        extras = np.array([0., carrier.length_mm, *carrier.structural_events_mm], np.float64)
        positions = np.r_[position, extras]
        slots = np.vstack((np.arange(2 * count, dtype=np.int64).reshape(count, 2),
            np.column_stack((np.arange(2 * count, carrier.slot_count, dtype=np.int64), np.full(len(extras), -1, np.int64)))))
        kinds = np.r_[np.zeros(count, np.int8), np.array([1, 1, *([2] * len(carrier.structural_events_mm))], np.int8)]
        order = np.argsort(positions, kind="stable")
        inverse = np.empty(len(order), np.int64); inverse[order] = np.arange(len(order))
        return cls(carrier, np.r_[lower, extras][order], np.r_[upper, extras][order], positions[order],
            slots[order], kinds[order], inverse[:count], inverse[count:])

    @property
    def rows(self):
        return len(self.positions_mm)

    @property
    def active_slots(self):
        return self.slot_ids >= 0

    def location(self, slot):
        if not 0 <= slot < self.carrier.slot_count:
            raise ValueError("Declared boundary slot required")
        if slot < 2 * self.carrier.cell_count:
            cell, subslot = divmod(int(slot), 2)
            return int(self.cell_to_row[cell]), subslot
        return int(self.extra_to_row[slot - 2 * self.carrier.cell_count]), 0

    def parameter(self, slot, fraction):
        row, _ = self.location(slot)
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Within-cell boundary fraction in [0,1] required")
        return float(self.lower_mm[row] + fraction * (self.upper_mm[row] - self.lower_mm[row]))


class OriginalXYZObservationProvider:
    def __init__(self, source):
        if type(source) is not RawPointSource:
            raise ValueError("Only verified original XYZ can provide model observations")
        self.source = source
        self.tree = cKDTree(source.xyz_mm)
        self.available_groups = np.flatnonzero(source.counts > 0)
        self.group_tree = cKDTree(source.mean_xyz_mm[self.available_groups])
        if len(source.xyz_mm) < LOCAL_NEIGHBORS:
            raise ValueError("Insufficient raw local support")

    def observations(self, layout, first=0, last=None):
        last = layout.rows if last is None else last
        if not 0 <= first < last <= layout.rows:
            raise ValueError("Nonempty complete sequence block required")
        carrier = layout.carrier
        positions = layout.positions_mm[first:last]
        world = carrier.point_at(positions)
        if carrier.kind == "line":
            tangent = np.broadcast_to(carrier.direction, world.shape)
            basis = line_transverse_basis(carrier.direction)
            first_axis, second_axis = np.broadcast_to(basis[0], world.shape), np.broadcast_to(basis[1], world.shape)
            radius = 0.
        else:
            radial = (world - carrier.arc.center_mm) / carrier.arc.radius_mm
            tangent = np.sign(carrier.arc.sweep_rad) * np.cross(carrier.arc.plane_normal, radial)
            first_axis = np.broadcast_to(carrier.arc.plane_normal, world.shape)
            second_axis = np.cross(tangent, first_axis)
            radius = carrier.arc.radius_mm
        distances, indices = self.tree.query(world, k=LOCAL_NEIGHBORS, workers=1)
        relative = self.source.xyz_mm[indices] - world[:, None]
        frames = np.stack((tangent, first_axis, second_axis), axis=1)
        coordinates = np.einsum("nki,nai->nka", relative, frames)
        local = np.concatenate((coordinates.mean(1), coordinates.std(1), coordinates.min(1), coordinates.max(1)), axis=1) / 10.
        distance_stats = np.column_stack((distances.min(1), distances.mean(1), distances.std(1), distances.max(1))) / 10.
        if carrier.kind == "line":
            perpendicular = np.linalg.norm(coordinates[:, :, 1:], axis=2)
        else:
            raw_radial = self.source.xyz_mm[indices] - carrier.arc.center_mm
            heights = raw_radial @ carrier.arc.plane_normal
            plane = raw_radial - heights[..., None] * carrier.arc.plane_normal
            perpendicular = np.hypot(heights, np.linalg.norm(plane, axis=2) - radius)
        perpendicular_stats = np.log1p(np.column_stack((perpendicular.min(1), perpendicular.mean(1), perpendicular.std(1), perpendicular.max(1))) / .001)
        features = np.empty((last - first, FEATURE_WIDTH), np.float32)
        features[:, :3] = (world - self.source.center_mm) / 1000.
        features[:, 3:6] = world / 1000.
        features[:, 6:9], features[:, 9:12], features[:, 12:15] = tangent, first_axis, second_axis
        features[:, 15] = math.log1p(carrier.length_mm) / math.log(10001.)
        features[:, 16] = math.log1p(radius) / math.log(10001.)
        features[:, 17], features[:, 18] = float(carrier.kind == "line"), float(carrier.weld_type == "vertical")
        features[:, 19], features[:, 20] = positions / carrier.length_mm, positions / 1000.
        features[:, 21] = layout.upper_mm[first:last] - layout.lower_mm[first:last]
        features[:, 22], features[:, 23] = layout.row_kind[first:last] == 1, layout.row_kind[first:last] == 2
        features[:, 24:36], features[:, 36:40], features[:, 40:44] = local, distance_stats, perpendicular_stats
        angle = positions[:, None] * (2 * np.pi) / FOURIER_PERIODS_MM[None]
        features[:, 44:] = np.stack((np.sin(angle), np.cos(angle)), axis=2).reshape(len(positions), 20)
        _, groups = self.group_tree.query(world, workers=1)
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite pure XYZ observation")
        return features, np.asarray(groups, np.int64)
