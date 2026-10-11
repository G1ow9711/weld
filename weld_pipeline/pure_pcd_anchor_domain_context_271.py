"""Label-free source inputs aligned to the complete fixed canonical slot pool.

Neural float32 channels describe sampled geometry. Exact permissions, carrier
parameters and authoritative raw world coordinates stay in the reviewed source
databases. Computational plane axes are not signed material normals.
"""
from __future__ import annotations

import math
import sqlite3
import struct
from pathlib import Path

import numpy as np

from weld_pipeline.original_curve_carriers_256 import read_fraction
from weld_pipeline.pure_pcd_initial_xyz_270 import initial_xyz_features

VERSION = "PURE_PCD_CANONICAL_ANCHOR_DOMAIN_CONTEXT_271_v1"
ANCHOR_CHANNELS = ["xyz_x", "xyz_y", "xyz_z", "raw_anchor", "triangle_anchor",
                   "log_surface_region_count", "log_surface_domain_count",
                   "log_carrier_region_count", "log_carrier_domain_count", "log_raw_portal_occurrence_count"]
COUNT_CHANNELS = ["surface_regions", "surface_domains", "carrier_regions", "carrier_domains", "raw_portal_occurrences"]
SURFACE_CHANNELS = (["chart_center_" + x for x in "xyz"] + ["computational_plane_axis_" + x for x in "xyz"]
                    + ["basis_u_" + x for x in "xyz"] + ["basis_v_" + x for x in "xyz"]
                    + ["uv_extent_u", "uv_extent_v", "z_extent", "log_unique_support_count",
                       "log_triangle_count", "log_boundary_edge_count", "log_scaled_sampled_chart_area",
                       "sampling_spacing_u", "sampling_spacing_v"])
CARRIER_CHANNELS = (["start_" + x for x in "xyz"] + ["end_" + x for x in "xyz"]
                    + ["occurrence_mean_" + x for x in "xyz"] + ["computational_plane_axis_" + x for x in "xyz"]
                    + ["extent_" + x for x in "xyz"] + ["first_raw_edge_delta_" + x for x in "xyz"]
                    + ["last_raw_edge_delta_" + x for x in "xyz"]
                    + ["scaled_native_length", "scaled_physical_length_lower", "scaled_physical_length_upper",
                       "log_edge_count", "closed", "positive_geometry", "log_source_path_alias_count", "log_source_face_count"])
SOURCE_CHANNELS = (["source_mean_" + x for x in "xyz"] + ["source_extent_" + x for x in "xyz"]
                   + ["source_std_" + x for x in "xyz"]
                   + ["source_cov_" + x for x in ("xx", "xy", "xz", "yy", "yz", "zz")]
                   + ["log_raw_point_count", "log_anchor_count", "log_surface_domain_count", "log_carrier_domain_count"])


def schema():
    return dict(version=VERSION, anchor_channels=ANCHOR_CHANNELS, exact_context_count_channels=COUNT_CHANNELS,
                surface_domain_channels=SURFACE_CHANNELS, carrier_domain_channels=CARRIER_CHANNELS,
                whole_source_context_channels=SOURCE_CHANNELS, feature_dtype="float32", geometry_dtype="float64",
                anchor_row_identity="anchor_id-1; complete contiguous canonical IDs only",
                surface_row_identity="ordered surface_face_ids.npy", carrier_row_identity="ordered carrier_ids.json",
                incidence_and_exact_permissions="unchanged reviewed source databases bound by SHA256",
                role_capacity_scope="one global canonical anchor pool; aliases add no capacity",
                computational_plane_axes_are_material_normals=False, material_channels=[],
                label_geometry_or_teacher_selection_used=False, source_names_or_IDs_are_neural_channels=False,
                train_only_coordinate_normalizer="270", other_transforms="fixed log1p/count or physical scale; no fitted development statistics",
                full_contract_pass=False)


def readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA mmap_size=0")
    db.execute("PRAGMA temp_store=FILE")
    return db


def plane_axis(key):
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("Reviewed canonical computational plane required")
    values = np.asarray(struct.unpack("<4d", key), dtype=np.float64)
    if not np.isfinite(values).all() or abs(float(values[:3] @ values[:3]) - 1) > 2e-9:
        raise ValueError("Finite numerical unit computational plane axis required")
    return values[:3]


def surface_vector(descriptor, key, normalizer):
    scale = normalizer["isotropic_scale_mm"]
    bounds = np.asarray(descriptor["uv_bounds_mm"], dtype=np.float64)
    basis = np.asarray(descriptor["basis_uv"], dtype=np.float64)
    origin = np.asarray(descriptor["origin_mm"], dtype=np.float64)
    if bounds.shape != (2, 2) or basis.shape != (2, 3) or origin.shape != (3,):
        raise ValueError("Complete finite source chart descriptor required")
    extent = bounds[1] - bounds[0]
    if (extent < 0).any():
        raise ValueError("Ordered finite chart bounds required")
    center = origin + bounds.mean(axis=0) @ basis
    coordinates = initial_xyz_features(center.reshape(1, 3), normalizer)[0]
    z = descriptor["z_bounds_mm"]
    area = float(descriptor["sampled_triangle_area_mm2"])
    if area < 0 or z[1] < z[0]:
        raise ValueError("Nonnegative sampled chart dimensions required")
    result = np.r_[coordinates, plane_axis(key), basis.reshape(-1), extent / scale,
                   (z[1] - z[0]) / scale, math.log1p(descriptor["unique_support_count"]),
                   math.log1p(descriptor["triangle_count"]), math.log1p(descriptor["boundary_edge_count"]),
                   math.log1p(area / scale**2), np.asarray(descriptor["sampling_spacing_uv_mm"]) / scale]
    if len(result) != len(SURFACE_CHANNELS) or not np.isfinite(result).all():
        raise ValueError("Finite fixed-schema surface features required")
    return result.astype(np.float32)


def carrier_vector(raw, ids, closed, key, metric, aliases, faces, normalizer):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) < 2 or (ids < 0).any() or (ids >= len(raw)).any():
        raise ValueError("Complete original raw carrier occurrence IDs required")
    if closed and ids[0] != ids[-1]:
        raise ValueError("Explicit original closing occurrence required")
    points = np.asarray(raw[ids], dtype=np.float64)
    core = points[:-1] if closed else points
    scale = normalizer["isotropic_scale_mm"]
    xyz = initial_xyz_features(np.vstack((points[0], points[-1], core.mean(axis=0))), normalizer).reshape(-1)
    native = read_fraction(metric["native_length_mm"])
    lower = read_fraction(metric["physical_length_lower_mm"])
    upper = read_fraction(metric["physical_length_upper_mm"])
    if not 0 <= lower <= upper or native < 0 or metric["edge_count"] != len(ids) - 1:
        raise ValueError("Complete sealed source carrier metric required")
    result = np.r_[xyz, plane_axis(key), np.ptp(core, axis=0) / scale,
                   (points[1] - points[0]) / scale, (points[-1] - points[-2]) / scale,
                   float(native) / scale, float(lower) / scale, float(upper) / scale,
                   math.log1p(metric["edge_count"]), int(closed), int(native > 0), math.log1p(aliases), math.log1p(faces)]
    if len(result) != len(CARRIER_CHANNELS) or not np.isfinite(result).all():
        raise ValueError("Finite fixed-schema carrier features required")
    return result.astype(np.float32)


def source_vector(statistics, anchors, surfaces, carriers, normalizer):
    mean = np.asarray(statistics["mean_mm"], dtype=np.float64)
    covariance = np.asarray(statistics["population_covariance_mm2"], dtype=np.float64)
    extent = np.asarray(statistics["maximum_mm"]) - np.asarray(statistics["minimum_mm"])
    scale = normalizer["isotropic_scale_mm"]
    diagonal = covariance.diagonal()
    if (diagonal < 0).any():
        raise ValueError("Nonnegative whole-source variances required")
    result = np.r_[initial_xyz_features(mean.reshape(1, 3), normalizer)[0], extent / scale,
                   np.sqrt(diagonal) / scale, covariance[np.triu_indices(3)] / scale**2,
                   [math.log1p(x) for x in (statistics["points"], anchors, surfaces, carriers)]]
    if len(result) != len(SOURCE_CHANNELS) or not np.isfinite(result).all():
        raise ValueError("Finite fixed-schema source context required")
    return result.astype(np.float32)


def build_features(surface_db, carrier_db, raw, statistics, normalizer, output, progress, gate):
    """Stream every canonical anchor and domain; never select a teacher subset."""
    if raw.dtype != np.float64 or raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("Original authoritative float64 world rows required")
    total, first, last = surface_db.execute("SELECT COUNT(*),MIN(anchor_id),MAX(anchor_id) FROM anchors").fetchone()
    if not total or (first, last) != (1, total):
        raise ValueError("Complete contiguous canonical anchor pool required")
    context = np.lib.format.open_memmap(output / "anchor_context_counts.npy", mode="w+", dtype=np.uint32, shape=(total, 5))
    context[:] = 0
    count_sql = [
        (surface_db, "SELECT anchor_id,COUNT(*),COUNT(DISTINCT face_id) FROM regions GROUP BY anchor_id", (0, 1)),
        (carrier_db, "SELECT anchor_id,COUNT(*),COUNT(DISTINCT carrier_id) FROM carrier_regions GROUP BY anchor_id", (2, 3)),
        (carrier_db, "SELECT anchor_id,COUNT(*) FROM raw_portal_members GROUP BY anchor_id", (4,))]
    for db, sql, columns in count_sql:
        cursor = db.execute(sql)
        checked = 0
        while True:
            rows = cursor.fetchmany(32768)
            if not rows:
                break
            values = np.asarray(rows, dtype=np.int64)
            if (values[:, 0] < 1).any() or (values[:, 0] > total).any() or (values[:, 1:] < 0).any() or (values[:, 1:] > np.iinfo(np.uint32).max).any():
                raise ValueError("Source context references must remain within the existing slot pool")
            for offset, column in enumerate(columns, 1):
                context[values[:, 0] - 1, column] = values[:, offset].astype(np.uint32)
            checked += len(rows)
            progress("COMPLETE_SOURCE_INCIDENCE_COUNTS", dict(context_columns=list(columns), groups=checked))
            gate()
    context.flush()
    features = np.lib.format.open_memmap(output / "anchor_features.npy", mode="w+", dtype=np.float32, shape=(total, len(ANCHOR_CHANNELS)))
    cursor = surface_db.execute("SELECT anchor_id,kind,seed_x,seed_y,seed_z,raw_a,raw_b,raw_c FROM anchors ORDER BY anchor_id")
    complete = 0
    while True:
        rows = cursor.fetchmany(32768)
        if not rows:
            break
        ids = np.asarray([r[0] for r in rows], dtype=np.int64)
        if not np.array_equal(ids, np.arange(complete + 1, complete + len(rows) + 1)):
            raise ValueError("Canonical source order or capacity changed")
        seeds = np.asarray([r[2:5] for r in rows], dtype=np.float64)
        raw_flags = np.asarray([r[1] == "RAW_REPRESENTATIVE" for r in rows])
        tri_flags = np.asarray([r[1] == "TRIANGLE_CENTROID" for r in rows])
        if not np.all(raw_flags ^ tri_flags):
            raise ValueError("Only the two frozen source anchor kinds are permitted")
        raw_rows = np.asarray([r[5] for r in rows], dtype=np.int64)
        expected = np.asarray(raw[raw_rows], dtype=np.float64).copy()
        selected = np.flatnonzero(tri_flags)
        if len(selected):
            triple_ids = np.asarray([rows[i][5:8] for i in selected], dtype=np.int64)
            expected[selected] = raw[triple_ids].mean(axis=1)
        if not np.array_equal(seeds, expected):
            raise ValueError("Anchor seeds no longer equal the original raw source barycenters")
        start, stop = complete, complete + len(rows)
        features[start:stop, :3] = initial_xyz_features(seeds, normalizer)
        features[start:stop, 3] = raw_flags
        features[start:stop, 4] = tri_flags
        features[start:stop, 5:] = np.log1p(np.asarray(context[start:stop], dtype=np.float64)).astype(np.float32)
        complete = stop
        progress("ALL_CANONICAL_ANCHOR_INPUTS", dict(complete_anchors=complete, total_anchors=total))
        gate()
    assert complete == total
    features.flush()
    surface_rows = list(surface_db.execute("SELECT face_id,descriptor_json,plane_key FROM domains ORDER BY face_id"))
    surface_features = np.empty((len(surface_rows), len(SURFACE_CHANNELS)), dtype=np.float32)
    for index, (face, descriptor_json, key) in enumerate(surface_rows):
        import json
        surface_features[index] = surface_vector(json.loads(descriptor_json), key, normalizer)
    np.save(output / "surface_face_ids.npy", np.asarray([r[0] for r in surface_rows], dtype=np.int64), allow_pickle=False)
    np.save(output / "surface_domain_features.npy", surface_features, allow_pickle=False)
    carrier_count = carrier_db.execute("SELECT COUNT(*) FROM carriers").fetchone()[0]
    carrier_features = np.lib.format.open_memmap(output / "carrier_domain_features.npy", mode="w+", dtype=np.float32, shape=(carrier_count, len(CARRIER_CHANNELS)))
    aliases = {r[0]: r[1:] for r in carrier_db.execute("SELECT carrier_id,COUNT(*),COUNT(DISTINCT face_id) FROM source_path_aliases GROUP BY carrier_id")}
    carrier_ids = []
    for index, (cid, key, closed, ids_json, metric_json) in enumerate(carrier_db.execute("SELECT carrier_id,plane_key,is_closed,raw_row_ids_json,metric_json FROM carriers ORDER BY carrier_id")):
        import json
        carrier_ids.append(cid)
        count_aliases, count_faces = aliases[cid]
        carrier_features[index] = carrier_vector(raw, json.loads(ids_json), bool(closed), key, json.loads(metric_json), count_aliases, count_faces, normalizer)
        progress("ALL_ORIGINAL_CARRIER_DOMAIN_INPUTS", dict(complete_carriers=index + 1, total_carriers=carrier_count))
        gate()
    assert len(carrier_ids) == carrier_count
    carrier_features.flush()
    import json
    (output / "carrier_ids.json").write_text(json.dumps(carrier_ids) + "\n", encoding="utf-8")
    np.save(output / "whole_source_context.npy", source_vector(statistics, total, len(surface_rows), carrier_count, normalizer), allow_pickle=False)
    del context, features, carrier_features
    return dict(canonical_anchors=total, surface_domains=len(surface_rows), carrier_domains=carrier_count,
                original_surface_regions=surface_db.execute("SELECT COUNT(*) FROM regions").fetchone()[0],
                original_carrier_regions=carrier_db.execute("SELECT COUNT(*) FROM carrier_regions").fetchone()[0],
                original_raw_portal_occurrences=carrier_db.execute("SELECT COUNT(*) FROM raw_portal_members").fetchone()[0])
