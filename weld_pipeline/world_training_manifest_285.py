"""Register all25 world targets separately from original XYZ runtime inputs."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .contract_curve_geometry import fit_ordered_circle, sample_arc, verify_sampled_arc
from .live_json_snapshot_282 import write_json_snapshot as dump
from .raw_triangle_neural_features_279 import file_sha

MODES = ("LINE_CHAIN", "CIRCLE_CHAIN", "MIXED_CHAIN")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def power_two(value):
    return 1 << int(math.ceil(math.log2(value)))


def _line_points(points, fractions):
    return points[0] + np.asarray(fractions)[:, None] * (points[-1] - points[0])


def parent_target(children):
    """Loss-only representation. Keep every original child and ordered endpoint."""
    children = sorted(children, key=lambda child: child["segment_index"])
    if [child["segment_index"] for child in children] != list(range(len(children))):
        raise ValueError("Original child order has a hole or duplicate")
    if any(child["segment_count"] != len(children) for child in children):
        raise ValueError("Original parent/child cardinality differs")
    reference = np.linspace(0., 1., 17)
    targets, arcs, maximum = [], [], 0.
    for child in children:
        points = np.asarray(child["points"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("Invalid original child coordinates")
        if child["kind"] == "line":
            delta = points[-1] - points[0]
            length = float(np.linalg.norm(delta))
            if length <= 0:
                raise ValueError("Original line has coincident endpoints")
            residual = float(np.linalg.norm(np.cross(points - points[0], delta / length), axis=1).max())
            parameter = (points - points[0]) @ delta / (length * length)
            if residual > 1e-5 or np.any(np.diff(parameter) <= 0):
                raise ValueError("Original line label is not an ordered world line")
            samples = _line_points(points, reference)
            representation = dict(kind="line", endpoints_mm=points[[0, -1]].tolist())
            arcs.append(None)
        elif child["kind"] == "arc":
            arc = fit_ordered_circle(points, tolerance_mm=1e-6, is_closed=child["is_closed"])
            exported = sample_arc(arc)
            verify_sampled_arc(exported)
            samples = arc.point_at(reference * arc.length_mm)
            residual = max(float(np.linalg.norm(samples[0] - points[0])),
                           float(np.linalg.norm(samples[-1] - points[-1])))
            representation = dict(kind="arc", arc_parameters=arc.to_parameters())
            arcs.append(arc)
        else:
            raise ValueError("Original geometry type is unsupported; retain failure, do not relabel")
        maximum = max(maximum, residual)
        targets.append(dict(original_child_id=child["id"], original_segment_index=child["segment_index"],
            original_parent_id=child["parent_id"], weld_type=child["type"], original_kind=child["kind"],
            world_representation=representation, original_points_mm=child["points"],
            loss_samples_mm=samples.tolist(), original_weld_side=child["weld_side"],
            original_is_closed=child["is_closed"], original_attributes=child["attributes"],
            formal_E_reviewed=False))
    gaps = ["SHARED" if np.array_equal(np.asarray(left["points"][-1]), np.asarray(right["points"][0]))
            else "POSITIVE_GAP" for left, right in zip(children[:-1], children[1:])]
    start = np.asarray(children[0]["points"][0], np.float64)
    end = np.asarray(children[-1]["points"][-1], np.float64)
    delta = end - start
    norm = float(np.linalg.norm(delta))
    mode = "MIXED_CHAIN"
    if all(child["kind"] == "line" for child in children) and norm > 0:
        all_points = np.concatenate([np.asarray(child["points"], np.float64) for child in children])
        deviation = float(np.linalg.norm(np.cross(all_points - start, delta / norm), axis=1).max())
        parameters = [(np.asarray(child["points"])[[0, -1]] - start) @ delta / (norm * norm) for child in children]
        if deviation <= 1e-5 and all(value[1] > value[0] for value in parameters):
            mode = "LINE_CHAIN"
    elif all(arc is not None for arc in arcs):
        first = arcs[0]
        same_circle = all(float(np.linalg.norm(arc.center_mm - first.center_mm)) <= 1e-5
            and abs(arc.radius_mm - first.radius_mm) <= 1e-5
            and float(arc.plane_normal @ first.plane_normal) > 1 - 1e-10 for arc in arcs)
        if same_circle:
            mode = "CIRCLE_CHAIN"
    return dict(original_parent_index=children[0]["parent_index"], original_parent_id=children[0]["parent_id"],
        mode=mode, children=targets, gaps=gaps, original_child_count=len(children),
        endpoints_mm=[start.tolist(), end.tolist()], is_closed=np.array_equal(start, end),
        target_representation_maximum_endpoint_residual_mm=maximum)


def build_manifest(root, report):
    root, report = Path(root), Path(report)
    output = report / "actual_full25_world_training_manifest_attempt1"
    output.mkdir(exist_ok=False)
    scope_path = root / "weld_pipeline/reports/pure_pcd_initial_features_270_20261003/actual_complete25_initial_xyz_component_attempt1/full_original25_initial_xyz_scope.json"
    scope = read(scope_path)
    assert len(scope) == 25 and [entry["source_index"] for entry in scope] == list(range(25))
    assert len({entry["original_pcd_sha256"] for entry in scope}) == 25
    counts, mode_counts = Counter(), Counter()
    inputs, training = [], []
    hashes = {str(scope_path.relative_to(root)): file_sha(scope_path)}
    maximum_parents, maximum_children, shared, positive, total = 0, 0, 0, 0, 0
    for entry in scope:
        source_index = entry["source_index"]
        source_receipt = root / entry["original_float64_xyz_path"]
        geometry_receipt = read(source_receipt.parent / "receipt.json")
        assert geometry_receipt["source_index"] == source_index
        assert geometry_receipt["original_pcd"]["sha256"] == entry["original_pcd_sha256"]
        input_file_sha = geometry_receipt["source_artifacts"]["fresh_original_xyz_mm.npy"]["sha256"]
        assert file_sha(source_receipt) == input_file_sha
        raw = np.load(source_receipt, mmap_mode="r", allow_pickle=False)
        assert raw.shape == (entry["original_points"], 3) and raw.dtype == np.float64
        # Metadata and losses alone know the old workpiece identity.
        target_path = root / "weld_pipeline/data/contract_child_v1" / (entry["piece_metadata_only"] + "_children.json")
        original = read(target_path)
        groups = {}
        for child in original:
            if child["workpiece_id"] != entry["piece_metadata_only"]:
                raise ValueError("Original label and source identity differ")
            groups.setdefault(child["parent_index"], []).append(child)
            counts[child["type"] + "_" + child["kind"]] += 1
        if sorted(groups) != list(range(len(groups))):
            raise ValueError("Original parent ledger is not complete")
        parents = [parent_target(groups[index]) for index in range(len(groups))]
        assert sum(parent["original_child_count"] for parent in parents) == len(original)
        assert len({child["original_child_id"] for parent in parents for child in parent["children"]}) == len(original)
        maximum_parents = max(maximum_parents, len(parents))
        maximum_children = max(maximum_children, max(parent["original_child_count"] for parent in parents))
        shared += sum(parent["gaps"].count("SHARED") for parent in parents)
        positive += sum(parent["gaps"].count("POSITIVE_GAP") for parent in parents)
        total += len(original)
        mode_counts.update(parent["mode"] for parent in parents)
        target_file = output / ("source_%02d_loss_targets.json" % source_index)
        dump(target_file, dict(source_index_metadata_only=source_index, supervision_only=True,
            original_parents=len(parents), original_children=len(original), parents=parents,
            formal_E_reviewed_cells=0, original_reference_path=str(target_path.relative_to(root)),
            original_reference_sha256=file_sha(target_path)))
        hashes[str(target_path.relative_to(root))] = file_sha(target_path)
        inputs.append(dict(source_index_metadata_only=source_index, original_role_metadata_only=entry["role_metadata_only"],
            original_pcd_path=geometry_receipt["original_pcd"]["path"], original_pcd_sha256=entry["original_pcd_sha256"],
            raw_xyz_mm_path=str(source_receipt.relative_to(root)), raw_xyz_file_sha256=input_file_sha,
            original_points=entry["original_points"], input_units="mm", source_conversion="ORIGINAL_m_TO_mm_ONCE",
            model_feature_inputs=["XYZ_ONLY"], original_pcd_name_is_feature=False))
        training.append(dict(source_index_metadata_only=source_index, role="FIT_NOT_INDEPENDENT",
            original_role=entry["role_metadata_only"], loss_target_path=str(target_file.relative_to(root)),
            loss_target_sha256=file_sha(target_file), original_parents=len(parents), original_children=len(original)))
        dump(output / "progress.json", dict(utc=datetime.now(timezone.utc).isoformat(), complete_sources=len(inputs),
            complete_children=total, full_original_sources=25, full_original_children=1605,
            actual_optimizer_updates=0, full_contract_pass=False))
    assert total == 1605 and sum(row["original_parents"] for row in training) == 1461
    assert counts == Counter(horizontal_line=1443, vertical_line=138, horizontal_arc=24)
    assert shared == 144 and positive == 0
    input_path, training_path = output / "world_source_only_inputs.json", output / "world_supervision_manifest.json"
    dump(input_path, inputs)
    dump(training_path, training)
    receipt = dict(version="ALL25_WORLD_SET_PARENT_SEQUENCE_FIT_285_v1", utc=datetime.now(timezone.utc).isoformat(),
        original_sources=25, original_points=sum(row["original_points"] for row in inputs), original_children=total,
        original_parents=1461, categories=dict(counts), parent_modes=dict(mode_counts),
        shared_cut_count=shared, positive_gap_count=positive, maximum_FIT_parents=maximum_parents,
        maximum_FIT_children_per_parent=maximum_children,
        global_parent_query_capacity=power_two(max(256, 2 * maximum_parents)),
        global_child_sequence_capacity=power_two(max(16, 2 * maximum_children)),
        global_source_tokens=2048, all_original_XYZ_contribute_to_pooling=True,
        purpose="ALL25_FIT_CAPACITY_TEST_NOT_INDEPENDENT_ACCEPTANCE", exposed_original_DEV_sources=[7, 14, 16, 17],
        formal_E_reviewed_cells=0, original_source_and_reference_bindings=hashes,
        input_scope_sha256=file_sha(input_path), supervision_manifest_sha256=file_sha(training_path),
        input_target_separation=True, GT_runtime_queries_or_count_supplied=False,
        first_real_neural_update_requires_full1605_targets_not_full_triangle_domain_enumeration=True,
        actual_optimizer_updates=0, new_model_accuracy=False, full_contract_pass=False)
    dump(output / "receipt.json", receipt)
    return receipt
