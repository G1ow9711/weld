"""Explain TRAIN arc witness gaps with unchanged input curves and D3 rules.

Reference projections here are offline diagnostics, never deployed features or
replacement input curves. Both finite-arc and underlying-circle distances are
reported; proximity to an underlying circle does not establish finite contact.
"""
import argparse
from collections import Counter
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from tools.prepare_full_task_supervision import verify as verify_witnesses
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
from weld_pipeline.server_full_reference_witnesses import ReferenceWitnessBuilder
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from weld_pipeline.world_set_evaluation_285 import distance_to_primitive, reference_segment

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/diagnose_training_arc_witnesses.py',
           'weld_pipeline/server_full_reference_witnesses.py',
           'weld_pipeline/server_circle_capacity_prefilter.py',
           'weld_pipeline/sjg_curve_queries_907.py',
           'weld_pipeline/contract_curve_geometry.py',
           'weld_pipeline/contract_geometry_d3_409.py',
           'weld_pipeline/legacy_incremental_event_runtime_393.py',
           'weld_pipeline/server_partitioned_full_dataset.py']


def interval_diagnostic(builder, pid, child, reference, full_closed):
    try:
        lo, hi = builder.projected_interval(pid, reference, full_closed)
    except ValueError as error:
        return dict(status='PROJECTED_EXTENT_REJECTED', error=str(error))
    grid_limit = False
    if hi <= lo or not np.any(np.diff(builder.query_ranges(pid, lo, hi), axis=1) > 0):
        lo, hi = builder.input_grid_point_limit(pid, lo, hi)
        grid_limit = True
    if not hi > lo:
        return dict(status='NO_POSITIVE_INPUT_GRID_EXTENT')
    segment = builder.segment(pid, lo, hi)
    segment['weld_type'] = child['weld_type']
    variants = []
    accepted = False
    for reversed_direction in (False, True):
        candidate = dict(segment, **reverse_segment(segment)) if reversed_direction else segment
        score = child_error(candidate, child, reference, full_closed, False)
        accepted |= score is not None
        a, b = [np.asarray(value['points'], float) for value in (candidate, reference)]
        pa, ta = [CircularArc.from_parameters(value['arc_parameters']) for value in (candidate, reference)]
        pt = np.sign(pa.sweep_rad) * np.cross(pa.plane_normal, pa.start_direction)
        tt = np.sign(ta.sweep_rad) * np.cross(ta.plane_normal, ta.start_direction)
        variants.append(dict(reversed_direction=reversed_direction,
            original_D3_accepted=score is not None,
            ordered_endpoint_maximum_mm=float(np.linalg.norm(a[[0, -1]] - b[[0, -1]], axis=1).max()),
            start_tangent_dot=float(pt @ tt),
            bidirectional_sampled_maximum_mm=float(max(distance_to_primitive(a, reference).max(),
                                                        distance_to_primitive(b, candidate).max()))))
    has_query = bool(np.any(builder.query_ranges(pid, lo, hi)[:, 1] > builder.query_ranges(pid, lo, hi)[:, 0]))
    return dict(status='ACCEPTED_WITNESS' if accepted and has_query else
                ('NO_FIXED_INPUT_QUERY' if accepted else 'ORIGINAL_D3_REJECTED'),
                interval_mm=[lo, hi], used_input_grid_extent=grid_limit,
                input_circle_closed=builder.curves[pid].arc.is_closed, variants=variants)


def main(args):
    for name in [args.dataset_name, args.coverage_audit_name, args.experiment_name]:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
            raise ValueError('Local dataset and experiment names are required')
    base = ROOT / 'server_rebuild' / args.dataset_name
    dataset = PartitionedFullShardDataset(ROOT, base)
    audit = base / args.coverage_audit_name
    audit_receipt = read(audit / 'receipt.json')
    if audit_receipt['training_registry_sha256'] != dataset.registry_sha256:
        raise ValueError('TRAIN coverage audit registration changed')
    for name, sha in audit_receipt['files'].items():
        if digest(audit / name) != sha:
            raise ValueError('Original TRAIN coverage audit changed')
    weak = [row for row in read(audit / 'weak_carrier_children.json') if row['kind'] == 'arc']
    sources = {name: digest(ROOT / name) for name in SOURCES}
    out = base / args.experiment_name
    out.mkdir(exist_ok=False)
    rows, profiles_bound = [], {}
    for key in sorted({row['version_key'] for row in weak}):
        job_key, version = dataset.versions[key]
        view = dataset.supervision(key)
        witness_folder = base / 'full_task_supervision_v1' / job_key / key
        verify_witnesses(witness_folder, dataset, version)
        shard_folder = dataset.shards / job_key
        receipt = read(shard_folder / 'receipt.json')
        path = shard_folder / 'profiles.json'
        if (receipt['registry_sha256'] != dataset.registry_sha256
                or receipt['workpiece_sha256'] != dataset.jobs[job_key]['workpiece_sha256']
                or receipt['GT_geometry_reads'] != 0 or digest(path) != receipt['files']['profiles.json']):
            raise ValueError('Frozen input-only TRAIN profiles changed')
        profiles = read(path)
        ids = [i for i, profile in enumerate(profiles) if profile['representation']['kind'] == 'arc']
        builder = ReferenceWitnessBuilder(SimpleNamespace(profiles=[profiles[i] for i in ids]))
        profiles_bound[job_key] = dict(profiles_sha256=digest(path), arc_profiles=len(ids))
        for prior in [row for row in weak if row['version_key'] == key]:
            parent = view.parents[prior['parent']]
            child = parent['children'][prior['child']]
            if child['original_child_id'] != prior['child_id']:
                raise ValueError('Weak arc child owner changed')
            reference = reference_segment(dict(child, world_representation=child['representation']))
            points = np.asarray(reference['points'], float)
            candidates = builder.possible_profiles(child, reference)
            if len(candidates) != prior['same_kind_prefilter_profiles']:
                raise ValueError('Original arc prefilter candidates changed')
            finite, circles = [], []
            for pid, curve in enumerate(builder.curves):
                finite.append(float(curve.project(points)[1].max()))
                delta = points - curve.arc.center_mm
                height = delta @ curve.arc.plane_normal
                planar = delta - height[:, None] * curve.arc.plane_normal
                circles.append(float(np.hypot(height, np.linalg.norm(planar, axis=1) - curve.arc.radius_mm).max()))
            full_closed = parent['is_closed'] and len(parent['children']) == 1
            results = [dict(profile_id=ids[int(pid)], **interval_diagnostic(
                builder, int(pid), child, reference, full_closed)) for pid in candidates]
            if any(result['status'] == 'ACCEPTED_WITNESS' for result in results):
                raise ValueError('An allegedly missing original witness is now accepted')
            truth_arc = CircularArc.from_parameters(reference['arc_parameters'])
            rows.append(dict(**prior, full_closed_reference=full_closed,
                reference_curve_closed=truth_arc.is_closed, reference_sweep_rad=truth_arc.sweep_rad,
                input_arc_profiles=len(ids), best_finite_input_arc_maximum_mm=min(finite) if finite else None,
                best_underlying_input_circle_maximum_mm=min(circles) if circles else None,
                closest_finite_profile_id=ids[int(np.argmin(finite))] if finite else None,
                original_interval_rejections=results))
        print('TRAIN_ARC_GAPS_DIAGNOSED', key, len(rows), '/', len(weak), flush=True)
    if len(rows) != len(weak) or sources != {name: digest(ROOT / name) for name in sources}:
        raise ValueError('Complete weak arc audit or source binding changed')
    write(out / 'arc_gaps.json', rows)
    write(out / 'receipt.json', dict(status='ALL_MISSING_TRAIN_ARC_WITNESSES_REPRODUCED_NOT_INFERENCE',
        training_registry_sha256=dataset.registry_sha256, prior_audit_receipt_sha256=digest(audit / 'receipt.json'),
        sources=sources, input_profiles=profiles_bound, missing_arc_labels=len(rows),
        no_prefilter_candidate_labels=sum(not r['original_interval_rejections'] for r in rows),
        interval_rejection_counts=dict(Counter(result['status'] for row in rows
                                              for result in row['original_interval_rejections'])),
        model_fit_calls=0, model_accuracy_evaluations=0, held_reference_reads=0,
        input_profiles_or_reference_coordinates_modified=False,
        reference_projections_used_only_for_offline_TRAIN_diagnostics=True,
        underlying_circle_proximity_is_not_finite_contact=True,
        files={'arc_gaps.json': digest(out / 'arc_gaps.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--coverage-audit-name', required=True)
    parser.add_argument('--experiment-name', required=True)
    main(parser.parse_args())
