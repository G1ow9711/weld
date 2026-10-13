"""Audit actual complete TRAIN query assignments against whole references.

No assignments or candidates are regenerated or modified. Exact differences
are reported as differences, not automatically as incompatible annotations.
"""
import argparse
from collections import defaultdict, Counter
from pathlib import Path
import traceback

import numpy as np

from tools.export_full_dataset_split_view import local_name
from tools.verify_original_best_frozen_replay import canonical
from tools.record_training_experiment import git_head
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_runtime_904 import ROOT, read, write, digest


BASE = ROOT / 'server_rebuild/full_dataset_v2'
DIAGNOSTIC = BASE / 'full_query_dense_endpoint_v5_geometry_comparison_20261012_v1/new_local'
SOURCES = ['tools/audit_complete_query_reference_collisions.py',
    'weld_pipeline/server_partitioned_full_dataset.py', 'weld_pipeline/server_full_shard_dataset.py',
    'weld_pipeline/sjg_curve_queries_907.py', 'weld_pipeline/contract_curve_geometry.py']


def geometry(child):
    return dict(kind=child['original_kind'], weld_type=child['weld_type'],
                representation=child['representation'])


def main(args):
    out = BASE / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    dataset = PartitionedFullShardDataset(ROOT, BASE)
    prior = read(DIAGNOSTIC / 'receipt.json')
    rows = prior['records']
    if (len(rows) != len(dataset.versions) or {r['version_key'] for r in rows} != set(dataset.versions)
            or prior['held_reference_reads'] != 0):
        raise ValueError('Every registered TRAIN view requires an authenticated actual assignment')
    sources = {name: digest(ROOT / name) for name in SOURCES}
    write(out / 'protocol.json', dict(source_commit=git_head(ROOT), sources=sources,
        training_registry_sha256=dataset.registry_sha256,
        assignment_receipt_sha256=digest(DIAGNOSTIC / 'receipt.json'),
        all_registered_TRAIN_views=len(rows), input_identity='Exact raw SHA, profile index and float64 query-position bytes',
        scope='Exact geometric/ordered-parent target multiplicity; not annotation infeasibility or model accuracy',
        optimizer_updates=0, held_or_test_reference_reads=0))
    try:
        groups = defaultdict(list); bindings = []; total = 0
        for index, row in enumerate(rows):
            key = row['version_key']; job_key, version = dataset.versions[key]
            path = (ROOT / row['path']).resolve(); path.relative_to(DIAGNOSTIC)
            if digest(path) != row['sha256']:
                raise ValueError('An existing complete actual assignment record changed')
            measured = read(path); view = dataset.supervision(key)
            by_id = {}
            for parent in view.parents:
                parent_geometry = dict(is_closed=parent['is_closed'], gaps=parent['gaps'],
                                       children=[geometry(c) for c in parent['children']])
                parent_sha = canonical(parent_geometry)
                for child_index, child in enumerate(parent['children']):
                    cid = child['original_child_id']
                    if cid in by_id:
                        raise ValueError('Repeated original child identity in one complete view')
                    by_id[cid] = (child, parent_sha, child_index, len(parent['children']))
            if (len(measured['children']) != version['children'] or len(by_id) != version['children']
                    or {x['original_child_id'] for x in measured['children']} != set(by_id)):
                raise ValueError('An original child disappeared from the complete assignment audit')
            for assigned in measured['children']:
                child, parent_sha, child_index, child_count = by_id[assigned['original_child_id']]
                position = float(assigned['query_position_mm'])
                identity = (job_key, int(assigned['profile_id']), np.float64(position).tobytes().hex())
                curve = CurveCarrier(child['representation'])
                groups[identity].append(dict(version=key, original_child_id=child['original_child_id'],
                    category=assigned['category'], geometry_sha256=canonical(geometry(child)),
                    parent_geometry_sha256=parent_sha, child_index=child_index, parent_child_count=child_count,
                    ordered_endpoints_mm=curve.at(np.array([0., curve.length])).tolist(),
                    family_equal_version_weight=version['family_equal_version_weight']))
            total += len(measured['children'])
            bindings.append(dict(version=key, assignment_sha256=row['sha256'],
                                 label_receipt_sha256=version['label_receipt_sha256']))
            write(out / 'progress.json', dict(complete_views=index + 1, total_views=len(rows), children=total))
        expected = sum(v['children'] for _, v in dataset.versions.values())
        if total != expected:
            raise ValueError('The full reference child denominator changed')
        collisions = []; counts = Counter()
        for (job, pid, position), values in sorted(groups.items()):
            if len(values) < 2:
                continue
            per_view = Counter(v['version'] for v in values)
            distinct_geometry = len({v['geometry_sha256'] for v in values})
            distinct_parents = len({v['parent_geometry_sha256'] for v in values})
            same_view_multiple = any(n > 1 for n in per_view.values())
            distinct_versions = len(per_view)
            counts['queries_with_multiple_reference_rows'] += 1
            counts['same_view_multiple_children_queries'] += int(same_view_multiple)
            counts['cross_version_queries'] += int(distinct_versions > 1)
            counts['queries_with_distinct_exact_geometry'] += int(distinct_geometry > 1)
            counts['queries_with_distinct_parent_geometry'] += int(distinct_parents > 1)
            distances = []
            for i, left in enumerate(values):
                for right in values[i + 1:]:
                    a, b = np.asarray(left['ordered_endpoints_mm']), np.asarray(right['ordered_endpoints_mm'])
                    direct = float(np.linalg.norm(a - b, axis=1).max())
                    reverse = float(np.linalg.norm(a - b[::-1], axis=1).max())
                    distances.append(dict(left=i, right=values.index(right),
                        ordered_endpoint_separation_mm=direct, direction_free_endpoint_separation_mm=min(direct, reverse)))
            collisions.append(dict(job_key=job, profile_id=pid, query_position_float64_hex=position,
                rows=values, distinct_exact_geometry=distinct_geometry, distinct_parent_geometry=distinct_parents,
                same_view_multiple_children=same_view_multiple, distinct_versions=distinct_versions,
                endpoint_comparisons=distances))
        write(out / 'collisions.json', collisions)
        if sources != {name: digest(ROOT / name) for name in sources}:
            raise ValueError('Audit source changed')
        write(out / 'receipt.json', dict(status='ALL_REGISTERED_TRAIN_QUERY_REFERENCE_MULTIPLICITY_AUDITED',
            protocol_sha256=digest(out / 'protocol.json'), views=len(rows), children=total,
            unique_actual_input_queries=len(groups), counters=dict(counts), records=bindings,
            collisions_sha256=digest(out / 'collisions.json'),
            assignments_or_labels_or_candidates_modified=False, fitting_updates=0,
            exact_differences_not_automatically_incompatible=True,
            query_identity_not_used_as_model_feature=True, held_or_test_reference_reads=0,
            complete_optimization_rounds_added=0, full_contract_pass=False))
        print('COMPLETE_QUERY_REFERENCE_MULTIPLICITY', total, len(groups), dict(counts), flush=True)
    except BaseException as exc:
        write(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    main(parser.parse_args())
