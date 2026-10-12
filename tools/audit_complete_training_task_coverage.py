"""Audit all frozen TRAIN assignments and label masks without fitting a model.

Carrier witness gaps are reported separately from full geometry supervision:
the deployed query model also learns free geometry for every reference child.
This audit reads TRAIN references only and is not a prediction accuracy score.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import torch

from tools.prepare_full_task_supervision import verify as verify_witnesses
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_partitioned_query_fit import verifier
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('front_wall', 'back_wall', 'start_height', 'end_height',
          'start_hole_height', 'end_hole_height', 'start_base_edge', 'end_base_edge')
SOURCES = ['tools/audit_complete_training_task_coverage.py',
           'tools/prepare_full_task_supervision.py',
           'weld_pipeline/server_partitioned_full_dataset.py',
           'weld_pipeline/server_partitioned_query_fit.py',
           'weld_pipeline/server_query_checkpoint_model.py',
           'weld_pipeline/server_full_shard_dataset.py',
           'weld_pipeline/server_full_training_samples.py',
           'weld_pipeline/server_full_reference_witnesses.py',
           'weld_pipeline/server_runtime_aligned_training_samples.py']


def main(args):
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value)
           for value in (args.dataset_name, args.experiment_name)):
        raise ValueError('Local dataset and experiment names are required')
    base = ROOT / 'server_rebuild' / args.dataset_name
    dataset = PartitionedFullShardDataset(ROOT, base)
    checkpoint = Path(args.query_checkpoint).resolve()
    checkpoint.relative_to(base)
    receipt_path = checkpoint.parent / 'receipt.json'
    checkpoint_sha = digest(checkpoint)
    if checkpoint_sha != args.query_sha256:
        raise ValueError('The frozen query checkpoint changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    fit = verifier(ROOT, base)(ROOT, state, checkpoint_sha, receipt_path)
    out = base / args.experiment_name
    out.mkdir(exist_ok=False)
    sources = {name: digest(ROOT / name) for name in SOURCES}
    rows, missing = [], []
    totals, kinds, weak_kinds = Counter(), Counter(), Counter()
    field_totals = {name: Counter() for name in FIELDS}
    for job_key, job in dataset.jobs.items():
        for version in job['versions']:
            key = version['key']
            view = dataset.supervision(key)
            folder = base / 'full_task_supervision_v1' / job_key / key
            witness_receipt = verify_witnesses(folder, dataset, version)
            coverage = read(folder / 'coverage.json')
            children = [(pi, ci, parent, child) for pi, parent in enumerate(view.parents)
                        for ci, child in enumerate(parent['children'])]
            with np.load(folder / 'owner_witnesses.npz', allow_pickle=False) as arrays:
                owner_rows = np.unique(arrays['attribute_row'])
                pairs = set(zip(arrays['parent'].tolist(), arrays['child'].tolist()))
                expected_pairs = {(children[int(i)][0], children[int(i)][1]) for i in owner_rows}
                if pairs != expected_pairs:
                    raise ValueError('Witness owner rows disagree with ordered reference children')
            if (len(children) != coverage['original_children']
                    or len(owner_rows) != coverage['represented_children']
                    or len(children) != witness_receipt['original_children']):
                raise ValueError('Complete reference or carrier denominator changed')
            weak_pairs = {(int(m['parent']), int(m['child'])) for m in coverage['missing_children']}
            if pairs & weak_pairs or pairs | weak_pairs != {(p, c) for p, c, _, _ in children}:
                raise ValueError('Carrier witness audit lost or duplicated a child')
            epoch_evidence = []
            for epoch in range(read(receipt_path)['epochs']):
                evidence = state['view_evidence'][f'{epoch}:{key}']
                if (evidence['job_key'] != job_key
                        or evidence['label_receipt_sha256'] != version['label_receipt_sha256']
                        or evidence['owner_witness_receipt_sha256'] != digest(folder / 'receipt.json')):
                    raise ValueError('Actual trained view differs from the audited label/witness version')
                for report in evidence['reports']:
                    if (report['children'] != len(children)
                            or report['strong_carrier_assignments'] != len(owner_rows)
                            or report['weak_free_geometry_assignments'] != len(weak_pairs)
                            or not report['complete_reference_denominator_retained']
                            or not report['free_geometry_targets_retained_for_all_children']):
                        raise ValueError('An actual optimizer update dropped complete geometry supervision')
                epoch_evidence.append(dict(epoch=epoch, actual_update_reports=len(evidence['reports']),
                    complete_child_assignments_per_update=len(children),
                    free_geometry_for_every_child=True))
            for pi, ci, parent, child in children:
                kind = child['original_kind']
                kinds[kind] += 1
                totals[child['weld_type']] += 1
                if (pi, ci) in weak_pairs:
                    weak_kinds[kind] += 1
                    reason = next(m for m in coverage['missing_children']
                                  if m['parent'] == pi and m['child'] == ci)
                    missing.append(dict(job_key=job_key, version_key=key,
                        parent_id=parent['original_parent_id'], child_id=child['original_child_id'],
                        weld_type=child['weld_type'], **reason))
            for column, name in enumerate(FIELDS):
                codes = view.attribute_arrays['state'][:, column]
                required = view.attribute_arrays['required_mask'][:, column]
                field_totals[name].update(dict(total_children=len(children),
                    known=int((codes >= 0).sum()), unknown=int((codes < 0).sum()),
                    exact_mm=int((codes == 0).sum()), required=int(required.sum()),
                    required_unknown=int((required & (codes < 0)).sum())))
            totals.update(dict(versions=1, parents=len(view.parents), children=len(children),
                carrier_witness_children=len(owner_rows), weak_free_geometry_children=len(weak_pairs),
                algebraically_known_endpoint_frames=int(view.frame_algebraically_known.sum())))
            rows.append(dict(job_key=job_key, version_key=key, family_id=view.family_id,
                label_receipt_sha256=version['label_receipt_sha256'],
                witness_receipt_sha256=digest(folder / 'receipt.json'),
                coverage_sha256=digest(folder / 'coverage.json'), original_children=len(children),
                strong_carrier_children=len(owner_rows), weak_free_geometry_children=len(weak_pairs),
                actual_training_epochs=epoch_evidence))
        print('TRAIN_TASK_COVERAGE_AUDITED', len(rows), '/', len(dataset.versions), flush=True)
    if totals['versions'] != len(dataset.versions):
        raise ValueError('A registered TRAIN version was not audited')
    if sources != {name: digest(ROOT / name) for name in sources}:
        raise ValueError('Audit sources changed during execution')
    write(out / 'training_versions.json', rows)
    write(out / 'weak_carrier_children.json', missing)
    write(out / 'receipt.json', dict(status='ALL_TRAIN_ASSIGNMENTS_AUDITED_NOT_MODEL_ACCURACY',
        recorded_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
        training_registry_sha256=dataset.registry_sha256, query_checkpoint_sha256=checkpoint_sha,
        full_query_fit=fit, training_jobs=len(dataset.jobs), totals=dict(totals), kinds=dict(kinds),
        weak_carrier_kinds=dict(weak_kinds), endpoint_field_label_availability=field_totals,
        all_reference_children_retained_in_actual_query_updates=True,
        carrier_witness_coverage_is_not_a_recall_ceiling=True,
        business_frame_semantics_certified=False, model_accuracy_evaluated=False,
        fit_calls=0, held_reference_reads=0,
        files={p.name: digest(p) for p in out.iterdir() if p.is_file()}))
    print('ALL_TRAIN_TASK_COVERAGE_AUDITED', dict(totals), dict(weak_kinds), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--query-checkpoint', required=True)
    parser.add_argument('--query-sha256', required=True)
    main(parser.parse_args())
