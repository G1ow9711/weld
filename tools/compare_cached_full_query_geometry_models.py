"""Assess the next complete query model against verified frozen TRAIN baselines.

The two previous complete registered-view diagnostic arms are reused only after every
record, model and input-assignment binding is verified. Only the new model runs
again. These reference-assigned child checks are not full deployment accuracy.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import time
import traceback

import torch

from tools.compare_full_query_geometry_models import (
    ROOT, SOURCES as ORIGINAL_SOURCES, assess_arm, process)
from tools.diagnose_full_query_geometry import atomic
from tools.export_full_dataset_split_view import local_name
from weld_pipeline.server_local_query_checkpoint import verify_complete_local_query_fit
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

SOURCES = list(dict.fromkeys(['tools/compare_cached_full_query_geometry_models.py', *ORIGINAL_SOURCES]))


def cached_baselines(base, dataset, name):
    folder = base / local_name(name)
    summary, protocol = read(folder / 'receipt.json'), read(folder / 'protocol.json')
    if (summary['status'] != 'ALL_TRAIN_QUERY_MODELS_COMPARED_AT_IDENTICAL_DENSE_INPUT_QUERIES'
            or not summary['query_assignment_fingerprints_identical']
            or protocol['training_registry_sha256'] != dataset.registry_sha256
            or protocol['sources'] != {name:digest(ROOT / name) for name in protocol['sources']}):
        raise ValueError('Frozen complete TRAIN diagnostic baseline changed')
    arms = {}
    for arm in ('original', 'new_local'):
        result = read(folder / arm / 'receipt.json')
        if (result['totals']['versions'] != len(dataset.versions)
                or result['totals']['children'] != sum(v['children'] for _,v in dataset.versions.values())
                or result['held_reference_reads'] != 0
                or set(result['assignment_fingerprints']) != set(dataset.versions)
                or {r['version_key'] for r in result['records']} != set(dataset.versions)
                or len(result['records']) != len(dataset.versions)):
            raise ValueError('A full cached diagnostic view or denominator is missing')
        for record in result['records']:
            path = (ROOT / record['path']).resolve()
            path.relative_to(folder.resolve())
            if digest(path) != record['sha256']:
                raise ValueError('An immutable previous diagnostic prediction record changed')
        arms[arm] = result
    if arms['original']['assignment_fingerprints'] != arms['new_local']['assignment_fingerprints']:
        raise ValueError('Previous complete models used different actual input queries')
    for arm, model_name in [('original', 'full_weld_reference_recall_v3'),
                            ('new_local', 'full_local_query_geometry_grid_v4_round01')]:
        model = base / model_name / 'last.pt'
        if digest(model) != arms[arm]['model_sha256']:
            raise ValueError('The model represented by the frozen baseline changed')
    return folder, protocol, arms


def main(args):
    torch.set_num_threads(2)
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    dataset = PartitionedFullShardDataset(ROOT, base)
    baseline, protocol, arms = cached_baselines(base, dataset, args.baseline_comparison_name)
    new = base / local_name(args.new_training_name)
    identity = process(args.new_training_pid)
    if identity['state'] in ('Z', 'X') or new.name not in identity['argv']:
        raise ValueError('The actual declared complete candidate training handle is absent or different')
    out = base / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    sources = {name:digest(ROOT / name) for name in SOURCES}
    write(out / 'protocol.json', dict(sources=sources, training_registry_sha256=dataset.registry_sha256,
        cached_baseline_comparison=str(baseline.relative_to(ROOT)),
        baseline_receipt_sha256=digest(baseline / 'receipt.json'),
        assignment_seed=protocol['assignment_seed'], new_training_identity=identity,
        every_cached_view_record_verified=True, all_registered_views_required=True,
        held_reference_reads=0, optimizer_updates=0, full_output_accuracy_not_claimed=True))
    try:
        while not (new / 'receipt.json').exists():
            if (new / 'failure.json').exists():
                raise RuntimeError('The complete alternating candidate training actually failed')
            current = process(args.new_training_pid)
            if current['state'] in ('Z', 'X') or current['start_ticks'] != identity['start_ticks']:
                raise RuntimeError('The same actual candidate handle stopped before its complete receipt')
            if sources != {name:digest(ROOT / name) for name in sources}:
                raise ValueError('The bound complete diagnostic source changed')
            atomic(out / 'progress.json', dict(status='VERIFIED_FULL_BASELINES_WAITING_LIVE_ALTERNATING_TRAINING',
                controller_pid=os.getpid(), new_training_pid=args.new_training_pid,
                cached_baseline_views=len(dataset.versions), held_reference_reads=0, unix_time=time.time()))
            time.sleep(10)
        checkpoint = new / 'last.pt'; sha = digest(checkpoint); receipt = read(new / 'receipt.json')
        if sha != receipt['checkpoint_sha256']:
            raise ValueError('The completed model differs from its immutable fitting receipt')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        fit = verify_complete_local_query_fit(ROOT, base, state, sha, new / 'receipt.json')
        after = assess_arm(dataset, base, out, state, True, protocol['assignment_seed'], sha)
        if after['assignment_fingerprints'] != arms['new_local']['assignment_fingerprints']:
            raise ValueError('The new candidate used different actual input queries')
        cached_baselines(base, dataset, args.baseline_comparison_name)
        if sources != {name:digest(ROOT / name) for name in sources}:
            raise ValueError('The bound diagnostic source changed after inference')
        atomic(out / 'receipt.json', dict(status='ALL_TRAIN_NEW_QUERY_COMPARED_WITH_VERIFIED_CACHED_MODELS',
            recorded_utc=datetime.now(timezone.utc).isoformat(), original=arms['original']['totals'],
            previous_local=arms['new_local']['totals'], new=after['totals'],
            original_categories=arms['original']['categories'], previous_local_categories=arms['new_local']['categories'],
            new_categories=after['categories'], new_complete_query_fit=fit,
            original_model_sha256=arms['original']['model_sha256'],
            previous_local_model_sha256=arms['new_local']['model_sha256'], new_model_sha256=sha,
            query_assignment_fingerprints_identical=True, old_predictions_reused_after_full_verification=True,
            held_reference_reads=0, optimizer_updates=0, completed_optimization_rounds=0,
            complete_parent_and_deployment_output_quality_not_evaluated=True, full_contract_pass=False))
    except BaseException:
        atomic(out / 'failure.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--baseline-comparison-name', default='full_query_dense_v3_v4_geometry_comparison_20261012_v1')
    parser.add_argument('--new-training-name', required=True)
    parser.add_argument('--new-training-pid', type=int, required=True)
    parser.add_argument('--experiment-name', required=True)
    main(parser.parse_args())
