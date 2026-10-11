"""Compare actual native input and prediction against the retained old input.

All original interval instances and connection features are compared. This
does not execute the final parent selector or claim a run446 output score.
"""
import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_native_shard_input import load_native_input, load_frozen_native_models
from weld_pipeline.sjg_interval_forest_runtime_1129 import predict_instances
from weld_pipeline.sjg_relative_forest_runtime_1139 import relative_link_blocks, source_frame
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
import weld_pipeline.sjg_relative_ordinal_runtime_1150 as original_runtime

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def main(job_key, output_name):
    start = time.perf_counter()
    dataset = FullShardDataset(ROOT)
    job = dataset.jobs[job_key]
    provider = job['prior_input_provider']
    if not provider or not provider['observed_features_complete']:
        raise ValueError('This parity control requires a retained original complete input')
    source = (ROOT / provider['runtime_directory']).resolve()
    source.relative_to(ROOT)
    key = job['runtime_key']
    observed = source / ('run344' if key in ('25', '36', '42') else 'run329') / ('source_' + key)
    # Rebase the original reader in this isolated verifier process only. Its
    # original input authentication and arithmetic remain unchanged.
    prior_out, prior_feature_path = original_runtime.OUT, original_runtime.feature_path
    original_runtime.OUT = source
    original_runtime.feature_path = lambda k: observed if k == key else (_ for _ in ()).throw(ValueError('Wrong source key'))
    try:
        original = original_runtime.load_case(key)
    finally:
        original_runtime.OUT, original_runtime.feature_path = prior_out, prior_feature_path
    case = load_native_input(dataset, job_key, BASE / 'legacy_surface_context_train_v1')
    for name in ('x', 'canonical'):
        if not np.array_equal(case[name], original[name]):
            raise AssertionError('Complete native input array changed')
    for name in ('offsets', 'spans'):
        if not np.array_equal(getattr(case['graph'], name), getattr(original['graph'], name)):
            raise AssertionError('Original finite graph partition changed')
    if case['graph'].profiles != original['graph'].profiles or case['surfaces'] != original['surfaces']:
        raise AssertionError('Native finite profiles or source face normals changed')
    models, checkpoint = load_frozen_native_models(ROOT)
    labels = {str((ROOT / v['label_directory']).resolve()) for j in dataset.jobs.values() for v in j['versions']}
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, Path)):
            name = str(Path(args[0]).resolve())
            if any(name == p or name.startswith(p + '/') for p in labels):
                raise AssertionError('Native parity control cannot open a qualified reference')
    sys.addaudithook(audit)
    with threadpool_limits(limits=2):
        predicted = predict_instances(case, models)
        expected = predict_instances(original, models)
        if predicted != expected:
            raise AssertionError('Every original interval-instance prediction must be preserved')
        frames = 0
        for node in predicted['nodes']:
            a, av = source_frame(node, case)
            b, bv = source_frame(node, original)
            if av != bv or not np.array_equal(a, b):
                raise AssertionError('Original native source frame changed')
            frames += 1
        rows, blocks = 0, 0
        feature_sha = hashlib.sha256()
        for actual, old in zip_longest(relative_link_blocks(predicted['nodes'], case),
                                      relative_link_blocks(expected['nodes'], original)):
            if actual is None or old is None or any(not np.array_equal(a, b) for a, b in zip(actual, old)):
                raise AssertionError('Original complete conditional connection features changed')
            rows += len(actual[0]); blocks += 1
            feature_sha.update(actual[1].tobytes())
    result = dict(status='PASS_ACTUAL_NATIVE_INPUT_AND_INTERVAL_RUNTIME_PARITY_NOT_FINAL_RUN446_REPLAY',
        job_key=job_key, runtime_key=key, feature_rows=len(case['x']), feature_columns=152,
        finite_profiles=len(case['graph'].profiles), canonical_fragments=len(case['canonical']),
        original_interval_nodes=len(predicted['nodes']), invalid_interval_nodes=len(predicted['invalid']),
        all_raw_features_and_graph_arrays_bitwise_equal=True, original_surface_bytes_authenticated=True,
        all_original_interval_instance_predictions_equal=True, all_source_frames_bitwise_equal=True,
        frames_compared=frames, connection_feature_blocks=blocks, connection_feature_rows=rows,
        complete_connection_feature_rows_bitwise_equal=True, connection_features_sha256=feature_sha.hexdigest(),
        checkpoint=checkpoint, context_receipt_sha256=case['preserved_context_receipt_sha256'],
        source_feature_receipt_sha256=original['feature_receipt_sha256'],
        full_shard_feature_provenance_sha256=case['feature_receipt_sha256'],
        reference_geometry_reads=0, optimizer_updates=0, final_parent_selection_executed=False,
        original_526_column_validity_head_executed=False, extraction_accuracy_evaluations=0,
        full_contract_pass=False, sources={name:digest(ROOT / name) for name in
            ['weld_pipeline/server_native_shard_input.py', 'tools/verify_native_shard_input.py']},
        seconds=time.perf_counter() - start)
    destination = BASE / output_name
    if destination.exists():
        raise ValueError('Native parity evidence is immutable; use a new output name')
    write(destination, result)
    print('ACTUAL_NATIVE_SHARD_INPUT_PARITY_PASS', job_key, len(predicted['nodes']), rows, flush=True)
    original['x']._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-key', default='cloud_c39d9a8e68d4afe4')
    parser.add_argument('--output-name', default='full_native_shard_input_validation_v1.json')
    args = parser.parse_args()
    main(args.job_key, args.output_name)
