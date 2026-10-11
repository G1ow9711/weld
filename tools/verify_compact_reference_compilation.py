"""Verify bounded decoding preserves every original D3 cost and global score."""
import argparse
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_predicted_parent_pool import decoded_subset
from weld_pipeline.server_offline_parent_set_reference import compile_compact_reference_table, evaluate_reference_subset
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from tools.train_full_predicted_parent_selector import load_job

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'server_rebuild/full_dataset_v1'


def main(job, pool_name, output_name):
    start = time.perf_counter()
    output = BASE/output_name
    if output.exists():
        raise ValueError('Choose a new immutable complete-cost control record')
    dataset = FullShardDataset(ROOT)
    checked = []
    with threadpool_limits(limits=2):
        row = load_job(dataset, job, BASE/pool_name)
        for view in row['views']:
            table, score = compile_compact_reference_table(row['pool']['parents'], view['target'], decoded_subset, 256)
            for name in ('costs', 'tp', 'child_counts', 'invalid_children'):
                np.testing.assert_array_equal(table[name], view['table'][name])
            assert score['categories']['overall'] == view['pool_capacity_reference_evaluation']
            rng = np.random.default_rng(1087)
            for size in (0, 1, min(128, len(table['child_counts']))):
                ids = rng.choice(len(table['child_counts']), size, replace=False)
                assert evaluate_reference_subset(table, ids) == evaluate_reference_subset(view['table'], ids)
            empty, empty_score = compile_compact_reference_table([], view['target'], decoded_subset, 256)
            assert empty['costs'].shape == (0, len(view['target']['parents']))
            assert empty_score['categories']['overall'] == dict(G=view['G'], predicted=0, TP=0, FP=0)
            checked.append(dict(version_key=view['key'], cost_rows=table['costs'].shape[0],
                                cost_columns=table['costs'].shape[1], full_capacity=score['categories']['overall']))
    result = dict(status='PASS_EVERY_COMPACT_D3_COST_AND_GLOBAL_SCORE_CONTROL_NOT_MODEL_EVALUATION',
        job_key=job, pool_name=pool_name, checked_training_views=checked,
        all_costs_and_TP_counts_bitwise_equal=True, empty_and_subset_controls_passed=True,
        global_matching_preserved=True, reference_children_removed=0, optimizer_updates=0,
        source_sha256=digest(__file__), helper_sha256=digest(ROOT/'weld_pipeline/server_offline_parent_set_reference.py'),
        selector_sha256=digest(ROOT/'tools/train_full_predicted_parent_selector.py'),
        seconds=time.perf_counter()-start)
    write(output, result)
    print('ALL_COMPACT_REFERENCE_COSTS_PARITY_PASS', checked, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--pool-name', required=True)
    parser.add_argument('--output-name', required=True)
    args = parser.parse_args()
    main(args.job, args.pool_name, args.output_name)
