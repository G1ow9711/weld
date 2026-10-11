"""Check the full consumer against real source features and all TRAIN labels."""
from collections import Counter
import gc
from pathlib import Path
import time

import numpy as np

from weld_pipeline.server_full_shard_dataset import FullShardDataset, IncompleteTrainingInputs
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.sjg_local_runtime_904 import digest, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def main():
    started = time.perf_counter()
    destination = BASE / 'full_shard_dataset_validation.json'
    if destination.exists():
        raise ValueError('Preserve this consumer validation; use a new artifact for a changed consumer')
    dataset = FullShardDataset(ROOT)
    plan = dataset.epoch_plan(0, require_ready=False)
    assert len(plan) == len(dataset.versions) == 170
    assert len({row['version_key'] for row in plan}) == 170
    assert plan == dataset.epoch_plan(0, require_ready=False)
    assert plan != dataset.epoch_plan(1, require_ready=False)
    mass = Counter()
    counts = Counter()
    for row in plan:
        view = dataset.supervision(row['version_key'])
        mass[view.family_id] += row['loss_weight']
        for parent in view.parents:
            assert parent['original_child_count'] == len(parent['children'])
            counts['parents'] += 1
            for child in parent['children']:
                counts[child['original_kind']] += 1
        assert np.array_equal(view.label_known, view.attribute_arrays['state'] >= 0)
        assert np.array_equal(view.numeric_known, view.attribute_arrays['state'] == 0)
        counts['known_attribute_states'] += int(view.label_known.sum())
        counts['known_export_frames'] += int(view.frame_algebraically_known.sum())
        assert not view.frame_axis_signs_reinterpreted and not view.physical_background_labels
    assert all(abs(value - 1 / len(dataset.families)) < 1e-12 for value in mass.values())
    blocked = False
    try:
        dataset.require_complete_inputs()
    except IncompleteTrainingInputs:
        blocked = True
    held_keys = [row['key'] for row in __import__('json').loads((BASE / 'model_use_policy.json').read_text())['examples']
                 if not row['fit_allowed']]
    assert set(held_keys).isdisjoint(dataset.versions)
    ready = dataset.ready_jobs()
    rows = []
    rng = np.random.default_rng(1031)
    for job_key in ready:
        inp = dataset.input_case(job_key)
        job = dataset.jobs[job_key]
        source = job['prior_input_provider']
        parity = False
        if source and source['observed_features_complete']:
            old = FixedQueries(ROOT / source['runtime_directory'], job['runtime_key'])
            try:
                assert old.n == inp.n
                ids = rng.choice(inp.n, min(5000, inp.n), replace=False)
                assert np.array_equal(old.features(ids), inp.features(ids))
                for pid in set([0, len(inp.profiles) // 2, len(inp.profiles) - 1]):
                    if inp.offsets[pid] == inp.offsets[pid + 1]:
                        continue
                    tokens, length = inp.tokens(pid)
                    original_tokens, original_length = old.tokens(pid)
                    assert np.array_equal(tokens, original_tokens) and length == original_length
                parity = True
            finally:
                old.close()
        query_ids = rng.choice(inp.all_query_count, min(64, inp.all_query_count), replace=False)
        geometry = inp.query_geometry(query_ids)
        direct = inp.all_query_features(query_ids)
        for pid in np.unique(geometry['profile_ids']):
            selected = np.flatnonzero(geometry['profile_ids'] == pid)
            assert np.array_equal(inp.features_at(int(pid), geometry['positions_mm'][selected]), direct[selected])
        kinds = Counter(profile['representation']['kind'] for profile in inp.profiles)
        rows.append(dict(job_key=job_key, profiles=len(inp.profiles), profile_kinds=dict(kinds),
                         canonical_queries=inp.n, all_actual_fragment_queries=inp.all_query_count,
                         original_feature_and_token_parity=parity, aliased_queries_use_actual_profile_arclength=True))
        print('FULL_CONSUMER_VERIFIED', job_key, len(inp.profiles), 'profiles', flush=True)
        del inp
        gc.collect()
    result = dict(status='PASS_FULL_LABEL_CONSUMER_CURRENT_READY_INPUTS_NOT_A_MODEL_FIT',
                  training_versions=170, training_families=len(dataset.families),
                  all_training_views_loaded=True, every_version_once_per_epoch=True,
                  family_equal_version_mass_verified=True, label_counts=dict(counts),
                  input_clouds_checked=len(ready), registered_input_clouds=len(dataset.jobs), rows=rows,
                  incomplete_full_fit_denied=blocked, remaining_inputs_not_claimed_validated=True,
                  held_and_protected_views_absent=True, original_inputs_checked_for_exact_parity=sum(row['original_feature_and_token_parity'] for row in rows),
                  fit_calls=0, model_predictions_scored=0, input_reference_geometry_feature_columns=0,
                  consumer_sha256=digest(ROOT / 'weld_pipeline/server_full_shard_dataset.py'),
                  registry_sha256=dataset.registry_sha256, source_sha256=digest(__file__),
                  seconds=time.perf_counter() - started)
    write(destination, result)
    print('FULL_DATASET_CONSUMER_VALIDATION_PASS', len(ready), 'inputs', 170, 'supervision views', flush=True)


if __name__ == '__main__':
    main()
