"""Prepare TRAIN reference-set supervision or input-only evaluation pools."""
import argparse
import time
from pathlib import Path
import numpy as np
from weld_pipeline.server_reference_parent_selection import build_competition_graph
from weld_pipeline.server_offline_parent_pair_utility import reference_pair_targets
from weld_pipeline.server_bound_parent_pool import load_parent_pool
from weld_pipeline.server_whole_parent_features import encode_parents
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'server_rebuild/expansion_v1'
DEST = BASE/'reference_parent_selection_v1'
OLD = ['21', 'BM79A_native_local', '10', '25', '36']
EVAL = ['new_cfc732e5f068', 'new_ceeae49d6777']


def evaluation_pool(key, out):
    source = BASE/'frozen_features/legacy_runtime'/('source_'+key)
    native = source/'run379'/('test_'+key)/'decoded.json'
    assert (native.parent/'solve_receipt.json').exists()
    lines = BASE/'reference_detector_balanced_endpoint_full_output_v1'/key
    freeze = read(lines/'prediction_freeze.json')
    assert freeze['GT_reads'] == 0
    assert freeze['compact_decoder_sha256'] == digest(ROOT/'weld_pipeline/server_compact_line_predictions.py')
    compact = lines/'compact_predictions.npz'
    assert digest(compact) == freeze['files'][compact.name]
    manifest = dict(native_parents=dict(path=str(native.relative_to(ROOT)), sha256=digest(native)),
                    line_predictions=dict(path=str(compact.relative_to(ROOT)), sha256=digest(compact)),
                    line_freeze_sha256=digest(lines/'prediction_freeze.json'), GT_reads=0)
    write(out/'parent_pool_manifest.json', manifest)
    pool, native_count = load_parent_pool(ROOT, manifest)
    inp = FixedQueries(source, key)
    encoded = encode_parents(inp, pool['parents'])
    np.save(out/'features.npy', encoded['features'])
    write(out/'parent_children.json', encoded['parent_children'])
    write(out/'child_metadata.json', encoded['child_metadata'])
    write(out/'input_freeze.json', dict(parent_pool_manifest_sha256=digest(out/'parent_pool_manifest.json'),
          input_bindings=inp.bindings, features_sha256=digest(out/'features.npy'),
          parent_children_sha256=digest(out/'parent_children.json'), child_metadata_sha256=digest(out/'child_metadata.json'),
          native_parents=native_count, new_line_parents=len(pool['parents'])-native_count,
          GT_reads=0, STEP_reads=0, fit_calls=0, encoder_module_sha256=digest(ROOT/'weld_pipeline/server_whole_parent_features.py'),
          numerical_point_limits=sum(r.get('numerically_collapsed_projection',False) for r in encoded['child_records'])))
    inp.close()
    return pool


def main(scope):
    control = read(BASE/'reference_parent_selection_controls_v1/receipt.json')
    assert control['status'] == 'PASS'
    for name, sha in control['modules'].items():
        assert digest(ROOT/'weld_pipeline'/f'{name}.py') == sha
    keys = OLD + read(BASE/'reference_query_task_v1/protocol.json')['training_keys'] if scope == 'train' else EVAL
    outroot = DEST/(scope+'_dataset'); outroot.mkdir(parents=True, exist_ok=False)
    write(outroot/'protocol.json', dict(scope=scope, keys=keys, neighbors=16, distance_ties_complete=True,
          no_hard_geometry_radius=True, pair_max_edges=4000000, reference_unmatched_is_not_physical_absence=True,
          full_parent_preserved=True, candidate_arrays_and_graph_freeze_before_targets=True,
          conditional_branch_evaluation=True, independent_acceptance=False, DEV_reads=0,
          modules=control['modules'], source_sha256=digest(__file__)))
    cases = []
    for key in keys:
        start = time.perf_counter(); out = outroot/key; out.mkdir()
        tablepath = None
        if key in OLD:
            folder = ROOT/'server_rebuild/unified_parent_scoring_v1'/('source_'+key)
            receipt = read(folder/'receipt.json')
            for name, sha in receipt['files'].items():
                assert digest(folder/name) == sha
            pool = read(folder/'parents.json')
            geometry = dict(kind='native', path=str((folder/'parents.json').relative_to(ROOT)), sha256=digest(folder/'parents.json'))
            tablepath = BASE/'whole_parent_reference_set_v1'/key/'reference_table.npz'
            assert digest(tablepath) == read(tablepath.parent/'receipt.json')['table_sha256']
        elif scope == 'train':
            folder = BASE/'augmented_whole_parent_pool_v1'/key
            frozen = read(folder/'input_freeze.json')
            for name in ['features', 'parent_children', 'child_metadata']:
                suffix = '.npy' if name == 'features' else '.json'
                assert digest(folder/(name+suffix)) == frozen[name+'_sha256']
            assert digest(folder/'parent_pool_manifest.json') == frozen['parent_pool_manifest_sha256']
            pool, _ = load_parent_pool(ROOT, read(folder/'parent_pool_manifest.json'))
            geometry = dict(kind='bound', path=str((folder/'parent_pool_manifest.json').relative_to(ROOT)), sha256=digest(folder/'parent_pool_manifest.json'))
            tablepath = folder/'reference_table.npz'
            assert digest(tablepath) == read(folder/'receipt.json')['table_sha256']
        else:
            folder = out
            print('EVALUATION_FEATURES', key, flush=True)
            pool = evaluation_pool(key, out)
            geometry = dict(kind='bound', path=str((out/'parent_pool_manifest.json').relative_to(ROOT)), sha256=digest(out/'parent_pool_manifest.json'))
        pc = read(folder/'parent_children.json'); meta = read(folder/'child_metadata.json')
        print('GRAPH', key, len(pool['parents']), flush=True)
        pairs, features = build_competition_graph(pool['parents'], meta, pc)
        np.save(out/'pairs.npy', pairs); np.save(out/'pair_features.npy', features)
        np.save(out/'child_counts.npy', np.asarray([p['child_count'] for p in pool['parents']], np.int64))
        files = {name: dict(path=str((folder/name).relative_to(ROOT)), sha256=digest(folder/name))
                 for name in ['features.npy', 'parent_children.json']}
        files.update({name: dict(path=str((out/name).relative_to(ROOT)), sha256=digest(out/name))
                      for name in ['pairs.npy', 'pair_features.npy', 'child_counts.npy']})
        row = dict(key=key, parents=len(pool['parents']), pairs=len(pairs), files=files, geometry=geometry,
                   full_parent_preserved=True, GT_reads=0, reference_input_at_inference=False)
        write(out/'input_freeze.json' if folder != out else out/'selection_input_freeze.json', row)
        if scope == 'train':
            with np.load(tablepath) as data:
                table = {name: data[name] for name in data.files}
            unary, penalties = reference_pair_targets(table, pairs)
            np.savez_compressed(out/'reference_targets.npz', unary=unary, penalties=penalties)
            row.update(reference_table=dict(path=str(tablepath.relative_to(ROOT)), sha256=digest(tablepath)),
                       targets=dict(path=str((out/'reference_targets.npz').relative_to(ROOT)), sha256=digest(out/'reference_targets.npz')),
                       positive_unary=int((unary > 0).sum()), positive_pair_penalties=int((penalties > 0).sum()))
        row['seconds'] = time.perf_counter() - start
        write(out/'receipt.json', row); cases.append(row)
        print('READY', key, len(pairs), round(row['seconds'], 2), flush=True)
    write(outroot/'receipt.json', dict(status='READY', scope=scope, cases=cases, fit_calls=0, DEV_reads=0,
                                      protocol_sha256=digest(outroot/'protocol.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=['train', 'evaluation'], required=True)
    main(parser.parse_args().scope)
