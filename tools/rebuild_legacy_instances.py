"""Original 974 orchestration with explicit rebuilt inputs and unchanged 978/973."""
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install

ADAPTER = install()
from weld_pipeline.sjg_local_runtime_904 import check_resources, digest, write
from weld_pipeline.sjg_density_modes_978 import density_partition, controls as density_controls
from weld_pipeline.sjg_overlap_instances_973 import stitch, controls as stitch_controls

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/best_baseline_restore'


def prepare(key):
    source = BASE / 'point_features' / ('source_' + key)
    case = json.loads((source / 'feature_manifest.json').read_text())
    assert case['status'] == 'POINT_FEATURES_COMPLETE' and case['core_coverage_complete']
    dest = BASE / 'instances' / ('source_' + key)
    dest.mkdir(parents=True, exist_ok=False)
    write(dest / 'protocol.json', dict(source_sha256=digest(__file__), adapter=ADAPTER,
          input_manifest_sha256=digest(source / 'feature_manifest.json'),
          original_sources={name:digest(ROOT / 'weld_pipeline' / name) for name in (
              'sjg_density_modes_978.py', 'sjg_overlap_instances_973.py', 'sjg_native_instance_freeze_974.py')},
          density_parameters=dict(radius=.4, seeds=128, iterations=12, min_mode_points=8),
          stitch_parameters=dict(min_votes=16, min_agreement=.9),
          cross_block_embedding_distances=False, blas_threads=2, GT_reads=0, STEP_reads=0,
          historical_instance_parity_proven=False))
    write(dest / 'controls.json', dict(density=density_controls(), stitch=stitch_controls()))
    n = case['original_points']
    seen = np.zeros(n, np.uint8)
    core_nodes = [np.full(n, -1, np.int32), np.full(n, -1, np.int32)]
    node_blocks = [[], []]
    local_rows = []
    started = time.perf_counter()
    try:
        for bi, entry in enumerate(case['blocks']):
            check_resources()
            path = source / entry['file']
            assert digest(path) == entry['sha256']
            with np.load(path) as z:
                core, context = z['original_point_ids'], z['context_original_point_ids']
                ids = np.r_[core, context]
                assert len(core) == entry['core_points'] and len(np.unique(ids)) == len(ids)
                assert np.all((ids >= 0) & (ids < n))
                np.add.at(seen, core, 1)
                record = dict(original_point_ids=ids, core_count=np.array(len(core), np.int64))
                for h, field in enumerate(('embedding', 'material_embedding')):
                    values = np.r_[z[field], z['context_' + field]]
                    labels = density_partition(values)
                    count = int(labels.max()) + 1
                    nodes = labels + len(node_blocks[h])
                    node_blocks[h].extend([bi] * count)
                    core_nodes[h][core] = nodes[:len(core)]
                    record[('face_node', 'material_node')[h]] = nodes
            file = dest / f'nodes_{bi:04d}.npz'
            np.savez(file, **record)
            local_rows.append(dict(file=file.name, sha256=digest(file), input_feature_sha256=entry['sha256']))
            if bi % 20 == 0 or bi + 1 == len(case['blocks']):
                progress = dict(source=key, block=bi+1, blocks=len(case['blocks']),
                                local_nodes=[len(v) for v in node_blocks], seconds=time.perf_counter()-started)
                write(dest / 'progress.json', progress)
                print('ORIGINAL_DENSITY', progress, flush=True)
        assert np.all(seen == 1) and all(np.all(v >= 0) for v in core_nodes)
        output = dict(core_coverage=seen)
        stats = {}
        for h, field in enumerate(('face_node', 'material_node')):
            count = len(node_blocks[h])
            codes = []
            for entry in local_rows:
                assert digest(dest / entry['file']) == entry['sha256']
                with np.load(dest / entry['file']) as z:
                    core = int(z['core_count']); ids = z['original_point_ids'][core:]
                    a = z[field][core:].astype(np.int64); b = core_nodes[h][ids].astype(np.int64)
                codes.append(np.minimum(a, b) * count + np.maximum(a, b))
            code, votes = np.unique(np.concatenate(codes), return_counts=True)
            del codes
            pairs = np.column_stack((code // count, code % count))
            labels, detail = stitch(node_blocks[h], pairs, votes)
            name = ('face_instance', 'material_instance')[h]
            output[name] = labels[core_nodes[h]]
            sizes = np.bincount(output[name])
            detail.update(local_nodes=count, global_instances=len(sizes), largest_instance_points=int(sizes.max()),
                          singleton_points=int((sizes == 1).sum()), points_in_instances_under_16=int(sizes[sizes < 16].sum()))
            stats[name] = detail
            np.savez(dest / (name + '_overlap.npz'), pairs=pairs, votes=votes,
                     node_block=np.asarray(node_blocks[h], np.int32), node_global=labels)
        file = dest / 'instances.npz'
        np.savez(file, **output)
        write(dest / 'receipt.json', dict(status='ORIGINAL_INSTANCE_PIPELINE_RECONSTRUCTED', source_key=key,
              original_points=n, input_sha256=case['input_sha256'], file=file.name, sha256=digest(file),
              blocks=local_rows, statistics=stats, all_original_points_covered_once=True,
              elapsed_seconds=time.perf_counter()-started, GT_reads=0, STEP_reads=0,
              historical_instance_parity_proven=False, weld_accuracy_measured=False))
        print('ORIGINAL_INSTANCES_COMPLETE', key, stats, flush=True)
    except BaseException:
        write(dest / 'failure.json', dict(error=traceback.format_exc(), completed_blocks=len(local_rows)))
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True, choices=['21', 'BM79A_native_local', '10', '25', '36'])
    args = ap.parse_args()
    with threadpool_limits(limits=2):
        prepare(args.key)
