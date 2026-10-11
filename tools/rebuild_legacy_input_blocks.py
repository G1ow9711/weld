"""Rebuild original 905 input partitions; no labels or learned features used."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.server_legacy_runtime import install

ADAPTER = install()
from weld_pipeline.sjg_local_blocks_905 import core_partitions
from weld_pipeline.sjg_local_runtime_904 import check_resources, digest, write

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'server_rebuild/best_baseline_restore/input_blocks'


def prepare(key):
    folder = DEST / ('source_' + key)
    src = ROOT / 'server_rebuild/inputs_v1' / key
    receipt = json.loads((src / 'receipt.json').read_text())
    path = src / 'xyz_mm.npy'
    assert digest(path) == receipt['xyz_sha256']
    assert receipt['unit'] == 'mm' and receipt['coordinate_frame'] == 'workpiece_local'
    if (folder / 'manifest.json').exists():
        prior = json.loads((folder / 'manifest.json').read_text())
        assert prior['input_sha256'] == receipt['xyz_sha256']
        for block in prior['blocks']:
            assert digest(DEST / block['file']) == block['sha256']
        return prior
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    xyz = np.load(path, mmap_mode='r', allow_pickle=False)
    assert xyz.dtype == np.float64 and xyz.shape == (receipt['original_points'], 3)
    assert np.isfinite(xyz).all()
    tree = cKDTree(xyz)
    seen = np.zeros(len(xyz), np.uint8)
    blocks = []
    for bi, core in enumerate(core_partitions(xyz)):
        if bi % 20 == 0:
            check_resources()
        near = np.asarray(tree.query(xyz[core].mean(0), k=min(8192, len(xyz)))[1], np.int64).reshape(-1)
        context = near[~np.isin(near, core)][:8192-len(core)]
        ids = np.r_[core, context].astype(np.int64)
        seen[core] += 1
        file = folder / f'block_{bi:04d}.npz'
        np.savez(file, xyz_mm=xyz[ids], original_point_ids=ids, core_count=np.int64(len(core)))
        with np.load(file) as z:
            assert np.array_equal(z['xyz_mm'], xyz[z['original_point_ids']])
            assert len(np.unique(z['original_point_ids'])) == len(ids)
            assert int(z['core_count']) == len(core)
        blocks.append(dict(file=str(file.relative_to(DEST)), sha256=digest(file), core_count=len(core)))
        if bi % 100 == 0:
            print(key, 'blocks', bi + 1, 'core points', int(seen.sum()), '/', len(xyz), flush=True)
    assert np.all(seen == 1)
    row = dict(source_key=key, role='TRAIN', input_file=str(path.relative_to(ROOT)),
               input_sha256=receipt['xyz_sha256'], coordinate_frame='workpiece_local', unit='mm',
               original_points=len(xyz), blocks=blocks, core_coverage_exactly_once=True,
               saved_block_xyz_equals_source=True, seconds=time.perf_counter()-started,
               historical_block_identity_proven=False)
    write(folder / 'manifest.json', row)
    xyz._mmap.close()
    del tree, xyz, seen
    gc.collect()
    print(key, 'COMPLETE', len(blocks), row['seconds'], flush=True)
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', nargs='+', default=['21', 'BM79A_native_local', '10', '25', '36'])
    args = ap.parse_args()
    assert len(args.keys) == len(set(args.keys)) and set(args.keys) <= {'21', 'BM79A_native_local', '10', '25', '36'}
    DEST.mkdir(parents=True, exist_ok=True)
    write(DEST / 'rebuild_protocol.json', dict(source_sha256=digest(__file__),
          original_partitioner_sha256=digest(ROOT / 'weld_pipeline/sjg_local_blocks_905.py'),
          adapter=ADAPTER, GT_reads=0, STEP_reads=0, optimizer_updates=0,
          cores_maximum=4096, block_maximum=8192, source_keys=args.keys,
          historical_parity_claim=False, resource_mapping_changed_for_linux=True))
    cases = [prepare(key) for key in args.keys]
    write(DEST / 'block_manifest.json', dict(cases=cases, GT_reads=0, STEP_reads=0,
          all_original_points_covered_once=True, historical_parity_proven=False))
