"""Original point-index point-index merge on reconstructed run26 feature outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from weld_pipeline.server_legacy_runtime import install

ADAPTER = install()
from weld_pipeline.sjg_local_runtime_904 import digest, write, check_resources

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/best_baseline_restore'


def merge(key):
    source = BASE / 'run26_point_features' / ('source_' + key)
    case = json.loads((source / 'feature_manifest.json').read_text())
    assert case['status'] == 'POINT_FEATURES_COMPLETE' and case['core_coverage_complete']
    xyz = ROOT / 'server_rebuild/inputs_v1' / key / 'xyz_mm.npy'
    assert digest(xyz) == case['input_sha256']
    dest = BASE / 'run26_merged_point_features' / ('source_' + key)
    dest.mkdir(parents=True, exist_ok=False)
    n = case['original_points']
    seen = np.zeros(n, np.uint16)
    output = dict(embedding=np.empty((n, 16), np.float32), normal=np.empty((n, 3), np.float32),
                  boundary=np.empty(n, np.float32))
    for i, entry in enumerate(case['blocks']):
        if i % 20 == 0:
            check_resources()
        path = source / entry['file']
        assert digest(path) == entry['sha256']
        with np.load(path) as z:
            ids = z['original_point_ids']
            assert len(ids) == entry['core_points'] and np.all((ids >= 0) & (ids < n))
            np.add.at(seen, ids, 1)
            for field, value in output.items():
                assert z[field].shape == value[ids].shape and np.isfinite(z[field]).all()
                value[ids] = z[field]
    assert np.all(seen == 1)
    output['core_coverage'] = seen.astype(np.uint8)
    file = dest / 'features.npz'
    np.savez(file, **output)
    with np.load(file) as z:
        assert all(np.array_equal(z[k], v) for k, v in output.items())
    receipt = dict(source_key=key, original_points=n, input_sha256=case['input_sha256'],
          input_file=str(xyz.relative_to(ROOT)), feature_file=file.name, feature_sha256=digest(file),
          block_manifest_sha256=digest(source / 'feature_manifest.json'),
          all_original_points_covered_once=True, reload_bitwise_equal=True,
          source_sha256=digest(__file__),
          historical_feature_parity_proven=False, GT_reads=0, STEP_reads=0)
    write(dest / 'receipt.json', receipt)
    print('MERGED_ORIGINAL_MODEL_FEATURES', key, n, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True, choices=['21', 'BM79A_native_local', '10', '25', '36'])
    args = ap.parse_args()
    merge(args.key)
