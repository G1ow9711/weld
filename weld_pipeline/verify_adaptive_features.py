"""Check exact feature equality against audited mixed caches on five pieces."""
import gc
import json
import time
from pathlib import Path
import numpy as np
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.adaptive_mixed_sequences import make_fast_mixed_sequences


def main():
    out = Path('weld_pipeline/reports/adaptive_feature_equivalence')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'receipt.json').exists():
        raise ValueError('Equivalence receipt already frozen')
    previous_path = Path('weld_pipeline/reports/fast_mixed_equivalence/receipt.json')
    previous = json.loads(previous_path.read_text())
    if previous['status'] != 'PASS':
        raise ValueError('Prior feature equivalence not established')
    sources = {**previous['sources'], 'weld_pipeline/adaptive_mixed_sequences.py': sha256('weld_pipeline/adaptive_mixed_sequences.py')}
    for path, wanted in sources.items():
        if sha256(path) != wanted:
            raise ValueError('Feature source changed: ' + path)
    rows = []
    for group in ['TD2-LB8A', '222-TT1A', '527-FR146A', '527-LB24A1', '234-FR71A']:
        folder = Path('weld_pipeline/data/mixed_surface_v1') / group
        old = load_piece(folder, verify=True)
        record = old['record']
        if sha256(record['input']) != record['pcd_sha256']:
            raise ValueError('Raw cloud changed')
        xyz = read_xyz_pcd(Path(record['input'])) * 1000
        start = time.perf_counter()
        new, _ = make_fast_mixed_sequences(xyz)
        seconds = time.perf_counter() - start
        if len(new) != len(old['sequences']):
            raise ValueError('Candidate count changed')
        for first, second in zip(new, old['sequences']):
            for key in ['xyz', 'features', 'direction']:
                np.testing.assert_array_equal(first[key], second[key])
            for key in ['sample_step_mm', 'type', 'curve_kind', 'curve', 'candidate_id']:
                if first.get(key) != second.get(key):
                    raise ValueError('Curve metadata changed: ' + key)
        row = {'group': group, 'feature_points': sum(len(s['xyz']) for s in new),
               'sequences': len(new), 'seconds': seconds, 'manifest_sha256': sha256(folder / 'manifest.json'),
               'input_sha256': record['pcd_sha256'], 'all_arrays_and_curve_metadata_equal': True}
        rows.append(row)
        (out / 'partial.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
        del old, new, xyz
        gc.collect()
    result = {'status': 'PASS', 'sources': sources, 'verifier_sha256': sha256(__file__),
              'previous_receipt_sha256': sha256(previous_path), 'groups': rows,
              'scope': 'Bitwise feature equivalence only. Concurrent diagnostic time is not speed acceptance.'}
    (out / 'receipt.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
