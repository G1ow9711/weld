"""Compare raw-only regenerated geometry and all core features and axial context with frozen caches."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from training.labels import sha256
from training.dataio import read_xyz_pcd
from weld_pipeline.pooled_shared_mixed_sequences import make_fast_mixed_sequences
from weld_pipeline.axial_context_features import append_axial_context
from weld_pipeline.axial_context_cache import load_piece


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--groups', nargs='+', required=True)
    args = parser.parse_args()
    root, out = Path(args.data), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((root/'split.json').read_text(encoding='utf-8'))
    old_path = Path('weld_pipeline/reports/pooled_shared_feature_equivalence/receipt.json')
    old = json.loads(old_path.read_text(encoding='utf-8'))
    assert old['status'] == 'PASS'
    sources = dict(old['sources'])
    sources.update({str(p): sha256(p) for p in [Path(__file__),
        Path('weld_pipeline/axial_context_features.py'), Path('weld_pipeline/axial_context_cache.py')]})
    for source, wanted in sources.items():
        if sha256(source) != wanted:
            raise ValueError('Source changed: '+source)
    rows = []
    for name in args.groups:
        folder = (root/split.get('piece_roots', {}).get(name, name)).resolve()
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        provenance = manifest['provenance']
        if provenance['augmentation_audit'] is not None:
            raise ValueError('Only original input coordinates in this check')
        source = Path(provenance['raw_input'])
        if sha256(source) != provenance['raw_input_sha256']:
            raise ValueError('Input changed')
        # All generation precedes loading the cached targets, truth and features.
        start = time.perf_counter()
        xyz = read_xyz_pcd(source)*1000.
        sequences, _ = make_fast_mixed_sequences(xyz)
        sequences, context_stats = append_axial_context(xyz, sequences)
        seconds = time.perf_counter()-start
        cached = load_piece(folder)
        if len(sequences) != len(cached['sequences']):
            raise ValueError('Sequence count differs')
        for index, (actual, expected) in enumerate(zip(sequences, cached['sequences'])):
            # A proposal diagnostic is intentionally absent from cached model
            # input; every serialized geometric and model input field is checked.
            diagnostic_only = {'wall_offset_from_hole_boundary_mm'}
            if actual.keys() - diagnostic_only != expected.keys():
                raise ValueError(f'{name}/{index}: metadata keys differ')
            for key in expected:
                if key == 'features':
                    for lo in range(0, len(actual[key]), 4096):
                        if not np.array_equal(actual[key][lo:lo+4096], expected[key][lo:lo+4096]):
                            raise ValueError(f'{name}/{index}: feature values differ')
                elif not np.array_equal(actual[key], expected[key]):
                    raise ValueError(f'{name}/{index}: {key} differs')
        row = {'group': name, 'sequences': len(sequences),
               'points': sum(len(s['xyz']) for s in sequences),
               'all_core_features_context_and_geometry_exact': True,'context_points':context_stats['context_points'],
               'manifest_sha256': sha256(folder/'manifest.json'),
               'input_sha256': sha256(source), 'seconds': seconds}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del cached, sequences, xyz
    receipt = {'status': 'PASS', 'groups': rows, 'sources': sources,
               'base_equivalence_sha256': sha256(old_path),
               'scope': 'Raw feature regeneration equality, not accuracy or speed acceptance.'}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
