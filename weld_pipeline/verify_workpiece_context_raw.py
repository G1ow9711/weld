"""Fresh raw-PCD feature replay before training, under reference/cache read guards."""
import builtins
import io
import json
import time
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline.signed_material_support import SignedMaterialSupport
from weld_pipeline.workpiece_relative_context import WorkpieceRelativeContext, frame_views

BASE = Path('weld_pipeline/data/material_support_v1')
DATA = Path('weld_pipeline/data/workpiece_context_v1')
OUT = Path('weld_pipeline/reports/workpiece_context_raw_153_20261001')


def main():
    if OUT.exists():
        raise ValueError('Preserve raw context verification')
    config = json.loads((DATA/'config.json').read_text('utf8'))
    for source, digest in config['sources'].items():
        if sha256(source) != digest:
            raise ValueError('Prepared relative context source changed')
    OUT.mkdir(parents=True); results = {}
    for name in ('207-FR79M', '203-TT1A', '521-CM56A'):
        manifest = json.loads((DATA/name/'manifest.json').read_text('utf8'))
        original = json.loads((BASE/name/'manifest.json').read_text('utf8'))
        pcd = Path(manifest['input']).resolve(); before = sha256(pcd)
        originals = (builtins.open, io.open, Path.open)
        forbidden = [Path(p).resolve() for p in ('DataSet', 'weld_pipeline/data', 'weld_pipeline/reports')]
        reads = []; controls = []; frame_errors = []
        def guard(opener):
            def checked(path, mode='r', *args, **kwargs):
                if isinstance(path, (str, bytes, Path)) and ('r' in str(mode) or '+' in str(mode)):
                    file = Path(path).resolve()
                    if file == pcd:
                        reads.append(str(file))
                    elif (any(file.is_relative_to(root) for root in forbidden)
                          or file.suffix.lower() in ('.step', '.stp', '.xlsx', '.xls')
                          or file.name.endswith('_welds.json')):
                        raise RuntimeError('Forbidden runtime reference/cache: '+str(file))
                return opener(path, mode, *args, **kwargs)
            return checked
        started = time.perf_counter()
        builtins.open, io.open, Path.open = tuple(guard(opener) for opener in originals)
        try:
            for opener in (builtins.open, io.open, Path.open):
                try:
                    opener(Path('weld_pipeline/data/forbidden_context_control.step'), 'r')
                except RuntimeError:
                    controls.append(True)
                else:
                    raise ValueError('Reference/cache guard negative control failed')
            xyz = read_xyz_pcd(pcd)*1000.
            sequences, _ = make_mixed_sequences(xyz)
            local = SignedMaterialSupport(xyz); context = WorkpieceRelativeContext(xyz)
            blocks = []; keys = []
            for si, sequence in enumerate(sequences):
                if sequence.get('curve_kind') == 'arc':
                    continue
                indices = np.unique(np.round(np.array([0., .25, .5, .75, 1.])
                    *(len(sequence['xyz'])-1)).astype(int))
                block = np.c_[local.describe_sequence(sequence, indices),
                              context.describe_sequence(sequence, indices)]
                blocks.append(block); keys.extend((si, int(i)) for i in indices)
                if len(frame_errors) < 32:
                    index = int(indices[len(indices)//2]); q = sequence['xyz'][index]
                    normals = sequence['features'][index, 60:66].reshape(2, 3)
                    tangent = sequence['direction']; expected = block[len(indices)//2]
                    views = frame_views(expected)
                    for exchange in (False, True):
                        ns = normals[::-1] if exchange else normals
                        for a, b in ((1., 1.), (-1., 1.), (1., -1.), (-1., -1.)):
                            nn = ns*np.array([a, b])[:, None]
                            for reverse in (False, True):
                                ts = -tangent if reverse else tangent
                                f = np.r_[local.describe(q, nn, ts), context.describe(q, nn, ts)]
                                error = min(float(np.abs(f-v).max()) for v in views)
                                frame_errors.append(error)
                                if error > 2e-6:
                                    raise ValueError('Actual contact frame differs from analytic permutation')
            fresh = np.concatenate(blocks)
        finally:
            builtins.open, io.open, Path.open = originals
        expected = np.c_[np.load(BASE/name/'signed.npy', allow_pickle=False),
                         np.load(DATA/name/'context.npy', allow_pickle=False)]
        same_order = keys == [(r['sequence'], r['sample_index']) for r in original['entries']]
        same_features = same_order and np.array_equal(fresh, expected)
        if not same_features or len(controls) != 3 or not reads or sha256(pcd) != before:
            raise ValueError('Fresh PCD/training relative features differ: '+name)
        np.save(OUT/(name+'_fresh_features.npy'), fresh)
        result = dict(piece=name, rows=len(fresh), fresh_reads=len(reads), negative_controls=len(controls),
            frame_cases=len(frame_errors), frame_max_error=max(frame_errors), candidate_query_order_exact=same_order,
            training_feature_values_exact=same_features, input_unchanged=True, input_sha256=before,
            base_manifest_sha256=sha256(BASE/name/'manifest.json'), context_manifest_sha256=sha256(DATA/name/'manifest.json'),
            fresh_feature_sha256=sha256(OUT/(name+'_fresh_features.npy')), elapsed_seconds=time.perf_counter()-started,
            runtime_STEP_used=False, runtime_labels_used=False, runtime_preprocessed_cache_used=False,
            all_checks_passed=True, full_contract_pass=False)
        results[name] = result
        (OUT/(name+'.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
        print('WORKPIECE_CONTEXT_RAW', name, len(fresh), 'EXACT', True, flush=True)
    (OUT/'complete.json').write_text(json.dumps(dict(pieces=results, all_checks_passed=True,
        source_sha256={p:sha256(p) for p in [__file__, 'weld_pipeline/workpiece_relative_context.py',
            'weld_pipeline/signed_material_support.py', 'weld_pipeline/mixed_sequences.py']},
        observed_check_read=False, full_contract_pass=False), indent=2, ensure_ascii=False), encoding='utf8')


if __name__ == '__main__':
    main()
