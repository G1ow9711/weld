"""Independent raw-PCD replay for the observed small-face proposal pool."""
import builtins
import io
import json
import time
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.small_finite_surface_events import small_surface_planes, small_face_positions
from weld_pipeline.mixed_sequences import make_mixed_sequences

AUDIT = Path('weld_pipeline/reports/small_finite_surface_events_165_20261002')
OUT = Path('weld_pipeline/reports/small_finite_surface_runtime_165_20261002')


def main():
    if OUT.exists():
        raise ValueError('Preserve small-face raw proof')
    done = json.loads((AUDIT/'complete.json').read_text('utf8'))
    OUT.mkdir(parents=True); results = {}
    for name in ('207-FR79M', '203-TT1A'):
        reference_path = AUDIT/(name+'.json')
        if sha256(reference_path) != done['piece_receipts'][name]:
            raise ValueError('Frozen complete small-face audit changed')
        reference = json.loads(reference_path.read_text('utf8'))
        parent = json.loads((Path('weld_pipeline/data/expanded_surface_v3')/name/'manifest.json').read_text('utf8'))
        pcd = Path(parent['record']['input']).resolve(); frozen = sha256(pcd)
        if frozen != reference['input_sha256']:
            raise ValueError('Original PCD changed')
        originals = builtins.open, io.open, Path.open
        forbidden = [Path(p).resolve() for p in ('DataSet', 'weld_pipeline/data', 'weld_pipeline/reports')]
        reads = []; controls = []
        def guard(opener):
            def checked(path, mode='r', *args, **kwargs):
                if isinstance(path, (str, bytes, Path)) and ('r' in str(mode) or '+' in str(mode)):
                    file = Path(path).resolve()
                    if file == pcd:
                        reads.append(str(file))
                    elif (any(file.is_relative_to(folder) for folder in forbidden)
                          or file.suffix.lower() in ('.step', '.stp', '.xlsx', '.xls')
                          or file.name.endswith('_welds.json')):
                        raise RuntimeError('Forbidden small-face runtime input: '+str(file))
                return opener(path, mode, *args, **kwargs)
            return checked
        began = time.perf_counter()
        builtins.open, io.open, Path.open = tuple(guard(o) for o in originals)
        try:
            for opener in (builtins.open, io.open, Path.open):
                try:
                    opener(Path('weld_pipeline/data/runtime_component_control.step'), 'r')
                except RuntimeError:
                    controls.append(True)
                else:
                    raise ValueError('Contact-component read control failed')
            raw = read_xyz_pcd(pcd)*1000.
            sequences, geometry = make_mixed_sequences(raw)
            pure = geometry['straight_geometry']; fresh = {}
            small_planes, small_evidence = small_surface_planes(raw, pure)
            for i, seq in enumerate(sequences):
                if seq['type'] == 'horizontal' and seq.get('curve_kind') != 'arc':
                    fresh[str(i)] = small_face_positions(seq, pure, small_planes)[0]
        finally:
            builtins.open, io.open, Path.open = originals
        with np.load(AUDIT/(name+'_pools.npz'), allow_pickle=False) as archive:
            same_keys = set(fresh) == set(archive.files)
            exact = same_keys and all(np.array_equal(values, archive[key]) for key, values in fresh.items())
            error = max(float(np.abs(values-archive[key]).max()) for key, values in fresh.items()) if exact else None
        meta = parent['record']['geometry']
        planes_exact = pure['vertical_planes'] == meta['vertical_planes']
        components_exact = pure['horizontal_lines'] == meta['horizontal_lines']
        small_planes_exact = small_planes == reference['planes']
        small_evidence_exact = small_evidence == reference['plane_evidence']
        passed = bool(exact and planes_exact and components_exact and small_planes_exact and small_evidence_exact and reads
                      and len(controls) == 3 and sha256(pcd) == frozen)
        result = dict(piece=name, input_sha256=frozen, proposal_pool_exact=exact,
            pool_max_error_mm=error, vertical_planes_exact=planes_exact,
            observed_components_exact=components_exact, small_planes_exact=small_planes_exact,
            small_surface_evidence_exact=small_evidence_exact, pool_arrays=len(fresh),
            PCD_reads=len(reads), negative_controls=len(controls), runtime_STEP_used=False,
            runtime_truth_used=False, runtime_preprocessed_cache_used=False,
            all_checks_passed=passed, elapsed_seconds=time.perf_counter()-began,
            raw_replay_is_not_geometry_acceptance=True, full_contract_pass=False)
        (OUT/(name+'.json')).write_text(json.dumps(result, indent=2), encoding='utf8')
        results[name] = result
        print('SMALL_FINITE_FACE_RAW_RUNTIME', name, passed, 'POOL_ERROR', error, flush=True)
        if not passed:
            raise ValueError('Pure PCD does not reproduce the audited small-face pool')
        del sequences, geometry, fresh
    (OUT/'complete.json').write_text(json.dumps(dict(pieces=results, all_checks_passed=True,
        audit_sha256=sha256(AUDIT/'complete.json'), sources={p:sha256(p) for p in
            [__file__, 'weld_pipeline/small_finite_surface_events.py', 'weld_pipeline/contact_component_event_candidates.py', 'weld_pipeline/physical_event_candidates.py',
             'weld_pipeline/mixed_sequences.py']}, runtime_adopted=False, full_contract_pass=False), indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
