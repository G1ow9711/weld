"""Rebuild fixed geometric proposals from original XYZ under read isolation."""
import builtins
import io
import json
import time
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline.physical_event_candidates import candidate_positions

AUDIT = Path('weld_pipeline/reports/physical_event_candidates_161_20261002')
OUT = Path('weld_pipeline/reports/physical_event_runtime_161_20261002')


def main():
    if OUT.exists():
        raise ValueError('Preserve raw physical-event proof')
    audit = json.loads((AUDIT/'complete.json').read_text('utf8'))
    OUT.mkdir(parents=True); results = {}
    for name in ('207-FR79M', '203-TT1A'):
        reference = json.loads((AUDIT/(name+'.json')).read_text('utf8'))
        parent = json.loads((Path('weld_pipeline/data/expanded_surface_v3')/name/'manifest.json').read_text('utf8'))
        pcd = Path(parent['record']['input']).resolve(); frozen = sha256(pcd)
        if frozen != reference['input_sha256']:
            raise ValueError('Raw input changed before independent proposal rebuild')
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
                        raise RuntimeError('Forbidden physical-event runtime read: '+str(file))
                return opener(path, mode, *args, **kwargs)
            return checked
        began = time.perf_counter()
        builtins.open, io.open, Path.open = tuple(guard(o) for o in originals)
        try:
            for opener in (builtins.open, io.open, Path.open):
                try:
                    opener(Path('weld_pipeline/data/runtime_physical_event_control.step'), 'r')
                except RuntimeError:
                    controls.append(True)
                else:
                    raise ValueError('Proposal read guard did not reject the negative control')
            sequences, geometry = make_mixed_sequences(read_xyz_pcd(pcd)*1000.)
            fresh = {}
            planes = geometry['straight_geometry']['vertical_planes']
            for i, seq in enumerate(sequences):
                if seq['type'] != 'horizontal' or seq.get('curve_kind') == 'arc':
                    continue
                for mode in ('finite', 'unlimited'):
                    fresh[str(i)+'_'+mode] = candidate_positions(seq, planes, finite=(mode == 'finite'))[0]
        finally:
            builtins.open, io.open, Path.open = originals
        with np.load(AUDIT/(name+'_pools.npz'), allow_pickle=False) as archive:
            same_keys = set(fresh) == set(archive.files)
            exact = same_keys and all(np.array_equal(value, archive[key]) for key, value in fresh.items())
            max_error = max(float(np.abs(value-archive[key]).max()) for key, value in fresh.items()) if exact else None
        same_planes = planes == parent['record']['geometry']['vertical_planes']
        passed = bool(exact and same_planes and reads and len(controls) == 3 and sha256(pcd) == frozen)
        result = dict(piece=name, input_sha256=frozen, exact_pool_arrays=exact,
            pool_position_max_error_mm=max_error, vertical_planes_exact=same_planes,
            pool_arrays=len(fresh), PCD_reads=len(reads), negative_controls=len(controls),
            runtime_STEP_used=False, runtime_truth_used=False, runtime_preprocessed_cache_used=False,
            all_checks_passed=passed, elapsed_seconds=time.perf_counter()-began,
            diagnostic_only=True, full_contract_pass=False)
        (OUT/(name+'.json')).write_text(json.dumps(result, indent=2), encoding='utf8')
        results[name] = result
        print('PHYSICAL_EVENT_RAW_RUNTIME', name, passed, 'POOL_ERROR', max_error, flush=True)
        if not passed:
            raise ValueError('Raw geometry does not reproduce the audited candidate pool')
        del sequences, geometry, fresh
    (OUT/'complete.json').write_text(json.dumps(dict(pieces=results, all_checks_passed=True,
        audit_complete_sha256=sha256(AUDIT/'complete.json'),
        candidate_coverage=audit['coverage'], sources={p:sha256(p) for p in
            [__file__, 'weld_pipeline/physical_event_candidates.py', 'weld_pipeline/mixed_sequences.py']},
        runtime_adopted=False, full_contract_pass=False), indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
