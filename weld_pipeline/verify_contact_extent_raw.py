"""Exercise the new physical head with fresh PCD-derived candidate points."""
import argparse
import builtins
import io
import json
from pathlib import Path

import joblib
import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.contact_plate_extent_model import predict_contact_extent
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates
from weld_pipeline.mixed_sequences import make_mixed_sequences


def run(pcd, outdir):
    pcd = Path(pcd).resolve(); out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    heads = Path('weld_pipeline/runs/contact_plate_extent_seed138/heads.joblib').resolve()
    forbidden = [Path(p).resolve() for p in ('weld_pipeline/data', 'weld_pipeline/reports', 'DataSet')]
    original_builtin, original_io, original_path = builtins.open, io.open, Path.open
    reads = []; blocked_controls = []

    def check(path, mode):
        reading = 'r' in str(mode) or '+' in str(mode)
        if isinstance(path, (str, bytes, Path)) and reading:
            path = Path(path).resolve()
            if path == pcd:
                reads.append(str(path))
                return
            if (any(path.is_relative_to(root) for root in forbidden) or
                path.suffix.lower() in ('.step', '.stp', '.xlsx', '.xls', '.cbf') or
                path.name.endswith('_welds.json')):
                raise RuntimeError('Forbidden reference/cache input: ' + str(path))

    def guarded(original):
        def call(path, mode='r', *args, **kwargs):
            check(path, mode)
            return original(path, mode, *args, **kwargs)
        return call

    builtins.open, io.open, Path.open = guarded(original_builtin), guarded(original_io), guarded(original_path)
    result = []
    try:
        for opener in (builtins.open, io.open, Path.open):
            try:
                opener(Path('weld_pipeline/data/forbidden_negative_control.step'), 'r')
            except RuntimeError:
                blocked_controls.append(True)
            else:
                raise ValueError('Reference isolation negative control failed')
        bundle = joblib.load(heads)
        xyz = read_xyz_pcd(pcd) * 1000.
        sequences, geometry = make_mixed_sequences(xyz)
        profile = FiniteContactPlates(xyz)
        available = [i for i, s in enumerate(sequences) if s.get('curve_kind') != 'arc']
        selected = available[::max(1, len(available) // 12)][:12]
        samples = []
        for i in selected:
            seq = sequences[i]
            positions = sorted(set((len(seq['xyz']) // 4, len(seq['xyz']) // 2, len(seq['xyz']) * 3 // 4)))
            for index in positions:
                q = seq['xyz'][index].astype(float)
                for normal in seq['features'][index, 60:66].reshape(2, 3):
                    if abs(normal[2]) >= .4:
                        continue
                    got = predict_contact_extent(bundle, profile, q, normal)
                    result.append({'candidate_index': i, 'pcd_position_index': index,
                                   'xyz_mm': q.tolist(), 'pcd_normal': normal.tolist(), 'prediction': got})
                    if len(samples) < 8:
                        samples.append((q.copy(), normal.copy(), got))
        shift = np.array([312.7, -148.1, 57.3])
        shifted = FiniteContactPlates(xyz + shift)
        largest = 0.
        for q, normal, wanted in samples:
            got = predict_contact_extent(bundle, shifted, q + shift, -normal)
            if got['state'] != wanted['state']:
                raise ValueError('Translation/normal-sign changed observability state')
            if got['state'] != 'unknown':
                difference = abs(got['contact_local_vertical_span_mm'] - wanted['contact_local_vertical_span_mm'])
                largest = max(largest, difference)
                if difference > .001:
                    raise ValueError('Real XYZ translation/normal-sign changed extent by more than .001mm')
    finally:
        builtins.open, io.open, Path.open = original_builtin, original_io, original_path
    if not reads or len(blocked_controls) != 3:
        raise ValueError('Actual PCD read and all three negative controls required')
    report = {'input': str(pcd), 'input_sha256': sha256(pcd), 'heads_sha256': sha256(heads),
        'guarded_pcd_reads': len(reads), 'negative_controls_passed': len(blocked_controls),
        'raw_candidate_count': len(sequences), 'exercised_point_normal_queries': len(result),
        'candidate_predictions': sum(r['prediction']['state'] != 'unknown' for r in result),
        'real_origin_and_normal_sign_checks': len(samples), 'maximum_extent_change_mm': largest,
        'runtime_step_read': False, 'runtime_truth_read': False, 'runtime_training_cache_read': False,
        'query_selection': '12 evenly spaced fresh PCD candidate indices, three fractional points, no reference selection',
        'scope': 'Physical-head raw-input and invariance checks; not weld detection or contract height acceptance.',
        'local_base_datum_resolved': False, 'full_contract_pass': False, 'source_sha256': sha256(__file__)}
    (out / 'raw_extent_queries.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    (out / 'raw_verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print('RAW_CONTACT_EXTENT_VERIFIED', report['exercised_point_normal_queries'],
          report['candidate_predictions'], largest, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pcd', required=True); parser.add_argument('--outdir', required=True)
    args = parser.parse_args(); run(args.pcd, args.outdir)
