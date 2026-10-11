"""Guarded raw-PCD replay and feature parity of the frozen shared auxiliary heads."""
import builtins
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.forest_event_centers import build_inputs, infer
from weld_pipeline.train_forest_event_centers import load_cached

RUN = Path('weld_pipeline/runs/forest_event_centers_seed166')
OUT = Path('weld_pipeline/reports/forest_event_centers_runtime_166_20261002')
CONTEXT = Path('weld_pipeline/data/joint_workpiece_context_pilot154')


def main():
    if OUT.exists():
        raise ValueError('Preserve fixed auxiliary raw-model proof')
    complete = json.loads((RUN/'complete.json').read_text('utf8'))
    checkpoint = RUN/'candidate.joblib'; frozen = sha256(checkpoint)
    if frozen != complete['checkpoint_sha256']:
        raise ValueError('Frozen shared auxiliary checkpoint required')
    OUT.mkdir(parents=True); results = {}; bundle = joblib.load(checkpoint)
    for name in ('207-FR79M', '203-TT1A'):
        manifest = json.loads((CONTEXT/name/'manifest.json').read_text('utf8'))
        pcd = Path(manifest['input']).resolve(); original = sha256(pcd)
        if original != manifest['input_sha256']:
            raise ValueError('Original raw model source changed')
        openers = builtins.open, io.open, Path.open
        roots = [Path(p).resolve() for p in ['DataSet', 'weld_pipeline/data', 'weld_pipeline/reports']]
        reads = []; controls = []
        def guard(opener):
            def checked(path, mode='r', *args, **kwargs):
                if isinstance(path, (str, bytes, Path)) and ('r' in str(mode) or '+' in str(mode)):
                    file = Path(path).resolve()
                    if file == pcd:
                        reads.append(str(file))
                    elif any(file.is_relative_to(root) for root in roots) or file.suffix.lower() in ('.stp', '.step', '.xlsx', '.xls') or file.name.endswith('_welds.json'):
                        raise RuntimeError('Forbidden auxiliary runtime label/reference/cache: '+str(file))
                return opener(path, mode, *args, **kwargs)
            return checked
        began = time.perf_counter()
        builtins.open, io.open, Path.open = tuple(guard(opener) for opener in openers)
        try:
            for opener in (builtins.open, io.open, Path.open):
                try:
                    opener(Path('weld_pipeline/data/runtime_forest_center_control.step'), 'r')
                except RuntimeError:
                    controls.append(True)
                else:
                    raise ValueError('Auxiliary runtime read guard failed')
            sequences, raw_data = build_inputs(read_xyz_pcd(pcd)*1000.)
            probabilities = infer(bundle, raw_data, sequences)
            predictions = decode_intervals(sequences, probabilities, threshold=.5,
                weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
        finally:
            builtins.open, io.open, Path.open = openers
        fresh = np.concatenate(probabilities); lengths = np.asarray([len(p) for p in probabilities])
        with np.load(RUN/'FIT'/(name+'_heatmaps.npz'), allow_pickle=False) as archive:
            same_grid = np.array_equal(lengths, archive['lengths'])
            error = float(np.abs(fresh-archive['probabilities']).max()) if same_grid else None
        reference = json.loads((RUN/'FIT'/(name+'.json')).read_text('utf8'))
        cached_data, _, _, _ = load_cached(name)
        deltas = {}
        for key in ['local', 'point_context', 'nodes']:
            same_shape = raw_data[key].shape == cached_data[key].shape
            deltas[key] = dict(shape_exact=same_shape,
                values_exact=same_shape and np.array_equal(raw_data[key], cached_data[key]),
                max_error=float(np.abs(raw_data[key]-cached_data[key]).max()) if same_shape else None)
        same_prediction = predictions == reference['predictions']
        passed = bool(same_grid and error == 0. and same_prediction and all(v['values_exact'] for v in deltas.values())
            and len(controls) == 3 and reads and sha256(pcd) == original and sha256(checkpoint) == frozen)
        result = dict(piece=name, checkpoint_sha256=frozen, original_input_sha256=original,
            probability_grid_exact=same_grid, probability_max_error=error, predictions_exact=same_prediction,
            feature_components=deltas, negative_controls=len(controls), PCD_reads=len(reads),
            runtime_STEP_used=False, runtime_labels_used=False, runtime_preprocessed_cache_used=False,
            all_checks_passed=passed, elapsed_seconds=time.perf_counter()-began,
            original_fit_geometry_score=reference['score'], full_contract_pass=False)
        (OUT/(name+'.json')).write_text(json.dumps(result, indent=2), encoding='utf8')
        results[name] = result
        print('FOREST_CENTERS_RAW_RUNTIME', name, passed, 'PROB_ERROR', error,
              'PREDICTIONS_EXACT', same_prediction, 'FEATURES', deltas, flush=True)
        if not passed:
            raise ValueError('Raw auxiliary model or feature construction differs from frozen FIT evaluation')
        del cached_data, raw_data, sequences, probabilities, fresh
    (OUT/'complete.json').write_text(json.dumps(dict(pieces=results, all_checks_passed=True,
        checkpoint_sha256=frozen, source_sha256={p:sha256(p) for p in [__file__,
            'weld_pipeline/forest_event_centers.py', 'weld_pipeline/train_forest_event_centers.py',
            'weld_pipeline/small_finite_surface_events_v3.py']},
        actual_read_isolation_verified=True, full_contract_pass=False), indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
