"""Actual experimental model inference under PCD-only read guards, then cached replay comparison."""
import builtins
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.predict_joint_workpiece_events import load_model, predict_xyz

RUN = Path('weld_pipeline/runs/joint_workpiece_event_capacity_seed154')
DATA = Path('weld_pipeline/data/joint_workpiece_context_pilot154')
OUT = Path('weld_pipeline/reports/joint_workpiece_runtime_154_20261001')


def main():
    if OUT.exists():
        raise ValueError('Preserve actual raw event inference proof')
    receipt = json.loads((RUN/'complete.json').read_text('utf8'))
    checkpoint = RUN/'final.pt'; frozen = sha256(checkpoint)
    if frozen != receipt['checkpoint_sha256']:
        raise ValueError('Current fixed event model required')
    torch.set_num_threads(2); torch.backends.cudnn.benchmark = False
    OUT.mkdir(parents=True); results = {}
    for name in ('207-FR79M', '203-TT1A'):
        manifest = json.loads((DATA/name/'manifest.json').read_text('utf8'))
        pcd = Path(manifest['input']).resolve(); before = sha256(pcd)
        originals = (builtins.open, io.open, Path.open)
        forbidden = [Path(p).resolve() for p in ('DataSet', 'weld_pipeline/data', 'weld_pipeline/reports')]
        controls = []; reads = []
        def guard(opener):
            def checked(path, mode='r', *args, **kwargs):
                if isinstance(path, (str, bytes, Path)) and ('r' in str(mode) or '+' in str(mode)):
                    file = Path(path).resolve()
                    if file == pcd:
                        reads.append(str(file))
                    elif (any(file.is_relative_to(root) for root in forbidden)
                          or file.suffix.lower() in ('.step', '.stp', '.xlsx', '.xls')
                          or file.name.endswith('_welds.json')):
                        raise RuntimeError('Forbidden model runtime reference/cache: '+str(file))
                return opener(path, mode, *args, **kwargs)
            return checked
        started = time.perf_counter()
        builtins.open, io.open, Path.open = tuple(guard(opener) for opener in originals)
        try:
            for opener in (builtins.open, io.open, Path.open):
                try:
                    opener(Path('weld_pipeline/data/runtime_context_control.step'), 'r')
                except RuntimeError:
                    controls.append(True)
                else:
                    raise ValueError('Runtime reference/cache negative control failed')
            model = load_model(checkpoint)
            predictions, sequences, probabilities = predict_xyz(read_xyz_pcd(pcd)*1000., model)
        finally:
            builtins.open, io.open, Path.open = originals
        fresh = np.concatenate(probabilities); lengths = np.array([len(p) for p in probabilities], np.int64)
        with np.load(RUN/('FIT_'+name+'_heatmaps.npz'), allow_pickle=False) as cache:
            length_equal = np.array_equal(lengths, cache['lengths'])
            error = float(np.abs(fresh-cache['probabilities']).max()) if length_equal else None
        old = json.loads((RUN/('FIT_'+name+'.json')).read_text('utf8'))
        predictions_equal = predictions == old['predictions']
        passed = bool(length_equal and error is not None and error <= 2e-6 and predictions_equal
            and reads and len(controls) == 3 and sha256(pcd) == before and sha256(checkpoint) == frozen)
        result = dict(piece=name, sequences=len(sequences), heatmap_rows=len(fresh), input_sha256=before,
            checkpoint_sha256=frozen, guarded_PCD_reads=len(reads), negative_controls=len(controls),
            candidate_grid_exact=length_equal, probability_replay_max_error=error, predictions_exact=predictions_equal,
            runtime_labels_used=False, runtime_STEP_used=False, runtime_preprocessed_cache_used=False,
            elapsed_seconds=time.perf_counter()-started, all_checks_passed=passed,
            geometry_score=old['score'], full_contract_pass=False)
        (OUT/(name+'.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
        results[name] = result
        print('JOINT_WORKPIECE_RUNTIME', name, passed, 'PROB_ERROR', error, 'PREDICTIONS_EXACT', predictions_equal, flush=True)
        if not passed:
            raise ValueError('Pure-PCD model differs from cached training evaluation')
        del model, fresh, probabilities; torch.cuda.empty_cache()
    (OUT/'complete.json').write_text(json.dumps(dict(pieces=results, all_checks_passed=True,
        sources={p:sha256(p) for p in [__file__, 'weld_pipeline/predict_joint_workpiece_events.py',
            'weld_pipeline/joint_workpiece_event_votes.py', 'weld_pipeline/workpiece_relative_context.py']},
        checkpoint_sha256=frozen, held_read=False, observed_check_read=False, full_contract_pass=False),
        indent=2, ensure_ascii=False), encoding='utf8')


if __name__ == '__main__':
    main()
