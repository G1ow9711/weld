"""PCD-only actual inference and exact full-grid replay for a fixed variant."""
import argparse
import builtins
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.predict_near_events import load_model, predict_xyz

RUNS = Path('weld_pipeline/runs/near_event_capacity_156')
DATA = Path('weld_pipeline/data/joint_workpiece_context_pilot154')


def main(variant):
    run = RUNS/variant
    out = Path('weld_pipeline/reports/near_event_runtime_156_20261001')/variant
    if out.exists():
        raise ValueError('Preserve actual near-model runtime proof')
    receipt = json.loads((run/'complete.json').read_text('utf8'))
    checkpoint = run/'final.pt'; frozen = sha256(checkpoint)
    if frozen != receipt['checkpoint_sha256'] or receipt['variant'] != variant:
        raise ValueError('Frozen trial variant required')
    torch.set_num_threads(2); torch.backends.cudnn.benchmark = False
    out.mkdir(parents=True); results = {}
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
                        raise RuntimeError('Forbidden near-model runtime truth/cache: '+str(file))
                return opener(path, mode, *args, **kwargs)
            return checked

        started = time.perf_counter()
        builtins.open, io.open, Path.open = tuple(guard(opener) for opener in originals)
        try:
            for opener in (builtins.open, io.open, Path.open):
                try:
                    opener(Path('weld_pipeline/data/runtime_near_control.step'), 'r')
                except RuntimeError:
                    controls.append(True)
                else:
                    raise ValueError('Runtime read guard negative control failed')
            model = load_model(checkpoint)
            predictions, sequences, probabilities = predict_xyz(read_xyz_pcd(pcd)*1000., model)
        finally:
            builtins.open, io.open, Path.open = originals
        fresh = np.concatenate(probabilities); lengths = np.array([len(p) for p in probabilities], np.int64)
        with np.load(run/('FIT_'+name+'_heatmaps.npz'), allow_pickle=False) as cache:
            same_grid = np.array_equal(lengths, cache['lengths'])
            error = float(np.abs(fresh-cache['probabilities']).max()) if same_grid else None
        reference = json.loads((run/('FIT_'+name+'.json')).read_text('utf8'))
        same_predictions = predictions == reference['predictions']
        passed = bool(same_grid and error is not None and error <= 2e-6 and same_predictions
            and reads and len(controls) == 3 and sha256(pcd) == before and sha256(checkpoint) == frozen)
        result = dict(piece=name, variant=variant, sequences=len(sequences), heatmap_rows=len(fresh),
            input_sha256=before, checkpoint_sha256=frozen, guarded_PCD_reads=len(reads),
            negative_controls=len(controls), candidate_grid_exact=same_grid,
            probability_replay_max_error=error, predictions_exact=same_predictions,
            runtime_labels_used=False, runtime_STEP_used=False, runtime_preprocessed_cache_used=False,
            elapsed_seconds=time.perf_counter()-started, all_checks_passed=passed,
            geometry_score=reference['score'], full_contract_pass=False)
        (out/(name+'.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
        results[name] = result
        print('NEAR_EVENT_RAW_RUNTIME', variant, name, passed, 'HEATMAP_ERROR', error,
              'PREDICTIONS_EXACT', same_predictions, flush=True)
        if not passed:
            raise ValueError('Near raw PCD inference differs from frozen FIT evaluation')
        del model, fresh, probabilities; torch.cuda.empty_cache()
    complete = dict(variant=variant, pieces=results, all_checks_passed=True,
        checkpoint_sha256=frozen, held_read=False, observed_check_read=False, full_contract_pass=False,
        sources={p:sha256(p) for p in [__file__, 'weld_pipeline/predict_near_events.py',
                                     'weld_pipeline/near_event_votes.py', 'weld_pipeline/predict_joint_workpiece_events.py']})
    (out/'complete.json').write_text(json.dumps(complete, indent=2, ensure_ascii=False), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('physical_matched', 'millimeter_loss'), required=True)
    main(parser.parse_args().variant)
