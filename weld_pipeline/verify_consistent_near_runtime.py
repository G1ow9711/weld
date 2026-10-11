"""Run each fixed model on original PCDs under read guards and compare exact FIT outputs."""
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
from weld_pipeline.predict_consistent_near_events import load_model, predict_xyz, VARIANTS

RUN = Path('weld_pipeline/runs/near_event_consistency_160')
DATA = Path('weld_pipeline/data/joint_workpiece_context_pilot154')
OUT = Path('weld_pipeline/reports/near_consistency_runtime_160_20261001')


def main(variant):
    root = RUN/variant; out = OUT/variant
    if out.exists():
        raise ValueError('Preserve consistency model runtime proof')
    done = json.loads((root/'complete.json').read_text('utf8'))
    checkpoint = root/'final.pt'; frozen = sha256(checkpoint)
    if done['variant'] != variant or done['checkpoint_sha256'] != frozen:
        raise ValueError('Frozen completed consistency/mining model required')
    torch.set_num_threads(2); torch.backends.cudnn.benchmark = False
    out.mkdir(parents=True); results = {}
    for name in ('207-FR79M', '203-TT1A'):
        manifest = json.loads((DATA/name/'manifest.json').read_text('utf8'))
        pcd = Path(manifest['input']).resolve(); before = sha256(pcd)
        forbidden = [Path(p).resolve() for p in ('DataSet', 'weld_pipeline/data', 'weld_pipeline/reports')]
        originals = (builtins.open, io.open, Path.open); controls = []; reads = []

        def guard(opener):
            def checked(path, mode='r', *args, **kwargs):
                if isinstance(path, (str, bytes, Path)) and ('r' in str(mode) or '+' in str(mode)):
                    file = Path(path).resolve()
                    if file == pcd:
                        reads.append(str(file))
                    elif (any(file.is_relative_to(folder) for folder in forbidden)
                          or file.suffix.lower() in ('.step', '.stp', '.xlsx', '.xls')
                          or file.name.endswith('_welds.json')):
                        raise RuntimeError('Forbidden consistency runtime truth/cache: '+str(file))
                return opener(path, mode, *args, **kwargs)
            return checked

        began = time.perf_counter()
        builtins.open, io.open, Path.open = tuple(guard(o) for o in originals)
        try:
            for opener in (builtins.open, io.open, Path.open):
                try:
                    opener(Path('weld_pipeline/data/runtime_consistency_control.step'), 'r')
                except RuntimeError:
                    controls.append(True)
                else:
                    raise ValueError('Read guard negative control failed')
            model = load_model(checkpoint)
            predictions, sequences, probabilities = predict_xyz(read_xyz_pcd(pcd)*1000., model)
        finally:
            builtins.open, io.open, Path.open = originals
        fresh = np.concatenate(probabilities); lengths = np.array([len(p) for p in probabilities], np.int64)
        with np.load(root/('FIT_'+name+'_heatmaps.npz'), allow_pickle=False) as archive:
            same_grid = np.array_equal(lengths, archive['lengths'])
            error = float(np.abs(fresh-archive['probabilities']).max()) if same_grid else None
        original = json.loads((root/('FIT_'+name+'.json')).read_text('utf8'))
        predictions_exact = predictions == original['predictions']
        passed = bool(same_grid and error is not None and error <= 2e-6 and predictions_exact
            and reads and len(controls) == 3 and sha256(pcd) == before and sha256(checkpoint) == frozen)
        result = dict(variant=variant, piece=name, input_sha256=before, checkpoint_sha256=frozen,
            candidate_grid_exact=same_grid, heatmap_rows=len(fresh), probability_replay_max_error=error,
            predictions_exact=predictions_exact, PCD_reads=len(reads), negative_controls=len(controls),
            runtime_STEP_used=False, runtime_truth_used=False, runtime_preprocessed_cache_used=False,
            all_checks_passed=passed, geometry_score=original['score'],
            elapsed_seconds=time.perf_counter()-began, full_contract_pass=False)
        (out/(name+'.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
        results[name] = result
        print('CONSISTENCY_RAW_RUNTIME', variant, name, passed, 'HEATMAP_ERROR', error, flush=True)
        if not passed:
            raise ValueError('Raw consistency model does not reproduce full-grid FIT predictions')
        del model, fresh, probabilities; torch.cuda.empty_cache()
    (out/'complete.json').write_text(json.dumps(dict(variant=variant, pieces=results,
        checkpoint_sha256=frozen, all_checks_passed=True, held_or_observed_read=False, full_contract_pass=False,
        sources={p:sha256(p) for p in [__file__, 'weld_pipeline/predict_consistent_near_events.py',
                                      'weld_pipeline/near_event_votes.py', 'weld_pipeline/predict_joint_workpiece_events.py']}),
        indent=2, ensure_ascii=False), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--variant', choices=VARIANTS, required=True)
    main(parser.parse_args().variant)
