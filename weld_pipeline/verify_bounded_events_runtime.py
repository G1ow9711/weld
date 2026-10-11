"""Guarded fresh-PCD replay of one fixed bounded model and both fixed poolings."""
import builtins
import gc
import io
import json
from pathlib import Path

import numpy as np
import torch

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.bounded_event_phase_v2 import bounded_features
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame
from weld_pipeline.continuous_event_decoder import decode_event_sets
from weld_pipeline.forest_event_centers import build_inputs
from weld_pipeline.neural_bounded_events import BoundedEventNet, infer_event_sets
from weld_pipeline.train_forest_event_centers import load_cached

RUN = Path('weld_pipeline/runs/deep_bounded_events_175')
OUT = Path('weld_pipeline/reports/deep_bounded_runtime_177_20261002')
CONTEXT = Path('weld_pipeline/data/joint_workpiece_context_pilot154')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def main():
    if OUT.exists():
        raise ValueError('Preserve bounded raw runtime proof')
    complete = json.loads((RUN/'complete.json').read_text('utf8'))
    config = json.loads((RUN/'config.json').read_text('utf8'))
    for path, digest in config['source_sha256'].items():
        if sha256(path)!=digest:
            raise ValueError('Frozen bounded source changed: '+path)
    checkpoint = RUN/'final.pt'; frozen = sha256(checkpoint)
    if frozen!=complete['checkpoint_sha256']:
        raise ValueError('Frozen bounded model changed')
    OUT.mkdir(parents=True); torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved = torch.load(checkpoint, map_location=device)
    model = BoundedEventNet().to(device); model.load_state_dict(saved['state_dict']); model.eval()
    results = {}
    for pooling in ('mean', 'or'):
        results[pooling] = {}
        for name in ('207-FR79M', '203-TT1A'):
            manifest = json.loads((CONTEXT/name/'manifest.json').read_text('utf8'))
            pcd = Path(manifest['input']).resolve(); original = sha256(pcd)
            if original!=manifest['input_sha256']:
                raise ValueError('Original raw bounded input changed')
            roots = [Path(p).resolve() for p in ('DataSet', 'weld_pipeline/data', 'weld_pipeline/reports', 'weld_pipeline/runs')]
            openers = builtins.open, io.open, Path.open; reads = []; controls = []

            def guard(opener):
                def checked(path, mode='r', *args, **kwargs):
                    if isinstance(path, (str, bytes, Path)) and ('r' in str(mode) or '+' in str(mode)):
                        file = Path(path).resolve()
                        if file==pcd:
                            reads.append(str(file))
                        elif (any(file.is_relative_to(root) for root in roots)
                                or file.suffix.lower() in ('.step', '.stp', '.xlsx', '.xls')
                                or file.name.endswith('_welds.json')):
                            raise RuntimeError('Forbidden bounded raw runtime reference: '+str(file))
                    return opener(path, mode, *args, **kwargs)
                return checked

            builtins.open, io.open, Path.open = tuple(guard(opener) for opener in openers)
            try:
                for opener in (builtins.open, io.open, Path.open):
                    try:
                        opener(Path('weld_pipeline/data/bounded_runtime_negative_control.step'), 'r')
                    except RuntimeError:
                        controls.append(True)
                    else:
                        raise ValueError('Bounded runtime negative read control failed')
                sequences, data = build_inputs(read_xyz_pcd(pcd)*1000.)
                foregrounds, events = infer_event_sets(model, data, sequences, saved['mean'], saved['std'], pooling)
                predictions = decode_event_sets(sequences, foregrounds, events)
            finally:
                builtins.open, io.open, Path.open = openers
            with np.load(RUN/'FIT'/pooling/(name+'_foregrounds.npz'), allow_pickle=False) as archive:
                grid_exact = np.array_equal(np.asarray([len(a) for a in foregrounds]), archive['lengths'])
                error = float(np.abs(np.concatenate(foregrounds)-archive['foregrounds']).max()) if grid_exact else None
            expected = json.loads((RUN/'FIT'/pooling/(name+'.json')).read_text('utf8'))
            cached, _, _, piece = load_cached(name)
            components = {key: bool(data[key].shape==cached[key].shape and np.array_equal(data[key], cached[key]))
                for key in ('local', 'point_context', 'nodes')}
            features_exact = True
            for entry, cached_entry in zip(data['entries'], cached['entries']):
                if entry!=cached_entry:
                    features_exact = False; break
                if entry['type']!='horizontal':
                    continue
                sequence, _, tf = canonical_frame(sequences[entry['sequence']])
                cached_sequence, _, cached_tf = canonical_frame(piece['sequences'][entry['sequence']])
                if tf!=cached_tf:
                    features_exact = False; break
                for start in range(0, len(sequence['xyz']), 4096):
                    ids = np.arange(start, min(start+4096, len(sequence['xyz'])))
                    if not np.array_equal(bounded_features(sequence, data, entry, ids),
                            bounded_features(cached_sequence, cached, cached_entry, ids)):
                        features_exact = False; break
            entries_exact = len(data['entries'])==len(cached['entries'])
            events_exact = events==expected['event_sets']; predictions_exact = predictions==expected['predictions']
            passed = bool(grid_exact and error==0. and events_exact and predictions_exact and entries_exact
                and features_exact and all(components.values()) and len(controls)==3 and reads
                and sha256(pcd)==original and sha256(checkpoint)==frozen)
            result = dict(checkpoint_sha256=frozen, input_sha256=original, foreground_grid_exact=bool(grid_exact),
                foreground_max_error=error, continuous_events_exact=events_exact, predictions_exact=predictions_exact,
                feature_components_exact=components, all_425_features_exact=features_exact, entries_exact=entries_exact,
                negative_controls=len(controls), PCD_reads=len(reads),
                runtime_STEP_used=False, runtime_labels_used=False, runtime_preprocessed_cache_or_run_outputs_used=False,
                all_checks_passed=passed, raw_replay_is_not_acceptance=True, full_contract_pass=False)
            dump(OUT/(pooling+'_'+name+'.json'), result); results[pooling][name] = result
            print('BOUNDED_RAW_RUNTIME', pooling, name, passed, 'FG_ERROR', error,
                'EVENTS_EXACT', events_exact, 'PREDICTIONS_EXACT', predictions_exact, flush=True)
            if not passed:
                raise ValueError('Bounded raw runtime differs from fixed FIT output')
            del data, sequences, cached, piece, foregrounds, events, predictions, expected; gc.collect()
    dump(OUT/'complete.json', dict(results=results, raw_replay_count=4, all_checks_passed=True,
        checkpoint_sha256=frozen, training_complete_sha256=sha256(RUN/'complete.json'),
        source_sha256={__file__:sha256(__file__)}, default_predictor_replaced=False, full_contract_pass=False))


if __name__=='__main__':
    main()
