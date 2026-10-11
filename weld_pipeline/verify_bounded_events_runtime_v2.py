"""Strict raw replay; compare runtime entries separately from training metadata.

The original 177 verifier remains unchanged. Cached entry-only supervision
indices are never runtime inputs. Every runtime entry field, candidate geometry,
425-feature value, and all seven raw model output channels must match exactly.
"""
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
from weld_pipeline.neural_bounded_events import BoundedEventNet, predict_fields, assemble_events
from weld_pipeline.train_forest_event_centers import load_cached

RUN = Path('weld_pipeline/runs/deep_bounded_events_175')
OUT = Path('weld_pipeline/reports/deep_bounded_runtime_181_20261002/runtime')
CONTEXT = Path('weld_pipeline/data/joint_workpiece_context_pilot154')
OLD_VERIFIER = Path('weld_pipeline/verify_bounded_events_runtime.py')
OLD_FAILURE = Path('weld_pipeline/reports/deep_bounded_runtime_177_20261002/mean_207-FR79M.json')
ENTRY_KEYS = ('sequence', 'node', 'type', 'start', 'stop', 'transform')
TRAINING_ONLY_KEYS = {'event_indices', 'foreground_indices'}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def compare_entries(fresh, cached):
    """Reject unexpected fields and compare every consumed field in order."""
    differences = []
    metadata = set()
    if len(fresh) != len(cached):
        differences.append({'reason': 'entry_count', 'fresh': len(fresh), 'cached': len(cached)})
    for index, (entry, reference) in enumerate(zip(fresh, cached)):
        extra = set(reference)-set(ENTRY_KEYS)
        metadata.update(extra)
        if set(entry) != set(ENTRY_KEYS) or not set(ENTRY_KEYS).issubset(reference) or not extra.issubset(TRAINING_ONLY_KEYS):
            differences.append({'index': index, 'reason': 'unexpected_entry_keys',
                'fresh_keys': sorted(entry), 'cached_keys': sorted(reference)})
            continue
        changed = [key for key in ENTRY_KEYS if entry[key] != reference[key]]
        if changed:
            differences.append({'index': index, 'reason': 'runtime_field_difference', 'fields': changed})
    return not differences, differences, sorted(metadata)


def compare_candidates(fresh, cached):
    """Compare complete candidate order and geometric fields before canonicalization."""
    if len(fresh) != len(cached):
        return False
    for sequence, reference in zip(fresh, cached):
        for key in ('type', 'curve_kind', 'sample_step_mm'):
            if sequence.get(key) != reference.get(key):
                return False
        for key in ('xyz', 'direction'):
            if not np.array_equal(np.asarray(sequence[key]), np.asarray(reference[key])):
                return False
    return True


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
    old_verifier_sha256 = sha256(OLD_VERIFIER)
    old_failure_sha256 = sha256(OLD_FAILURE)
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
                    for forbidden in (Path('weld_pipeline/data/bounded_runtime_negative_control.step'),
                            CONTEXT/name/'manifest.json', RUN/'FIT'/pooling/(name+'.json')):
                        try:
                            opener(forbidden, 'r')
                        except RuntimeError:
                            controls.append(True)
                        else:
                            raise ValueError('Bounded runtime negative read control failed')
                sequences, data = build_inputs(read_xyz_pcd(pcd)*1000.)
                fields = predict_fields(model, data, sequences, saved['mean'], saved['std'])
                foregrounds, events = assemble_events(data, sequences, fields, pooling)
                predictions = decode_event_sets(sequences, foregrounds, events)
            finally:
                builtins.open, io.open, Path.open = openers
            with np.load(RUN/'FIT'/pooling/(name+'_foregrounds.npz'), allow_pickle=False) as archive:
                grid_exact = np.array_equal(np.asarray([len(a) for a in foregrounds]), archive['lengths'])
                error = float(np.abs(np.concatenate(foregrounds)-archive['foregrounds']).max()) if grid_exact else None
            expected = json.loads((RUN/'FIT'/pooling/(name+'.json')).read_text('utf8'))
            with np.load(RUN/'FIT'/(name+'_raw_fields.npz'), allow_pickle=False) as archive:
                raw_grid_exact = np.array_equal(np.asarray([len(a) for a in fields]), archive['lengths'])
                joined_fields = np.concatenate(fields)
                raw_fields_exact = bool(raw_grid_exact and np.array_equal(joined_fields, archive['raw']))
                raw_error = float(np.max(np.abs(joined_fields-archive['raw']))) if raw_grid_exact else None
            cached, _, _, piece = load_cached(name)
            components = {key: bool(data[key].shape==cached[key].shape and np.array_equal(data[key], cached[key]))
                for key in ('local', 'point_context', 'nodes')}
            entries_exact, entry_differences, training_only_metadata = compare_entries(data['entries'], cached['entries'])
            candidates_exact = compare_candidates(sequences, piece['sequences'])
            features_exact = True; transforms_exact = True; compared_rows = 0; compared_candidates = 0
            feature_difference = None; maximum_feature_error = 0.
            expected_rows = sum(entry['stop']-entry['start'] for entry in data['entries'] if entry['type']=='horizontal')
            for entry, cached_entry in zip(data['entries'], cached['entries']):
                if entry['type']!='horizontal':
                    continue
                compared_candidates += 1
                sequence, _, tf = canonical_frame(sequences[entry['sequence']])
                cached_sequence, _, cached_tf = canonical_frame(piece['sequences'][entry['sequence']])
                if tf!=cached_tf:
                    transforms_exact = False
                if len(sequence['xyz']) != len(cached_sequence['xyz']):
                    features_exact = False
                    continue
                for start in range(0, len(sequence['xyz']), 4096):
                    ids = np.arange(start, min(start+4096, len(sequence['xyz'])))
                    actual = bounded_features(sequence, data, entry, ids)
                    reference = bounded_features(cached_sequence, cached, cached_entry, ids)
                    compared_rows += len(ids)
                    if not np.array_equal(actual, reference):
                        features_exact = False
                        rows, columns = np.nonzero(actual != reference)
                        maximum_feature_error = max(maximum_feature_error,
                            float(np.max(np.abs(actual.astype(float)-reference.astype(float)))))
                        if feature_difference is None:
                            feature_difference = {'sequence': entry['sequence'],
                                'row': start+int(rows[0]), 'column': int(columns[0]),
                                'fresh': float(actual[rows[0], columns[0]]),
                                'cached': float(reference[rows[0], columns[0]])}
            complete_feature_coverage = compared_rows==expected_rows and len(data['entries'])==len(cached['entries'])
            events_exact = events==expected['event_sets']; predictions_exact = predictions==expected['predictions']
            passed = bool(grid_exact and error==0. and events_exact and predictions_exact and entries_exact
                and candidates_exact and transforms_exact and features_exact and complete_feature_coverage
                and raw_fields_exact and raw_error==0. and all(components.values()) and len(controls)==9 and reads
                and sha256(pcd)==original and sha256(checkpoint)==frozen
                and sha256(OLD_VERIFIER)==old_verifier_sha256 and sha256(OLD_FAILURE)==old_failure_sha256)
            result = dict(checkpoint_sha256=frozen, input_sha256=original, foreground_grid_exact=bool(grid_exact),
                foreground_max_error=error, continuous_events_exact=events_exact, predictions_exact=predictions_exact,
                feature_components_exact=components, all_425_features_exact=features_exact, entries_exact=entries_exact,
                runtime_entry_fields=list(ENTRY_KEYS), entry_differences=entry_differences,
                isolated_training_metadata_keys=training_only_metadata, training_metadata_not_used_by_inference=True,
                complete_candidate_geometry_and_order_exact=candidates_exact, canonical_transforms_exact=transforms_exact,
                compared_feature_candidates=compared_candidates, compared_feature_rows=compared_rows,
                expected_feature_rows=expected_rows, complete_feature_coverage=complete_feature_coverage,
                feature_max_error=maximum_feature_error, first_feature_difference=feature_difference,
                raw_output_grid_exact=bool(raw_grid_exact), all_7_raw_output_channels_exact=raw_fields_exact,
                raw_output_max_error=raw_error,
                negative_controls=len(controls), PCD_reads=len(reads),
                runtime_STEP_used=False, runtime_labels_used=False, runtime_preprocessed_cache_or_run_outputs_used=False,
                all_checks_passed=passed, raw_replay_is_not_acceptance=True, full_contract_pass=False)
            dump(OUT/(pooling+'_'+name+'.json'), result); results[pooling][name] = result
            print('BOUNDED_RAW_RUNTIME', pooling, name, passed, 'FG_ERROR', error,
                'EVENTS_EXACT', events_exact, 'PREDICTIONS_EXACT', predictions_exact, flush=True)
            if not passed:
                raise ValueError('Bounded raw runtime differs from fixed FIT output')
            del data, sequences, cached, piece, fields, joined_fields, foregrounds, events, predictions, expected; gc.collect()
    dump(OUT/'complete.json', dict(results=results, raw_replay_count=4, all_checks_passed=True,
        checkpoint_sha256=frozen, training_complete_sha256=sha256(RUN/'complete.json'),
        source_sha256={__file__:sha256(__file__)}, old_failed_verifier_sha256=old_verifier_sha256,
        old_failure_receipt_sha256=old_failure_sha256, old_failure_preserved=True,
        exact_feature_comparison_not_relaxed=True, raw_output_comparison_added=True,
        default_predictor_replaced=False, full_contract_pass=False))


if __name__=='__main__':
    main()
