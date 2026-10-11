"""Raw PCD geometry and independently associated height candidates, then score."""
import argparse
import builtins
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.contract_child_evaluation import child_units, score_children
from weld_pipeline.contract_horizontal_model import from_checkpoint
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates
from weld_pipeline.full_fit_hybrid import infer_from_sequences, point_structure
from weld_pipeline.local_height_datum_model import estimate_children
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline.prepare_contract_child import child_attributes
from weld_pipeline.train_endpoint_process_attributes import known_label


def raw_infer(pcd, out):
    started = time.perf_counter(); torch.set_num_threads(3)
    geometry_path = Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4/best.pt')
    height_path = Path('weld_pipeline/runs/local_height_datum_seed140/heads.joblib')
    state = torch.load(geometry_path, map_location='cpu')
    verify_cached_inference_sources(state, geometry_path); verify_cached_inference_sources(state['horizontal'], geometry_path)
    model = from_checkpoint(state['horizontal']).cuda().eval(); bundle = joblib.load(height_path)
    xyz = read_xyz_pcd(pcd) * 1000.
    sequences, geometry = make_mixed_sequences(xyz)
    structural, process = point_structure(xyz, sequences)
    predictions, _ = infer_from_sequences(state, model, sequences, structural, process)
    heights = estimate_children(bundle, FiniteContactPlates(xyz), sequences, predictions)
    torch.cuda.synchronize()
    (out / 'predictions.json').write_text(json.dumps(predictions, indent=2), encoding='utf8')
    (out / 'height_candidates.json').write_text(json.dumps(heights, indent=2, ensure_ascii=False), encoding='utf8')
    return predictions, heights, {'points': len(xyz), 'pcd_candidates': len(sequences), 'geometry': geometry,
        'elapsed_seconds_diagnostic_only': time.perf_counter() - started,
        'geometry_checkpoint_sha256': sha256(geometry_path), 'height_heads_sha256': sha256(height_path),
        'height_contact_selection': 'predicted_line_geometric_association_to_fresh_PCD_candidates_no_truth_mapping',
        'model_combination': 'v4_retained_geometry_plus_separate140_candidate_head_not_released_as_new_contract_model',
        'full_contract_pass': False}


def score(piece, prediction, heights, name):
    geometry = score_children(prediction, piece['truth'])
    pc, tc = child_units(prediction), child_units(piece['truth'])
    indexed = {(r['parent_id'], r['child_index']): r['fields'] for r in heights}
    mapping = {m['truth_index']: m for m in geometry['matches']}
    paths = Path('weld_pipeline/runs/local_height_datum_seed140')
    role = 'FIT' if (paths / ('FIT_datum_' + name + '.json')).exists() else 'OBSERVED_CHECK'
    physical = json.loads((paths / (role + '_datum_' + name + '.json')).read_text('utf8'))
    targets = {(r['parent'], r['child'], r['endpoint']): r['measurement'] for r in physical['rows']}
    totals = {field: {'truth_children': len(tc), 'old_known': 0, 'old_reference_valid': 0,
                     'new_candidate_known': 0, 'new_candidate_valid': 0} for field in ('start_height', 'end_height')}
    rows = []
    for ti, truth in enumerate(tc):
        attrs = child_attributes(piece['truth'][truth['parent_index']], truth['segment_index'])
        fields = None
        if ti in mapping:
            match = mapping[ti]; selected = pc[match['prediction_index']]
            fields = indexed[(selected['parent_id'], selected['segment_index'])]
            if match['reversed']:
                fields = {'start_height': fields['end_height'], 'end_height': fields['start_height']}
        comparisons = {}
        for field, endpoint in (('start_height', 'start'), ('end_height', 'end')):
            old = known_label(field, attrs[field])
            candidate = targets.get((truth['parent_id'], truth['segment_index'], endpoint), {})
            new = candidate.get('height_mm') if candidate.get('state') == 'LOCAL_CONTACT_HEIGHT_CANDIDATE' else None
            got = fields[field] if fields is not None else {'state': 'unknown', 'value': None, 'reason': 'GEOMETRY_MISS'}
            value = got.get('value')
            old_ok = value is not None and old is not None and abs(value - old['value']) <= 3.
            new_ok = value is not None and new is not None and abs(value - new) <= 3.
            totals[field]['old_known'] += old is not None; totals[field]['old_reference_valid'] += old_ok
            totals[field]['new_candidate_known'] += new is not None; totals[field]['new_candidate_valid'] += new_ok
            comparisons[field] = {'prediction': got, 'old_reference': old, 'new_unreviewed_candidate_mm': new,
                                  'old_reference_pass': old_ok, 'new_candidate_pass': new_ok}
        rows.append({'child_id': truth['id'], 'fields': comparisons})
    return {'name': name, 'geometry': geometry, 'height_counts': totals, 'rows': rows,
        'geometry_misses_and_unknown_included': True, 'customer_reviewed_new_truth': False,
        'formal_contract_height_score': False, 'full_contract_pass': False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--name', required=True); args = parser.parse_args()
    split = json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    piece = load_piece(split['piece_roots'][args.name], verify=True); pcd = Path(piece['record']['input']).resolve()
    out = Path('weld_pipeline/runs/local_height_datum_seed140') / ('raw_joint_' + args.name)
    out.mkdir(parents=True, exist_ok=True)
    forbidden = [Path(root).resolve() for root in ('weld_pipeline/data', 'weld_pipeline/reports', 'DataSet')]
    original_builtin, original_io, original_path = builtins.open, io.open, Path.open
    reads = []; controls = []
    def check(path, mode):
        if isinstance(path, (str, bytes, Path)) and ('r' in str(mode) or '+' in str(mode)):
            path = Path(path).resolve()
            if path == pcd:
                reads.append(str(path)); return
            if (any(path.is_relative_to(root) for root in forbidden) or path.suffix.lower() in
                ('.step', '.stp', '.xlsx', '.xls', '.cbf') or path.name.endswith('_welds.json')):
                raise RuntimeError('Forbidden reference/cache read: ' + str(path))
    def guard(original):
        def checked(path, mode='r', *args, **kwargs):
            check(path, mode); return original(path, mode, *args, **kwargs)
        return checked
    builtins.open, io.open, Path.open = guard(original_builtin), guard(original_io), guard(original_path)
    try:
        for opener in (builtins.open, io.open, Path.open):
            try:
                opener(Path('weld_pipeline/data/negative_control.step'), 'r')
            except RuntimeError:
                controls.append(True)
            else:
                raise ValueError('Raw-reference isolation negative control failed')
        prediction, heights, receipt = raw_infer(pcd, out)
    finally:
        builtins.open, io.open, Path.open = original_builtin, original_io, original_path
    if not reads or len(controls) != 3:
        raise ValueError('Fresh PCD reads and all reference-negative controls required')
    result = score(piece, prediction, heights, args.name)
    receipt.update(input=str(pcd), input_sha256=sha256(pcd), guarded_pcd_reads=len(reads),
        negative_controls_passed=len(controls), runtime_step_used=False, runtime_truth_used=False,
        runtime_preprocessing_cache_used=False, source_sha256=sha256(__file__))
    (out / 'raw_verification.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf8')
    (out / 'joint_height_diagnostic.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    print('RAW_LOCAL_HEIGHT_JOINT', args.name, result['geometry']['valid_children'],
          result['height_counts'], flush=True)


if __name__ == '__main__':
    main()
