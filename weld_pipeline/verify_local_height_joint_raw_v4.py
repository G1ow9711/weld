"""Raw-only finite-plane height interventions; references opened only for scoring."""
import argparse
import builtins
import copy
import io
import json
import shutil
import time
from pathlib import Path

import joblib
import torch

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.contract_horizontal_model import from_checkpoint
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates
from weld_pipeline.finite_joint_base_support import FiniteJointBases
from weld_pipeline.full_fit_hybrid import infer_from_sequences, point_structure
from weld_pipeline.local_height_datum_model_v4 import estimate_children
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline.verify_local_height_joint_raw_v3 import score


def raw_infer(pcd, out):
    started = time.perf_counter(); torch.set_num_threads(2)
    geometry_path = Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4/best.pt')
    height_path = Path('weld_pipeline/runs/local_height_datum_seed140/heads.joblib')
    state = torch.load(geometry_path, map_location='cpu')
    verify_cached_inference_sources(state, geometry_path)
    verify_cached_inference_sources(state['horizontal'], geometry_path)
    model = from_checkpoint(state['horizontal']).cuda().eval()
    height = joblib.load(height_path)
    xyz = read_xyz_pcd(pcd)*1000.
    sequences, geometry = make_mixed_sequences(xyz)
    structural, process = point_structure(xyz, sequences)
    predictions, _ = infer_from_sequences(state, model, sequences, structural, process)
    heights = estimate_children(height, FiniteContactPlates(xyz), FiniteJointBases(xyz), sequences, predictions)
    torch.cuda.synchronize()
    (out/'predictions.json').write_text(json.dumps(predictions, indent=2), encoding='utf8')
    (out/'height_candidates.json').write_text(json.dumps(heights, indent=2, ensure_ascii=False), encoding='utf8')
    return predictions, heights, dict(points=len(xyz), candidates=len(sequences), geometry=geometry,
        geometry_checkpoint_sha256=sha256(geometry_path), height_heads_sha256=sha256(height_path),
        elapsed_seconds_diagnostic_only=time.perf_counter()-started,
        model_combination='retained_v4_geometry_frozen140_head_new_finite_UV_contact_base_hypotheses',
        direct_physical_value_reported_independently=True, full_contract_pass=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--name', required=True)
    args = parser.parse_args()
    split = json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    if args.name not in [n for n in split['train'] if '__' not in n] + split['validation']:
        raise ValueError('Unregistered source role')
    piece = load_piece(split['piece_roots'][args.name], verify=True)
    pcd = Path(piece['record']['input']).resolve()
    out = Path('weld_pipeline/reports/finite_base_uv_20261001')/('raw_'+args.name)
    if out.exists():
        raise ValueError('Raw frozen diagnostic exists; preserve it')
    out.mkdir(parents=True)
    sources = [__file__, 'weld_pipeline/local_height_datum_model_v4.py',
               'weld_pipeline/finite_joint_base_support.py', 'weld_pipeline/finite_contact_plate_profiles.py',
               'weld_pipeline/local_base_sampling.py', 'weld_pipeline/verify_local_height_joint_raw_v3.py']
    config = dict(role='FIT' if args.name in split['train'] else 'OBSERVED_CHECK_NOT_BLIND',
        source_sha256={path:sha256(path) for path in sources}, input=str(pcd),
        input_sha256=sha256(pcd), geometric_tolerance_mm=3., parameter_selection_uses_check=False,
        physical_and_learned_outputs_separately_scored=True, full_contract_pass=False)
    (out/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot/Path(path).name)
    forbidden = [Path(root).resolve() for root in ('weld_pipeline/data','weld_pipeline/reports','DataSet')]
    original = (builtins.open, io.open, Path.open)
    reads, controls = [], []
    def guard(opener):
        def checked(path, mode='r', *positional, **kwargs):
            if isinstance(path, (str, bytes, Path)) and ('r' in str(mode) or '+' in str(mode)):
                resolved = Path(path).resolve()
                if resolved == pcd:
                    reads.append(str(resolved))
                elif (any(resolved.is_relative_to(root) for root in forbidden) or
                      resolved.suffix.lower() in ('.step','.stp','.xlsx','.xls','.cbf') or
                      resolved.name.endswith('_welds.json')):
                    raise RuntimeError('Forbidden inference reference/cache read: '+str(resolved))
            return opener(path, mode, *positional, **kwargs)
        return checked
    builtins.open, io.open, Path.open = tuple(guard(opener) for opener in original)
    try:
        for opener in (builtins.open, io.open, Path.open):
            try:
                opener(Path('weld_pipeline/data/negative_control.step'), 'r')
            except RuntimeError:
                controls.append(True)
            else:
                raise ValueError('Isolation negative control did not reject')
        predictions, heights, receipt = raw_infer(pcd, out)
    finally:
        builtins.open, io.open, Path.open = original
    if not reads or len(controls)!=3:
        raise ValueError('Fresh PCD and all three isolation controls required')
    learned = score(piece, predictions, heights, args.name)
    direct_heights = copy.deepcopy(heights)
    for result in direct_heights:
        for field in result['fields'].values():
            field['value'] = field.get('physical_height_mm')
            if field['value'] is None:
                field.update(state='unknown', reason=field.get('physical_diagnostic',{}).get('reason', field['reason']))
    direct = score(piece, predictions, direct_heights, args.name)
    receipt.update(config=config, guarded_fresh_PCD_reads=len(reads), negative_controls=len(controls),
        runtime_STEP_used=False, runtime_truth_used=False, runtime_preprocessing_cache_used=False,
        sources_unchanged=all(sha256(path)==digest for path,digest in config['source_sha256'].items()),
        input_unchanged=sha256(pcd)==config['input_sha256'])
    (out/'raw_verification.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf8')
    (out/'joint_height_diagnostic.json').write_text(json.dumps(learned, indent=2, ensure_ascii=False), encoding='utf8')
    (out/'direct_physical_height_diagnostic.json').write_text(json.dumps(direct, indent=2, ensure_ascii=False), encoding='utf8')
    print('RAW_FINITE_UV_HEIGHT', args.name, 'GEOMETRY', learned['geometry']['valid_children'],
          'LEARNED', learned['height_counts'], 'DIRECT', direct['height_counts'], flush=True)


if __name__ == '__main__':
    main()
