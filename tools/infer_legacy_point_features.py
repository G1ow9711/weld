"""Run preserved run162 weights on rebuilt 905 blocks, without retraining.

Outputs follow the run165 block feature schema. They are new reconstruction
artifacts, not historical equivalence certificates or weld predictions.
"""
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
from weld_pipeline.server_legacy_runtime import install

ADAPTER = install()
from weld_pipeline.sjg_local_runtime_904 import check_resources, digest, write

ROOT = Path(__file__).resolve().parents[1]
RESTORE = ROOT / 'server_rebuild/best_baseline_restore'
BLOCKS = RESTORE / 'input_blocks'
MODEL_DIR = ROOT / 'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776/run162'


def validate_output(result, ids, count):
    n = len(ids)
    shapes = dict(original_point_ids=(count,), embedding=(count, 16), normal=(count, 3),
                  boundary=(count,), material_embedding=(count, 16),
                  context_original_point_ids=(n-count,), context_embedding=(n-count, 16),
                  context_material_embedding=(n-count, 16))
    assert set(result) == set(shapes)
    for key, shape in shapes.items():
        assert result[key].shape == shape and np.isfinite(result[key]).all(), key
    assert np.array_equal(result['original_point_ids'], ids[:count])
    assert np.array_equal(result['context_original_point_ids'], ids[count:])
    assert np.all((result['boundary'] >= 0) & (result['boundary'] <= 1))


def run(key, limit):
    checkpoint = MODEL_DIR / 'last.pt'
    original_receipt = json.loads((MODEL_DIR / 'receipt.json').read_text())
    assert original_receipt['status'] == 'NATIVE_PTV3_STRUCTURAL_TRAINING_COMPLETE'
    checkpoint_sha = digest(checkpoint)
    assert checkpoint_sha == original_receipt['checkpoint_sha256']
    case = json.loads((BLOCKS / ('source_' + key) / 'manifest.json').read_text())
    assert digest(ROOT / case['input_file']) == case['input_sha256']
    import torch
    from weld_pipeline.sjg_material_model_859 import SurfaceMaterialModel
    from weld_pipeline.sjg_ptv3_801 import voxelize
    torch.set_num_threads(2)
    # Original checkpoint is authenticated against its transferred receipt;
    # it may contain optimizer/RNG metadata unsupported by weights_only mode.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SurfaceMaterialModel()
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.cuda().eval()
    dest = RESTORE / ('point_features_smoke' if limit else 'point_features') / ('source_' + key)
    dest.mkdir(parents=True, exist_ok=False)
    write(dest / 'protocol.json', dict(checkpoint_sha256=checkpoint_sha,
          checkpoint_receipt_sha256=digest(MODEL_DIR / 'receipt.json'), adapter=ADAPTER,
          source_sha256=digest(__file__), input_sha256=case['input_sha256'],
          model_source_sha256=digest(ROOT / 'weld_pipeline/sjg_material_model_859.py'),
          block_manifest_sha256=digest(BLOCKS / ('source_' + key) / 'manifest.json'),
          limit_blocks=limit, optimizer_updates=0, GT_reads=0, STEP_reads=0,
          historical_feature_parity_proven=False, torch_version=torch.__version__))
    entries = case['blocks'][:limit] if limit else case['blocks']
    seen = np.zeros(case['original_points'], np.uint8)
    rows = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for bi, entry in enumerate(entries):
                check_resources()
                path = BLOCKS / entry['file']
                assert digest(path) == entry['sha256']
                with np.load(path) as z:
                    xyz, ids, count = z['xyz_mm'], z['original_point_ids'], int(z['core_count'])
                assert count == entry['core_count'] and len(ids) == len(xyz)
                assert np.all((ids >= 0) & (ids < len(seen)))
                embedding, normal, boundary, material = model(voxelize(xyz))
                result = dict(original_point_ids=ids[:count],
                    embedding=embedding[:count].cpu().numpy(), normal=normal[:count].cpu().numpy(),
                    boundary=boundary[:count].sigmoid().cpu().numpy(),
                    material_embedding=material[:count].cpu().numpy(),
                    context_original_point_ids=ids[count:], context_embedding=embedding[count:].cpu().numpy(),
                    context_material_embedding=material[count:].cpu().numpy())
                validate_output(result, ids, count)
                np.add.at(seen, ids[:count], 1)
                assert np.all(seen <= 1)
                file = dest / f'features_{bi:04d}.npz'
                np.savez(file, **result)
                rows.append(dict(file=file.name, sha256=digest(file), core_points=count,
                                 input_block_sha256=entry['sha256']))
                del embedding, normal, boundary, material, result
                if bi % 20 == 0:
                    print(key, 'FEATURE_BLOCK', bi + 1, '/', len(entries), flush=True)
        complete = len(entries) == len(case['blocks'])
        if complete:
            assert np.all(seen == 1)
        torch.cuda.synchronize()
        write(dest / 'feature_manifest.json', dict(source_key=key, role=case['role'],
              input_sha256=case['input_sha256'], original_points=len(seen), blocks=rows,
              core_coverage_complete=complete, original_core_coverage_once=complete,
              status='POINT_FEATURES_COMPLETE' if complete else 'SMOKE_ONLY_INCOMPLETE_CASE',
              seconds=time.perf_counter()-started, historical_parity_proven=False,
              weld_accuracy_measured=False))
        print(key, 'COMPLETE' if complete else 'SMOKE_ONLY', len(rows), flush=True)
    except BaseException:
        write(dest / 'failure.json', dict(error=traceback.format_exc(), completed_blocks=len(rows)))
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', choices=['21', 'BM79A_native_local', '10', '25', '36'], required=True)
    ap.add_argument('--limit-blocks', type=int, default=0)
    args = ap.parse_args()
    assert args.limit_blocks >= 0
    run(args.key, args.limit_blocks)
