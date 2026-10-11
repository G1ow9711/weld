"""Add pure-PCD context to unchanged 19-source material samples and labels."""
import json
import shutil
import time
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.workpiece_relative_context import WorkpieceRelativeContext, VERSION, PARAMETERS

BASE = Path('weld_pipeline/data/material_support_v1')
OUT = Path('weld_pipeline/data/workpiece_context_v1')
PROOF = Path('weld_pipeline/reports/workpiece_context_physical_153v2_20261001.json')


def main():
    if OUT.exists():
        raise ValueError('Preserve relative-context data')
    proof = json.loads(PROOF.read_text('utf8'))
    if not proof['all_passed']:
        raise ValueError('Physical context proof required')
    for path, digest in proof['sources'].items():
        if sha256(path) != digest:
            raise ValueError('Physical context proof source changed')
    cfg = json.loads((BASE/'config.json').read_text('utf8'))
    complete = json.loads((BASE/'complete.json').read_text('utf8'))
    if sha256(BASE/'config.json') != complete['config_sha256']:
        raise ValueError('Base fixed data configuration changed')
    split = json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    sources = [__file__, 'weld_pipeline/workpiece_relative_context.py', 'weld_pipeline/curved_cache.py',
               'training/dataio.py', 'weld_pipeline/signed_material_support.py']
    OUT.mkdir(parents=True)
    config = dict(version=VERSION, parameters=PARAMETERS, pieces=cfg['pieces'], folds=cfg['folds'],
        base_data_config_sha256=sha256(BASE/'config.json'), base_complete_sha256=sha256(BASE/'complete.json'),
        physical_proof_sha256=sha256(PROOF), feature_count=22,
        sources={p:sha256(p) for p in sources}, sampling='Unchanged original fixed five fractions',
        runtime_inputs='Finite raw PCD XYZ in mm only; global +Z is fixed gravity axis',
        whole_height_is_not_contract_plate_height=True, observed_check_read=False, full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    snapshot = OUT/'source_snapshot'; snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot/Path(source).name)
    manifests = {}; started = time.perf_counter()
    for name in cfg['pieces']:
        base = json.loads((BASE/name/'manifest.json').read_text('utf8'))
        if sha256(BASE/name/'manifest.json') != complete['manifests'][name]:
            raise ValueError('Original fixed sample manifest changed')
        for file, digest in base['files'].items():
            if sha256(BASE/name/file) != digest:
                raise ValueError('Original fixed features or labels changed')
        piece = load_piece(split['piece_roots'][name], verify=True)
        pcd = Path(piece['record']['input'])
        if piece['cache_manifest_sha256'] != base['cache_manifest_sha256'] or sha256(pcd) != base['pcd_sha256']:
            raise ValueError('Relative context input/candidate source changed')
        context = WorkpieceRelativeContext(read_xyz_pcd(pcd)*1000.)
        result = []
        # Query order remains the original fixed material sample order, never truth-driven.
        for row in base['entries']:
            sequence = piece['sequences'][row['sequence']]
            result.append(context.describe_sequence(sequence, [row['sample_index']])[0])
        result = np.asarray(result, np.float32)
        root = OUT/name; root.mkdir()
        np.save(root/'context.npy', result)
        manifest = dict(piece=name, rows=len(result), base_manifest_sha256=sha256(BASE/name/'manifest.json'),
            input=str(pcd), input_sha256=sha256(pcd), source_cache_manifest_sha256=piece['cache_manifest_sha256'],
            files={'context.npy':sha256(root/'context.npy')}, reference_order_unchanged=True,
            base_signed_features_and_labels_not_modified=True)
        (root/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf8')
        manifests[name] = sha256(root/'manifest.json')
        print('WORKPIECE_CONTEXT_DATA', name, len(result), round(time.perf_counter()-started, 2), flush=True)
        del context, piece, result
    (OUT/'complete.json').write_text(json.dumps(dict(config_sha256=sha256(OUT/'config.json'),
        manifests=manifests, elapsed_seconds=time.perf_counter()-started,
        original_labels_changed=False, observed_check_read=False, full_contract_pass=False),
        indent=2, ensure_ascii=False), encoding='utf8')
    print('WORKPIECE_CONTEXT_DATA_COMPLETE', len(manifests), flush=True)


if __name__ == '__main__':
    main()
