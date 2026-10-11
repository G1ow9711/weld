"""Freeze point-derived structural features; STEP is never opened."""
import json
import time
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.point_plate_context import PlateContext, VERSION
from weld_pipeline.prepare_wide_context_dataset import augmentation_transform


def main():
    parent = Path('weld_pipeline/data/contract_child_v1/split.json')
    focused = Path('weld_pipeline/data/focused_vertical_surface_v1/split.json')
    original, split = [json.loads(p.read_text('utf8')) for p in [parent, focused]]
    variants = {r['name']: r for r in original['augmentation_policy']['variants']}
    root = Path('weld_pipeline/data/point_plate_context_v1')
    root.mkdir(parents=True, exist_ok=True)
    if (root/'complete.json').exists():
        raise ValueError('Frozen dataset exists')
    sources = {p: sha256(p) for p in [__file__, 'weld_pipeline/point_plate_context.py',
                                    'weld_pipeline/prepare_wide_context_dataset.py']}
    records = []
    for name in split['train'] + split['validation']:
        started = time.perf_counter()
        folder = Path(original['piece_roots'][name])
        manifest = json.loads((folder/'manifest.json').read_text('utf8'))
        rotation, translation, receipt = augmentation_transform(name, variants.get(name), manifest)
        piece = load_piece(split['piece_roots'][name], verify=True)
        source = Path(piece['record']['input'])
        if sha256(source) != piece['record']['pcd_sha256']:
            raise ValueError('Raw source changed')
        xyz = read_xyz_pcd(source)*1000. @ rotation.T + translation
        if '__' in name:
            xyz[:, 2] += piece['record']['translation_z_mm']
        context = PlateContext(xyz)
        structural, process, diagnostic = [], [], []
        for seq in piece['sequences']:
            a, b, info = context.describe(seq)
            structural.append(a)
            process.append(b)
            diagnostic.append(info)
        destination = root/(name+'.npz')
        np.savez_compressed(destination, structural=np.stack(structural), process=np.stack(process))
        record = {'name': name, 'file_sha256': sha256(destination), 'raw_sha256': sha256(source),
                  'parent_manifest_sha256': sha256(Path(split['piece_roots'][name])/'manifest.json'),
                  'augmentation_receipt': receipt, 'sequences': len(structural),
                  'seconds': time.perf_counter()-started, 'diagnostic': diagnostic}
        (root/(name+'.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
        records.append({k:v for k,v in record.items() if k!='diagnostic'})
        print('POINT_CONTEXT', name, len(structural), round(record['seconds'], 2), flush=True)
        del xyz, context, piece
    (root/'split.json').write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf8')
    (root/'complete.json').write_text(json.dumps({'version': VERSION, 'sources': sources,
        'parent_split_sha256': sha256(focused), 'point_only': True, 'pieces': records,
        'references_used_for_features': False}, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
