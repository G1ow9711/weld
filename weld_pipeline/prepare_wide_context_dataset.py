"""Prepare point-only wide context without changing the existing split or labels."""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path
import numpy as np
from training.labels import sha256
from training.dataio import read_xyz_pcd
from weld_pipeline.curved_cache import load_piece as load_base
from weld_pipeline.wide_vertical_features import make_vertical_context, FEATURE_VERSION
from weld_pipeline.wide_context_cache import write_piece, load_piece, STORAGE_FORMAT


def augmentation_transform(name, variant, base_manifest):
    if variant is None:
        return np.eye(3), np.zeros(3), None
    provenance = base_manifest['provenance']
    partial = variant.get('partial_supervision', False)
    wanted = provenance['audit_sha256' if partial else 'augmentation_receipt_sha256']
    if partial:
        candidates = [Path('weld_pipeline/reports/partial_vertical_augmentation') / name.removesuffix('__partial_vertical') / 'receipt.json',
                      Path('weld_pipeline/data/partial_augmented_surface_v1/partial_evidence/receipt.json')]
        candidates += [Path('weld_pipeline/data') / version / 'evidence' / name / 'receipt.json'
                       for version in ['partial_augmented_surface_v2', 'partial_augmented_surface_v3']]
    else:
        candidates = [Path(root) / variant['source_group'] / 'receipt.json' for root in
                      ['weld_pipeline/reports/rigid_vertical_augmentation', 'weld_pipeline/data/rigid_augmented_surface_v1/augmentation_evidence']]
    matching = [p for p in candidates if p.exists() and sha256(p) == wanted]
    if not matching:
        raise ValueError('No hash-matched augmentation transform: ' + name)
    path = matching[0]
    audit = json.loads(path.read_text(encoding='utf-8'))
    expected_status = 'MASKED_REPRESENTATION_READY' if partial else 'REPRESENTATION_READY'
    if audit['status'] != expected_status or audit['source_group'] != variant['source_group']:
        raise ValueError('Unaudited augmentation')
    if audit['source_manifest_sha256'] != provenance['source_manifest_sha256'] or audit['source_pcd_sha256'] != base_manifest['record']['pcd_sha256']:
        raise ValueError('Augmentation source mismatch')
    rotation = np.asarray(audit['rotation'], float)
    translation = np.asarray(audit['translation_mm'], float)
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError('Invalid rigid transform')
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-8):
        raise ValueError('Non-rigid augmentation')
    if 'rotation' in provenance and (not np.array_equal(rotation, provenance['rotation']) or not np.array_equal(translation, provenance['translation_mm'])):
        raise ValueError('Manifest and audit transformations differ')
    return rotation, translation, {'path': str(path), 'sha256': wanted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    root, out = Path(args.parent), Path(args.outdir)
    split = json.loads((root / 'split.json').read_text(encoding='utf-8'))
    sources = [Path(__file__), *[Path('weld_pipeline') / n for n in
               ['wide_vertical_features.py', 'wide_context_cache.py', 'wide_vertical_augmentation.py', 'curved_cache.py', 'mapped_cache.py']]]
    source_hashes = {str(p): sha256(p) for p in sources}
    spec = {'parent_split_sha256': sha256(root / 'split.json'), 'feature_version': FEATURE_VERSION, 'sources': source_hashes}
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / 'build_spec.json'
    if spec_path.exists() and json.loads(spec_path.read_text(encoding='utf-8')) != spec:
        raise ValueError('Existing build specification differs')
    spec_path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
    variants = {v['name']: v for v in split['augmentation_policy']['variants']}
    groups = split['train'] + split['validation']
    results = []
    for name in groups:
        start = time.perf_counter()
        folder = (root / split.get('piece_roots', {}).get(name, name)).resolve()
        base_manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if name in variants:
            variant = variants[name]
            if name not in split['train'] or variant['source_group'] not in split['augmentation_policy']['original_train']:
                raise ValueError('Augmentation must belong to original TRAIN')
            if sha256(folder / 'manifest.json') != variant['manifest_sha256']:
                raise ValueError('Augmentation cache changed')
        else:
            variant = None
        rotation, translation, audit = augmentation_transform(name, variant, base_manifest)
        source = Path(base_manifest['record']['input'])
        if sha256(source) != base_manifest['record']['pcd_sha256']:
            raise ValueError('Raw input changed')
        provenance = {'parent_manifest_sha256': sha256(folder / 'manifest.json'), 'sources': source_hashes,
                      'raw_input': str(source), 'raw_input_sha256': sha256(source), 'augmentation_audit': audit,
                      'rotation': rotation.tolist(), 'translation_mm': translation.tolist()}
        dest = out / name
        if (dest / 'manifest.json').exists():
            manifest = json.loads((dest / 'manifest.json').read_text(encoding='utf-8'))
            if manifest['provenance'] != provenance:
                raise ValueError('Resume provenance differs')
            piece = load_piece(dest)
        else:
            base = load_base(folder)
            xyz = read_xyz_pcd(source) * 1000.
            xyz = xyz @ rotation.T + translation
            extras = make_vertical_context(xyz, base['sequences'])
            manifest = write_piece(dest, folder, extras, provenance)
            piece = load_piece(dest)
            for old, new, target in zip(base['sequences'], piece['sequences'], piece['targets']):
                if not np.array_equal(old['features'], new['features'][:, :84]) or not np.array_equal(old['xyz'], new['xyz']):
                    raise ValueError('Base geometry or features changed')
                if target.flags.writeable:
                    raise ValueError('Targets must remain read-only')
            if base['truth'] != piece['truth'] or any(not np.array_equal(a, b) for a, b in zip(base['targets'], piece['targets'])):
                raise ValueError('Labels changed')
            del base, xyz, extras
        row = {'group': name, 'vertical_points': manifest['context_shape'][0],
               'manifest_sha256': sha256(dest / 'manifest.json'), 'context_bytes': (dest / 'vertical_context.npy').stat().st_size,
               'elapsed_seconds': time.perf_counter() - start}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del piece
    new_split = copy.deepcopy(split)
    new_split.update(feature_version=FEATURE_VERSION, storage_format=STORAGE_FORMAT,
                     piece_roots={n: n for n in groups}, parent_split_sha256=spec['parent_split_sha256'])
    for variant in new_split['augmentation_policy']['variants']:
        variant['manifest_sha256'] = sha256(out / variant['name'] / 'manifest.json')
    (out / 'split.json').write_text(json.dumps(new_split, ensure_ascii=False, indent=2), encoding='utf-8')
    ready = {'status': 'READY_PARTIAL', 'required_trainer_capability': 'partial_supervision_v1',
             'required_feature_loader': STORAGE_FORMAT, 'split_sha256': sha256(out / 'split.json'),
             'parent_split_sha256': spec['parent_split_sha256'], 'source_hashes': source_hashes,
             'base_features_geometry_labels_unchanged': True, 'pieces': results,
             'scope': 'Feature preparation only; no new independent workpieces and no accuracy claim.'}
    (out / 'preprocessing_complete.json').write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding='utf-8')
    snapshot = out / 'source_snapshot'; snapshot.mkdir(exist_ok=True)
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
    print('READY', len(results), 'CONTEXT_BYTES', sum(r['context_bytes'] for r in results), flush=True)


if __name__ == '__main__':
    main()
