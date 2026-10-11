"""Add audited partial-supervision variants without changing a frozen split."""
import argparse
import json
import os
import shutil
from pathlib import Path
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece, write_piece
from weld_pipeline.partial_supervision import clean_targets
from weld_pipeline.curved_sequences import decode_curved
from weld_pipeline.evaluation import score_welds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--previous', default='weld_pipeline/data/partial_augmented_surface_v1')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--audits', nargs='+', required=True)
    a = ap.parse_args()
    previous, out = Path(a.previous), Path(a.outdir)
    if out.exists():
        raise ValueError('Use a fresh dataset directory')
    split = json.loads((previous / 'split.json').read_text(encoding='utf-8'))
    if json.loads((previous / 'preprocessing_complete.json').read_text())['status'] != 'READY_PARTIAL':
        raise ValueError('Expected mask-aware parent dataset')
    original = set(split['augmentation_policy']['original_train'])
    pending = []
    for folder in map(Path, a.audits):
        audit_path = folder / 'receipt.json'
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if audit['status'] != 'MASKED_REPRESENTATION_READY':
            raise ValueError('Audit is not representation-ready')
        parent = audit['source_group']
        if parent not in original or parent in split['validation']:
            raise ValueError('Augmentation parent must be original TRAIN')
        parent_path = (previous / split['piece_roots'][parent]).resolve()
        if sha256(parent_path / 'manifest.json') != audit['source_manifest_sha256']:
            raise ValueError('Original TRAIN manifest changed')
        for source, wanted in audit['source_hashes'].items():
            if sha256(source) != wanted:
                raise ValueError('Audit source changed: ' + source)
        piece = load_piece(folder / 'candidate_cache', verify=True)
        if sha256(piece['record']['input']) != audit['source_pcd_sha256']:
            raise ValueError('Original point cloud changed')
        if piece['record'].get('target_policy') != 'all_channel_minus_one_is_unknown':
            raise ValueError('Mask encoding missing')
        targets, known, unknown = [], 0, 0
        for target in piece['targets']:
            clean, mask = clean_targets(target)
            targets.append(clean)
            known += int(mask.sum())
            unknown += int((mask == 0).sum())
        if (known, unknown) != (audit['supervised_points'], audit['unknown_points']):
            raise ValueError('Masked counts differ')
        score = score_welds(decode_curved(piece['sequences'], targets), piece['truth'])
        if score['valid_welds'] != len(piece['truth']) or score['false_or_duplicate_predictions']:
            raise ValueError('Known target representation does not reconstruct exactly')
        name = piece['record']['name']
        if name in split['train'] or any(p[0] == name for p in pending):
            raise ValueError('Duplicate augmentation')
        verification = {'status': 'PASS', 'scope': 'Masked target representation only, not model accuracy.',
                        'audit_sha256': sha256(audit_path), 'manifest_sha256': sha256(folder / 'candidate_cache/manifest.json'),
                        'known_points': known, 'unknown_points': unknown, 'oracle_score': score}
        pending.append((name, piece, folder, audit, verification))
    out.mkdir(parents=True)
    split['piece_roots'] = {n: os.path.relpath((previous / p).resolve(), out.resolve())
                            for n, p in split['piece_roots'].items()}
    for name, piece, folder, audit, verification in pending:
        write_piece(out / name, piece, {'audit_sha256': verification['audit_sha256'],
                    'required_trainer_capability': 'partial_supervision_v1',
                    'source_group': audit['source_group'], 'source_manifest_sha256': audit['source_manifest_sha256'],
                    'target_policy': 'all_channel_minus_one_is_unknown'})
        split['train'].append(name)
        split['piece_roots'][name] = name
        split['augmentation_policy']['variants'].append({'name': name, 'source_group': audit['source_group'],
              'welds': len(piece['truth']), 'partial_supervision': True,
              'known_points': verification['known_points'], 'unknown_points': verification['unknown_points'],
              'manifest_sha256': sha256(out / name / 'manifest.json')})
        evidence = out / 'evidence' / name
        evidence.mkdir(parents=True)
        shutil.copy2(folder / 'receipt.json', evidence / 'receipt.json')
        (evidence / 'cache_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
        for source in audit['source_hashes']:
            shutil.copy2(source, evidence / Path(source).name)
    split['augmentation_policy']['prior_split_sha256'] = sha256(previous / 'split.json')
    (out / 'split.json').write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf-8')
    ready = {'status': 'READY_PARTIAL', 'required_trainer_capability': 'partial_supervision_v1',
             'training_entries': len(split['train']), 'independent_training_groups': len(original),
             'augmentation_entries': len(split['augmentation_policy']['variants']),
             'validation_groups': split['validation'], 'split_sha256': sha256(out / 'split.json'),
             'source_sha256': sha256(Path(__file__)), 'added_variants': [p[0] for p in pending]}
    (out / 'preprocessing_complete.json').write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(__file__, out / Path(__file__).name)
    print(json.dumps(ready, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
