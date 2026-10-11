"""TRAIN augmentation views with local windows and one shared process height.

No original/DEV arrays or truth are changed. Translation preserves geometry;
unknown target masks are retained. Labels guide TRAIN augmentation only.
"""
import copy
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.curved_cache import load_piece, write_piece
from weld_pipeline.curved_sequences import decode_curved
from weld_pipeline.evaluation import score_welds
from weld_pipeline.partial_supervision import clean_targets
from weld_pipeline.train_candidate_tree import mapped_references


def main():
    parent_path = Path('weld_pipeline/data/contract_child_v1/split.json')
    original = json.loads(parent_path.read_text('utf8'))
    out = Path('weld_pipeline/data/focused_vertical_surface_v1')
    if (out / 'preprocessing_complete.json').exists():
        raise ValueError('Frozen focused-view dataset already exists')
    out.mkdir(parents=True, exist_ok=True)
    real = [n for n in original['train'] if '__' not in n]
    upper = []
    for name in real:
        piece = load_piece(original['piece_roots'][name], verify=True)
        for w in piece['truth']:
            if w['type'] == 'vertical':
                upper.append(round(max(p[2] for seg in w['segments'] for p in [seg['points'][0], seg['points'][-1]]), 3))
    anchor = Counter(upper).most_common(1)[0][0]
    split = copy.deepcopy(original)
    split.update(status='FROZEN_FOCUSED_VERTICAL_TRAIN_AUGMENTATION_V1',
                 parent_split_sha256=sha256(parent_path),
                 augmentation_view_policy='TRAIN only; crop positive augmented candidates to reference endpoints +/-20 mm; translate entire augmented geometry to shared upper height fitted to real TRAIN. Unknown masks preserved. Not independent truth.')
    rows = []
    for name in original['train']:
        if '__' not in name:
            continue
        piece = load_piece(original['piece_roots'][name], verify=True)
        refs = mapped_references(piece)
        maximum = max(p[2] for w in piece['truth'] for seg in w['segments'] for p in [seg['points'][0], seg['points'][-1]])
        delta = float(anchor - maximum)
        truth = copy.deepcopy(piece['truth'])
        for w in truth:
            for seg in w['segments']:
                pts = np.asarray(seg['points'], float)
                pts[:, 2] += delta
                seg['points'] = pts.tolist()
            # Old parent distances/heights and signed frames are deliberately
            # not reinterpreted as augmented attribute truth.
            w['augmented_attribute_truth_valid'] = False
        sequences, targets, details = [], [], []
        for i, (seq, target) in enumerate(zip(piece['sequences'], piece['targets'])):
            begin, stop = 0, len(seq['xyz'])
            if (target[:, 0] > .5).any():
                welds = refs.get(i, [])
                if len(welds) != 1:
                    raise ValueError('A focused augmentation candidate must have one parent')
                pts = np.array([p for s in welds[0]['segments'] for p in [s['points'][0], s['points'][-1]]])
                along = (pts - seq['xyz'][0]) @ seq['direction']
                step = seq['sample_step_mm']
                begin = max(0, int(np.floor((along.min() - 20.) / step)))
                stop = min(len(seq['xyz']), int(np.ceil((along.max() + 20.) / step)) + 1)
            feature = np.array(seq['features'][begin:stop], copy=True)
            xyz = np.array(seq['xyz'][begin:stop], copy=True)
            xyz[:, 2] += delta
            feature[:, 69] += delta / 10000.
            t = np.arange(stop - begin) * seq['sample_step_mm'] / 1000.
            feature[:, 55], feature[:, 56] = t, t[-1] - t
            changed = {**seq, 'features': feature, 'xyz': xyz}
            cropped = np.array(target[begin:stop], copy=True)
            if not np.array_equal(cropped, target[begin:stop]):
                raise ValueError('Target crop changed supervision')
            if not np.array_equal(feature[:, 0:49], seq['features'][begin:stop, 0:49]) or not np.array_equal(feature[:, 70:84], seq['features'][begin:stop, 70:84]):
                raise ValueError('Rigid translation changed local distance evidence')
            sequences.append(changed)
            targets.append(cropped)
            details.append({'sequence': i, 'original_points': len(seq['xyz']), 'view_points': stop - begin,
                            'unknown_retained': int(np.all(cropped == -1, axis=1).sum())})
        clean = [clean_targets(t)[0] for t in targets]
        score = score_welds(decode_curved(sequences, clean), truth)
        if score['valid_welds'] != len(truth) or score['false_or_duplicate_predictions']:
            raise ValueError('Focused augmentation no longer reconstructs known truth')
        record = {**piece['record'], 'augmentation': 'focused_window_shared_train_height',
                  'focused_upper_height_mm': anchor, 'translation_z_mm': delta,
                  'attribute_scope': 'Not augmented attribute truth', 'parent_manifest_sha256': piece['cache_manifest_sha256']}
        folder = out / name
        write_piece(folder, {'record': record, 'truth': truth, 'sequences': sequences, 'targets': targets},
                    {'source_manifest_sha256': piece['cache_manifest_sha256'], 'source_sha256': sha256(__file__),
                     'target_policy': piece['record'].get('target_policy', 'complete_known_original_rigid_view'),
                     'scope': split['augmentation_view_policy']})
        split['piece_roots'][name] = str(folder)
        rows.append({'name': name, 'translation_z_mm': delta, 'sequences': details, 'oracle_valid': score['valid_welds'],
                     'truth_count': len(truth), 'source_manifest_sha256': piece['cache_manifest_sha256'],
                     'view_manifest_sha256': sha256(folder / 'manifest.json')})
        print('FOCUSED', name, 'known_welds', len(truth), 'points', sum(r['original_points'] for r in details),
              '->', sum(r['view_points'] for r in details), flush=True)
    (out / 'split.json').write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf8')
    (out / 'augmentation_audit.json').write_text(json.dumps({'upper_height_mm': anchor, 'anchor_counts': dict(Counter(upper)),
        'rows': rows, 'original_train_and_dev_roots_unchanged': True, 'independent_truth': False}, ensure_ascii=False, indent=2), encoding='utf8')
    (out / 'preprocessing_complete.json').write_text(json.dumps({'status': 'READY_FOCUSED_MASKED_VIEWS',
        'split_sha256': sha256(out / 'split.json'), 'augmentation_count': len(rows), 'full_contract_pass': False}), encoding='utf8')
    shutil.copy2(__file__, out / Path(__file__).name)


if __name__ == '__main__':
    main()
