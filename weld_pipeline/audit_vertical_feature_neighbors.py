"""TRAIN-only feature neighbors for reference endpoints; diagnostic, not inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.labels import sha256
from weld_pipeline.axial_context_cache import load_piece
from weld_pipeline.axial_context_features import feature_window
from weld_pipeline.sequence_augmentation import change_frame
from weld_pipeline.vertical_frame_augmentation import swap_vertical_planes

CHANNELS = list(range(49)) + list(range(70, 84))


def patch(seq, index, reverse=False, swap=False, flip1=False, flip2=False):
    x = feature_window(seq, index - 15, index + 16).T.copy()
    valid = np.ones((1, 31), np.float32)
    y = np.zeros((7, 31), np.float32)
    # Only the central real sample is needed to validate the vertical frame.
    valid[:, :15] = 0
    valid[:, 16:] = 0
    if swap:
        swap_vertical_planes(x, valid)
    change_frame(x, y, valid, reverse=reverse, flip_normal=flip1, flip_vertical_cross=flip2)
    return x[CHANNELS].reshape(-1).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--query-group', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    root, out = Path(args.data), Path(args.outdir)
    split = json.loads((root / 'split.json').read_text(encoding='utf-8'))
    if args.query_group not in split['validation']:
        raise ValueError('Expected separately identified development diagnosis')
    vectors, metadata, hashes = [], [], {}
    for group in split['train']:
        folder = (root / split['piece_roots'][group]).resolve()
        hashes[group] = sha256(folder / 'manifest.json')
        piece = load_piece(folder)
        negative_candidates = []
        for si, (seq, targets) in enumerate(zip(piece['sequences'], piece['targets'])):
            if seq['type'] != 'vertical':
                continue
            known = np.all(targets >= 0, axis=1)
            for channel in [1, 2]:
                for index in np.flatnonzero(known & (targets[:, channel] > .65)):
                    vectors.append(patch(seq, int(index), reverse=channel == 1))
                    metadata.append({'group': group, 'sequence': si, 'index': int(index),
                                     'positive': True, 'source_channel': channel,
                                     'foreground': float(targets[index, 0])})
            indices = np.flatnonzero(known & (targets[:, 1:3].max(1) < .01))
            for index in indices[np.linspace(0, max(0, len(indices)-1), min(8, len(indices)), dtype=int)]:
                negative_candidates.append((si, int(index)))
        selected = np.linspace(0, max(0, len(negative_candidates)-1), min(128, len(negative_candidates)), dtype=int)
        for j in selected:
            si, index = negative_candidates[j]
            for reverse in [False, True]:
                vectors.append(patch(piece['sequences'][si], index, reverse=reverse))
                metadata.append({'group': group, 'sequence': si, 'index': index,
                                 'positive': False, 'source_channel': 1 if reverse else 2,
                                 'foreground': float(piece['targets'][si][index, 0])})
        print(json.dumps({'training_group': group, 'cumulative_vectors': len(vectors)}), flush=True)
    matrix = np.stack(vectors)
    positive = np.array([m['positive'] for m in metadata])
    squared = np.einsum('ij,ij->i', matrix, matrix)
    folder = (root / split['piece_roots'][args.query_group]).resolve()
    piece = load_piece(folder)
    mapping = piece['record']['mapping']['line_mapping']['weld_to_sequence']
    line_ids = [i for i, s in enumerate(piece['sequences']) if s.get('curve_kind') != 'arc']
    lines = [w for w in piece['truth'] if all(s['kind'] == 'line' for s in w['segments'])]
    rows = []
    for wi, weld in enumerate(lines):
        if weld['type'] != 'vertical' or str(wi) not in mapping:
            continue
        si = line_ids[mapping[str(wi)]]
        seq = piece['sequences'][si]
        xyz = np.asarray(seq['xyz'])
        ends = np.array([weld['segments'][0]['points'][0], weld['segments'][-1]['points'][-1]])
        ends = ends[np.argsort((ends - xyz[0]) @ np.asarray(seq['direction']))]
        for channel, point in zip([1, 2], ends):
            index = int(np.argmin(np.linalg.norm(xyz - point, axis=1)))
            queries = np.stack([patch(seq, index, reverse=channel == 1, swap=s, flip1=f1, flip2=f2)
                               for s in [False, True] for f1 in [False, True] for f2 in [False, True]])
            distance2 = np.maximum(0., np.einsum('ij,ij->i', queries, queries)[:, None] + squared[None] - 2 * queries @ matrix.T)
            distance2 = distance2.min(axis=0)
            result = {'truth_id': weld['id'], 'number': weld['metadata'].get('WeldNumber (f/v + Name number)'),
                      'channel': channel, 'sequence': si, 'index': index}
            for label, mask in [('positive', positive), ('negative', ~positive)]:
                indices = np.flatnonzero(mask)
                chosen = indices[np.argsort(distance2[indices])[:3]]
                result[label] = [{**metadata[int(j)], 'feature_rms_mm': float(np.sqrt(distance2[j] / matrix.shape[1]) * 50)} for j in chosen]
            rows.append(result)
    out.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'Reference-assisted comparison to sampled TRAIN patches only. Feature RMS is not endpoint error, prediction accuracy, or proof of incorrect labels.',
              'radius_samples': 15, 'distance_channels': CHANNELS, 'training_manifest_hashes': hashes,
              'query_manifest_sha256': sha256(folder / 'manifest.json'),
              'source_sha256': sha256(__file__), 'known_negative_sampling': 'At most 8 positions per sequence and 128 per training entry; both directions; all targets known; endpoint targets below .01.',
              'vectors': len(vectors), 'positive_vectors': int(positive.sum()), 'rows': rows}
    (out / 'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for channel in [1, 2]:
        selected = [r for r in rows if r['channel'] == channel]
        print(json.dumps({'channel': channel, 'queries': len(selected),
            'closer_to_negative': sum(r['negative'][0]['feature_rms_mm'] < r['positive'][0]['feature_rms_mm'] for r in selected),
            'median_positive_feature_rms_mm': float(np.median([r['positive'][0]['feature_rms_mm'] for r in selected])),
            'median_negative_feature_rms_mm': float(np.median([r['negative'][0]['feature_rms_mm'] for r in selected]))}), flush=True)


if __name__ == '__main__':
    main()
