"""Inspect vertical endpoint evidence; never change inference or acceptance."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.model_registry_branch import model_from_checkpoint
from weld_pipeline.typed_contact_model import infer_probabilities


def main():
    ap = argparse.ArgumentParser()
    for name in ['data', 'group', 'checkpoint', 'outdir']:
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    root, out = Path(args.data), Path(args.outdir)
    split = json.loads((root / 'split.json').read_text(encoding='utf-8'))
    if args.group not in split['train'] + split['validation']:
        raise ValueError('Group outside checkpoint split')
    with Path(args.checkpoint).open('rb') as stream:
        state = torch.load(stream, map_location='cpu')
    verify_cached_inference_sources(state, args.checkpoint)
    folder = (root / split.get('piece_roots', {}).get(args.group, args.group)).resolve()
    if sha256(folder / 'manifest.json') != state['data_hashes'][args.group]:
        raise ValueError('Cache manifest changed')
    piece = load_piece(folder)
    mapping = piece['record']['mapping']['line_mapping']['weld_to_sequence']
    line_ids = [i for i, seq in enumerate(piece['sequences']) if seq.get('curve_kind') != 'arc']
    lines = [w for w in piece['truth'] if all(s['kind'] == 'line' for s in w['segments'])]
    torch.set_num_threads(4)
    model = model_from_checkpoint(state).cuda().eval()
    probabilities = infer_probabilities(model, piece['sequences'])
    rows = []
    for wi, weld in enumerate(lines):
        if weld['type'] != 'vertical':
            continue
        mapped = mapping.get(str(wi))
        if mapped is None:
            rows.append({'truth_id': weld['id'], 'candidate_missing': True})
            continue
        si = line_ids[mapped]
        seq, prob = piece['sequences'][si], probabilities[si]
        xyz = np.asarray(seq['xyz'])
        ends = np.asarray([weld['segments'][0]['points'][0], weld['segments'][-1]['points'][-1]])
        positions = (ends - xyz[0]) @ np.asarray(seq['direction'])
        ends = ends[np.argsort(positions)]
        heads = []
        for channel, point in zip([1, 2], ends):
            distance = np.linalg.norm(xyz - point, axis=1)
            nearby = distance <= 3.
            heads.append({'channel': channel, 'reference_xyz_mm': point.tolist(),
                          'nearest_sample_distance_mm': float(distance.min()),
                          'max_probability_within_3mm': float(prob[nearby, channel].max()) if nearby.any() else None})
        projected = (xyz - xyz[0]) @ np.asarray(seq['direction'])
        lo, hi = sorted(positions)
        inside = (projected >= lo) & (projected <= hi)
        rows.append({'truth_id': weld['id'], 'number': weld['metadata'].get('WeldNumber (f/v + Name number)'),
                     'sequence': si, 'reference_segment_count': len(weld['segments']), 'heads': heads,
                     'foreground_mean_inside_truth': float(prob[inside, 0].mean()) if inside.any() else None,
                     'internal_break_max_inside_truth': float(prob[inside, 3].max()) if inside.any() else None,
                     'work_range_any_mean_inside_truth': float(prob[inside, 5].mean()) if inside.any() else None,
                     'side_split_any_mean_inside_truth': float(prob[inside, 6].mean()) if inside.any() else None})
    report = {'scope': 'Reference-assisted diagnosis only; no prediction modification or acceptance claim.',
              'role': 'development' if args.group in split['validation'] else 'training',
              'group': args.group, 'checkpoint_sha256': sha256(args.checkpoint),
              'cache_sha256': sha256(folder / 'manifest.json'), 'source_sha256': sha256(__file__), 'rows': rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'endpoint_heads.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    missing = sum(r.get('candidate_missing', False) for r in rows)
    start_ready = sum((r.get('heads', [{}])[0].get('max_probability_within_3mm') or 0.) >= .5 for r in rows)
    end_ready = sum((r['heads'][1]['max_probability_within_3mm'] or 0.) >= .5 for r in rows if 'heads' in r)
    print(json.dumps({'group': args.group, 'role': report['role'], 'truth_verticals': len(rows),
                      'candidate_missing': missing, 'start_peak_ge_05': start_ready,
                      'end_peak_ge_05': end_ready, 'diagnostic_only': True}), flush=True)


if __name__ == '__main__':
    main()
