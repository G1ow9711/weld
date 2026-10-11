"""Reference-assisted gradient locality diagnosis, never endpoint prediction."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from weld_pipeline.axial_context_cache import load_piece
from weld_pipeline.axial_context_features import feature_window
from weld_pipeline.model_registry_axial_branch import model_from_checkpoint
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources


def main():
    ap = argparse.ArgumentParser()
    for name in ['data', 'checkpoint', 'outdir']:
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--groups', nargs='+', required=True)
    args = ap.parse_args()
    root, out = Path(args.data), Path(args.outdir)
    split = json.loads((root / 'split.json').read_text(encoding='utf-8'))
    with Path(args.checkpoint).open('rb') as stream:
        state = torch.load(stream, map_location='cpu')
    verify_cached_inference_sources(state, args.checkpoint)
    model = model_from_checkpoint(state).cuda().eval().requires_grad_(False)
    torch.set_num_threads(4)
    records = []
    for group in args.groups:
        if group not in split['train'] + split['validation']:
            raise ValueError('Group outside recorded split')
        folder = (root / split['piece_roots'][group]).resolve()
        if sha256(folder / 'manifest.json') != state['data_hashes'][group]:
            raise ValueError('Cache changed')
        piece = load_piece(folder)
        mapping = piece['record']['mapping']['line_mapping']['weld_to_sequence']
        line_ids = [i for i, s in enumerate(piece['sequences']) if s.get('curve_kind') != 'arc']
        lines = [w for w in piece['truth'] if all(s['kind'] == 'line' for s in w['segments'])]
        for wi, weld in enumerate(lines):
            if weld['type'] != 'vertical' or str(wi) not in mapping:
                continue
            seq = piece['sequences'][line_ids[mapping[str(wi)]]]
            xyz = np.asarray(seq['xyz'])
            ends = np.array([weld['segments'][0]['points'][0], weld['segments'][-1]['points'][-1]])
            ends = ends[np.argsort((ends - xyz[0]) @ np.asarray(seq['direction']))]
            for channel, point in zip([1, 2], ends):
                index = int(np.argmin(np.linalg.norm(xyz - point, axis=1)))
                # One full receptive field. No other samples affect center.
                f = feature_window(seq, index - 256, index + 257)
                x = torch.from_numpy(f.T.copy()[None]).cuda().requires_grad_(True)
                logit = model(x)[0, channel, 256]
                grad = torch.autograd.grad(logit, x)[0][0].detach().cpu().numpy()
                mass = np.abs(grad).sum(axis=0)
                mass_times_input = np.abs(grad * f.T).sum(axis=0)
                distances = np.abs(np.arange(513) - 256) * seq['sample_step_mm']
                row = {'group': group, 'truth_id': weld['id'], 'channel': channel,
                       'probability_at_nearest_sample': float(logit.detach().sigmoid()),
                       'gradient_l1': float(mass.sum()),
                       'gradient_input_l1': float(mass_times_input.sum())}
                for radius in [30, 60, 120, 256]:
                    row[f'gradient_outside_{radius}mm_fraction'] = float(mass[distances > radius].sum() / max(mass.sum(), 1e-20))
                    row[f'gradient_input_outside_{radius}mm_fraction'] = float(mass_times_input[distances > radius].sum() / max(mass_times_input.sum(), 1e-20))
                records.append(row)
        for channel in [1, 2]:
            rows = [r for r in records if r['group'] == group and r['channel'] == channel]
            print(json.dumps({'group': group, 'channel': channel, 'count': len(rows),
                'median_gradient_outside_30mm': float(np.median([r['gradient_outside_30mm_fraction'] for r in rows])),
                'median_gradient_input_outside_30mm': float(np.median([r['gradient_input_outside_30mm_fraction'] for r in rows]))}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'Reference-assisted local sensitivity; gradients do not prove causal error or improved accuracy.',
              'checkpoint_epoch': state['epoch'], 'checkpoint_sha256': sha256(args.checkpoint),
              'source_sha256': sha256(__file__), 'rows': records}
    (out / 'receipt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
