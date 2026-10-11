"""Ablate only the added channels in a fixed checkpoint; never alter predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.model_registry_wide_branch import model_from_checkpoint
from weld_pipeline.wide_context_cache import load_piece
from weld_pipeline.wide_vertical_features import ContextFeatures
from weld_pipeline.typed_contact_model import infer_probabilities


def main():
    ap = argparse.ArgumentParser()
    for key in ['data', 'checkpoint', 'outdir']:
        ap.add_argument('--'+key, required=True)
    ap.add_argument('--groups', nargs='+', required=True)
    args = ap.parse_args()
    with Path(args.checkpoint).open('rb') as f:
        state = torch.load(f, map_location='cpu')
    verify_cached_inference_sources(state, args.checkpoint)
    model = model_from_checkpoint(state).cuda().eval()
    torch.set_num_threads(4)
    root = Path(args.data)
    split = json.loads((root/'split.json').read_text(encoding='utf-8'))
    rows = []
    for name in args.groups:
        if name not in split['train']+split['validation']:
            raise ValueError('Group outside declared split')
        folder = (root/split.get('piece_roots', {}).get(name, name)).resolve()
        if sha256(folder/'manifest.json') != state['data_hashes'][name]:
            raise ValueError('Manifest changed')
        piece = load_piece(folder)
        sequences = [s for s in piece['sequences'] if s['type']=='vertical']
        original = infer_probabilities(model, sequences)
        zeros = [{**s, 'features': ContextFeatures(s['features'].base)} for s in sequences]
        ablated = infer_probabilities(model, zeros)
        delta = np.concatenate([np.abs(a-b) for a,b in zip(original, ablated)])
        row = {'group': name, 'role': 'development' if name in split['validation'] else 'training',
               'points': len(delta), 'mean_probability_change_per_head': delta.mean(0).tolist(),
               'max_probability_change_per_head': delta.max(0).tolist(),
               'manifest_sha256': sha256(folder/'manifest.json')}
        rows.append(row)
        print(json.dumps(row), flush=True)
    weights = state['model']['vertical.net.0.weight'][:,84:165]
    result = {'checkpoint_sha256': sha256(args.checkpoint), 'checkpoint_epoch': state['epoch'],
              'added_weight_abs_max': float(weights.abs().max()), 'added_weight_l2': float(weights.norm()),
              'groups': rows, 'source_sha256': sha256(__file__),
              'scope': 'Diagnostic sensitivity only. A changed probability is not improved accuracy.'}
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'response.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
