"""Diagnose irrelevant process-plane coordinates without changing truth data."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.model_registry_residual import model_from_checkpoint
from weld_pipeline.typed_contact_model import infer_probabilities
from weld_pipeline.interval_decoder import decode_intervals
from weld_pipeline.interval_config import load_interval_config
from weld_pipeline.evaluation import score_welds


def gate_parallel_planes(features, axes, base_features=84):
    count = len(axes)
    if features.shape[1] != base_features + 5 * count:
        raise ValueError('Unexpected process-plane encoding')
    result = features.clone()
    crossing = (features[:, 57 + axes].abs() > 1e-4).to(features.dtype)
    radial = result[:, base_features:base_features + 4 * count]
    radial *= crossing.repeat_interleave(4, dim=1)
    result[:, base_features + 4 * count:] *= crossing
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--groups', nargs='+', required=True)
    ap.add_argument('--data', default='weld_pipeline/data/partial_augmented_surface_v2')
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.outdir)
    split = json.loads((root/'split.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4)
    with Path(a.checkpoint).open('rb') as f:
        state = torch.load(f, map_location='cpu')
    verify_cached_inference_sources(state, a.checkpoint)
    model = model_from_checkpoint(state).cuda().eval()
    parameters, _ = load_interval_config(a.config, a.checkpoint)
    rows = []
    for name in a.groups:
        if name not in split['train'] + split['validation']:
            raise ValueError('Group outside frozen split')
        folder = (root/split.get('piece_roots', {}).get(name, name)).resolve()
        if sha256(folder/'manifest.json') != state['data_hashes'][name]:
            raise ValueError('Cache changed')
        piece = load_piece(folder)
        if piece['record'].get('target_policy') == 'all_channel_minus_one_is_unknown':
            raise ValueError('Whole-piece diagnostic requires complete truth')
        baseline = infer_probabilities(model, piece['sequences'])
        handle = model.net[0].register_forward_pre_hook(
            lambda module, inputs: (gate_parallel_planes(inputs[0], model.plane_axes),))
        try:
            gated = infer_probabilities(model, piece['sequences'])
        finally:
            handle.remove()
        modes = {}
        for label, probabilities in [('original', baseline), ('parallel_plane_gated', gated)]:
            pred = decode_intervals(piece['sequences'], probabilities, **parameters)
            score = score_welds(pred, piece['truth'])
            vertical_truth = [w for w in piece['truth'] if w['type'] == 'vertical']
            modes[label] = {k:score[k] for k in ['valid_welds', 'truth_count', 'prediction_count', 'false_or_duplicate_predictions']}
            if vertical_truth:
                vs = score_welds([w for w in pred if w['type'] == 'vertical'], vertical_truth)
                modes[label]['vertical'] = {k:vs[k] for k in ['valid_welds', 'truth_count', 'prediction_count']}
        changes = []
        for seq, target, before, after in zip(piece['sequences'], piece['targets'], baseline, gated):
            foreground = target[:, 0] > .5
            if seq['type'] == 'vertical' and foreground.any():
                changes.append({'foreground_before': float(before[foreground, 0].mean()),
                                'foreground_after': float(after[foreground, 0].mean()),
                                'max_prediction_change': float(np.max(np.abs(before-after)))})
        row = {'group':name, 'role':'development' if name in split['validation'] else 'training',
               'modes':modes, 'positive_vertical_sequences':changes}
        rows.append(row)
        print(json.dumps({k:row[k] for k in ['group', 'role', 'modes']}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out/'receipt.json').write_text(json.dumps({'scope':'Frozen-weight architecture ablation; not a trained replacement or acceptance.',
        'checkpoint_sha256':sha256(a.checkpoint), 'config_sha256':sha256(a.config),
        'source_sha256':sha256(__file__), 'rows':rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
