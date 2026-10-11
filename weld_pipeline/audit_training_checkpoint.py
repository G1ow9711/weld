"""Separate original TRAIN fitting failures by weld type and endpoint cause."""
import argparse
import json
from pathlib import Path
import torch
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.model_registry_crossing import model_from_checkpoint
from weld_pipeline.typed_contact_model import infer_probabilities
from weld_pipeline.interval_decoder import decode_intervals
from weld_pipeline.interval_config import load_interval_config
from weld_pipeline.evaluation import score_welds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--groups', nargs='+', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--data', default='weld_pipeline/data/partial_augmented_surface_v2')
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.outdir)
    split = json.loads((root/'split.json').read_text(encoding='utf-8'))
    original = split.get('augmentation_policy', {}).get('original_train', split['train'])
    if not set(a.groups) <= set(original) or set(a.groups) & set(split['validation']):
        raise ValueError('Original TRAIN only')
    torch.set_num_threads(4)
    with Path(a.checkpoint).open('rb') as f:
        state = torch.load(f, map_location='cpu')
    verify_cached_inference_sources(state, a.checkpoint)
    model = model_from_checkpoint(state).cuda().eval()
    config, _ = load_interval_config(a.config, a.checkpoint)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in a.groups:
        folder = (root/split.get('piece_roots', {}).get(name, name)).resolve()
        if sha256(folder/'manifest.json') != state['data_hashes'][name]:
            raise ValueError('Cache changed')
        piece = load_piece(folder)
        pred = decode_intervals(piece['sequences'], infer_probabilities(model, piece['sequences']), **config)
        score = score_welds(pred, piece['truth'])
        types = {}
        for typ in ['horizontal', 'vertical']:
            truth = [w for w in piece['truth'] if w['type'] == typ]
            if truth:
                s = score_welds([w for w in pred if w['type'] == typ], truth)
                types[typ] = {k:s[k] for k in ['valid_welds', 'truth_count', 'prediction_count']}
        score['by_type'] = types
        (out/(name+'_score.json')).write_text(json.dumps(score, indent=2), encoding='utf-8')
        row = {'group':name, 'by_type':types}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out/'receipt.json').write_text(json.dumps({'scope':'Original TRAIN diagnostic, not independent validation.',
        'checkpoint_sha256':sha256(a.checkpoint), 'config_sha256':sha256(a.config),
        'source_sha256':sha256(__file__), 'rows':rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
