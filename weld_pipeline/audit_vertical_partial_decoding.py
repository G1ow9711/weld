"""Separate local endpoint learning from full-context partial-label decoding."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.model_registry_vertical import model_from_checkpoint
from weld_pipeline.typed_contact_model import infer_probabilities
from weld_pipeline.partial_supervision import clean_targets
from weld_pipeline.event_gap_config import load_interval_config
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.partial_evaluation import score_partial_welds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.outdir)
    split = json.loads((root / 'split.json').read_text(encoding='utf-8'))
    with Path(a.checkpoint).open('rb') as f:
        state = torch.load(f, map_location='cpu')
    verify_cached_inference_sources(state, a.checkpoint)
    torch.set_num_threads(4)
    model = model_from_checkpoint(state).cuda().eval()
    parameters, _ = load_interval_config(a.config, a.checkpoint)
    rows = []
    for variant in split['augmentation_policy']['variants']:
        if not variant.get('partial_supervision'):
            continue
        name = variant['name']
        folder = (root / split['piece_roots'][name]).resolve()
        if sha256(folder / 'manifest.json') != state['data_hashes'][name]:
            raise ValueError('Cache changed')
        piece = load_piece(folder)
        probabilities = infer_probabilities(model, piece['sequences'])
        constrained, heads = [], []
        for index, (p, target) in enumerate(zip(probabilities, piece['targets'])):
            clean, known = clean_targets(target)
            constrained.append(np.where(known[:, None] > 0, p, 0.))
            peaks = {}
            for channel in [1, 2, 3]:
                mask = (clean[:, channel] > .5) & (known > 0)
                peaks[str(channel)] = {'target_present': bool(mask.any()),
                                      'predicted_peak_near_target': float(p[mask, channel].max()) if mask.any() else None}
            heads.append({'sequence': index, 'heads': peaks,
                          'unknown_foreground_mean': float(p[known == 0, 0].mean()) if (known == 0).any() else None})
        full = decode_intervals(piece['sequences'], probabilities, **parameters)
        # The mask comes from TRAIN annotations. This counterfactual must never
        # be used for inference, threshold selection, or acceptance scoring.
        masked = decode_intervals(piece['sequences'], constrained, **parameters)
        row = {'group': name, 'full_context_known_recall': score_partial_welds(full, piece['truth']),
               'oracle_known_region_diagnostic': score_partial_welds(masked, piece['truth']), 'heads': heads}
        rows.append(row)
        print(json.dumps({'group': name, 'full_valid': row['full_context_known_recall']['valid_welds'],
                          'oracle_masked_valid': row['oracle_known_region_diagnostic']['valid_welds'], 'heads': heads}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'TRAIN diagnostic only. Oracle masking is not an inference result or an acceptance metric.',
              'checkpoint_sha256': sha256(a.checkpoint), 'configuration_sha256': sha256(a.config),
              'source_sha256': sha256(__file__), 'rows': rows}
    (out / 'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
