"""Offline causal head diagnostics; teacher values never enter runtime inference."""
import json
import time
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame, restore_probability_order
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.near_event_votes import dense_targets, vote_heatmaps

RUN = Path('weld_pipeline/runs/near_event_capacity_156')
BASE = Path('weld_pipeline/data/joint_relation_v1')
OUT = Path('weld_pipeline/reports/near_event_head_interventions_157_20261001')
INTERVENTIONS = ('learned', 'teacher_FG', 'teacher_outer', 'teacher_FG_outer',
                 'teacher_cut', 'teacher_FG_cut', 'teacher_outer_cut', 'all_teacher')


def main():
    if OUT.exists():
        raise ValueError('Preserve fixed head-intervention diagnosis')
    done = json.loads((RUN/'complete.json').read_text('utf8'))
    config = json.loads((RUN/'config.json').read_text('utf8')); names = config['train']
    if done['selected'] is not None or any(r['fit_gate_passed'] for r in done['variants'].values()):
        raise ValueError('This audit is for the two frozen failed capacity trials')
    hashes = {v:r['checkpoint_sha256'] for v, r in done['variants'].items()}
    if any(sha256(RUN/v/'final.pt') != h for v, h in hashes.items()):
        raise ValueError('Frozen near-event weights changed')
    OUT.mkdir(parents=True)
    protocol = dict(train=names, variants=list(hashes), interventions=INTERVENTIONS,
        frozen_checkpoint_sha256=hashes, original_labels_changed=False,
        decode_parameters=dict(threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.),
        selection='Diagnostic only; no teacher-assisted predictor or new threshold adopted.',
        observed_check_read=False, held_sources_read=False, new_model_training=False,
        sources={p:sha256(p) for p in [__file__, 'weld_pipeline/near_event_votes.py',
                                     'weld_pipeline/contract_child_evaluation.py', 'weld_pipeline/event_gap_decoder.py']},
        full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf8')
    results = {v:{key:{} for key in INTERVENTIONS} for v in hashes}
    cache_hashes = {}; started = time.perf_counter()
    for name in names:
        source = json.loads((BASE/name/'manifest.json').read_text('utf8'))
        piece = load_piece(source['source_cache_root'], verify=True); sequences = piece['sequences']
        truth = [w for w in piece['truth'] if w['type'] == 'horizontal'
                 and all(s['kind'] == 'line' for s in w['segments'])]
        teachers = []
        for seq, target in zip(sequences, piece['targets']):
            p = np.zeros((len(target), 7), np.float32)
            if seq['type'] == 'horizontal' and seq.get('curve_kind') != 'arc':
                _, labels, transform = canonical_frame(seq, target)
                dense = dense_targets(labels, seq['sample_step_mm'])
                p = restore_probability_order(vote_heatmaps(dense[:, 0], dense[:, 1:4],
                    dense[:, 4:7], seq['sample_step_mm']), transform)
            teachers.append(p)
        for variant in hashes:
            root = RUN/variant; heatmap = root/('FIT_'+name+'_heatmaps.npz')
            cache_hashes[str(heatmap)] = sha256(heatmap)
            with np.load(heatmap, allow_pickle=False) as archive:
                lengths = archive['lengths']; values = archive['probabilities']
            if lengths.tolist() != [len(s['xyz']) for s in sequences]:
                raise ValueError('Frozen full-grid prediction order changed')
            learned = np.split(values, np.cumsum(lengths)[:-1])
            for mask, label in enumerate(INTERVENTIONS):
                probabilities = []
                for pred, teacher in zip(learned, teachers):
                    mixed = pred.copy()
                    if mask & 1:
                        mixed[:, 0] = teacher[:, 0]
                    if mask & 2:
                        mixed[:, 1:3] = teacher[:, 1:3]
                    if mask & 4:
                        mixed[:, 3] = teacher[:, 3]
                    probabilities.append(mixed)
                predictions = decode_intervals(sequences, probabilities,
                    threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
                score = score_children(predictions, truth); results[variant][label][name] = score
                if mask == 0:
                    frozen = json.loads((root/('FIT_'+name+'.json')).read_text('utf8'))
                    if (predictions != frozen['predictions'] or score['valid_children'] != frozen['score']['valid_children']
                            or score['false_predictions'] != frozen['score']['false_predictions']):
                        raise ValueError('No-intervention result differs from the frozen trial')
                if mask == 7 and (score['valid_children'] != score['truth_count'] or score['false_predictions'] != 0):
                    raise ValueError('Correct near-target baseline failed under intervention')
                (OUT/(variant+'_'+label+'_'+name+'.json')).write_text(json.dumps(
                    dict(score=score, predictions=predictions, diagnostic_teacher_channels=mask), indent=2), encoding='utf8')
            print('NEAR_HEAD_INTERVENTIONS', variant, name,
                  {k:(results[variant][k][name]['valid_children'], results[variant][k][name]['false_predictions'])
                   for k in INTERVENTIONS}, flush=True)
    totals = {v:{key:aggregate(scores) for key, scores in interventions.items()}
              for v, interventions in results.items()}
    if any(s['truth_count'] != 1221 for r in totals.values() for s in r.values()):
        raise ValueError('Full head-intervention denominator changed')
    if any(sha256(RUN/v/'final.pt') != h for v, h in hashes.items()):
        raise ValueError('Diagnostic changed frozen weights')
    receipt = dict(config_sha256=sha256(OUT/'config.json'), results=totals,
        frozen_checkpoint_sha256=hashes, full_grid_prediction_sha256=cache_hashes,
        actual_baselines_exact=True, all_teacher_targets_1221_FP0=True,
        diagnostic_only=True, teachers_used_only_after_inference=True,
        model_or_decoder_adopted=False, held_or_observed_read=False,
        elapsed_seconds=time.perf_counter()-started, full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf8')
    print('NEAR_HEAD_INTERVENTIONS_COMPLETE', {v:{k:(s['valid_children'], s['false_predictions'])
          for k, s in rows.items()} for v, rows in totals.items()}, flush=True)


if __name__ == '__main__':
    main()
