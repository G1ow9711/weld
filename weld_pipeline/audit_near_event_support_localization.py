"""Fixed FIT diagnosis of event existence and known localization, never runtime truth."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.labels import sha256
from weld_pipeline.audit_event_offset_distance_bins import raw_predictions
from weld_pipeline.canonical_horizontal_patch_heatmaps import restore_probability_order
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.near_event_votes import vote_heatmaps
from weld_pipeline.predict_near_events import load_model
from weld_pipeline.train_joint_relation_crossfit import DATA, load_data
from weld_pipeline.train_joint_workpiece_event_capacity import attach_context_and_targets

RUN = Path('weld_pipeline/runs/near_event_capacity_156')
VARIANT = 'physical_matched'
TARGETS = Path('weld_pipeline/data/near_event_pilot156')
OUT = Path('weld_pipeline/reports/near_event_support_localization_158_20261001')
INTERVENTIONS = (
    ('learned', (), ()), ('teacher_support', (0, 1, 2), ()),
    ('teacher_known_offsets', (), (0, 1, 2)), ('teacher_both', (0, 1, 2), (0, 1, 2)),
    ('teacher_outer_support', (0, 1), ()), ('teacher_outer_known_offsets', (), (0, 1)),
    ('teacher_cut_support', (2,), ()), ('teacher_cut_known_offsets', (), (2,)))


def probabilities(item, sequences, raw, targets, known, support_heads, offset_heads):
    result = [np.zeros((len(s['xyz']), 7), np.float32) for s in sequences]
    for entry in item['horizontal']:
        a, b = entry['start'], entry['stop']; index = entry['sequence']
        logits = raw[a:b]
        sigmoid = lambda v:1./(1.+np.exp(-np.clip(v, -60., 60.)))
        fg = sigmoid(logits[:, 0]); support = sigmoid(logits[:, 1:4]); offsets = logits[:, 4:7].copy()
        for c in support_heads:
            valid = known[a:b, c+1] > .5
            support[valid, c] = targets[a:b, c+1][valid]
        for c in offset_heads:
            valid = known[a:b, c+4] > .5
            offsets[valid, c] = targets[a:b, c+4][valid]
        p = vote_heatmaps(fg, support, offsets, sequences[index]['sample_step_mm'])
        result[index] = restore_probability_order(p, entry['transform'])
    return result


def sample_metrics(item, raw, targets, known):
    rows = np.isfinite(raw[:, 0]); result = {}
    for c, head in enumerate(('start', 'end', 'cut')):
        valid = rows & (known[:, c+1] > .5)
        gold = targets[valid, c+1] > .5
        predicted = raw[valid, c+1] >= 0.
        pos = int(gold.sum()); neg = int((~gold).sum())
        tp = int((gold & predicted).sum()); fp = int((~gold & predicted).sum())
        loc = rows & (known[:, c+4] > .5)
        errors = np.abs(np.clip(raw[loc, c+4], -1., 1.)-targets[loc, c+4])*16.
        result[head] = dict(positive_points=pos, negative_points=neg,
            support_TP=tp, support_FP=fp, support_FN=pos-tp,
            support_recall=tp/pos if pos else None, support_negative_FP_rate=fp/neg if neg else None,
            known_offset_points=len(errors), offset_error_sum_mm=float(errors.astype(float).sum()),
            mean_offset_error_mm=float(errors.mean()) if len(errors) else None,
            p95_offset_error_mm=float(np.quantile(errors, .95)) if len(errors) else None,
            offset_within_3mm=int((errors <= 3.).sum()),
            offset_within_3mm_fraction=float((errors <= 3.).mean()) if len(errors) else None,
            sample_scores_are_not_geometry_acceptance=True)
    return result


def main():
    if OUT.exists():
        raise ValueError('Preserve fixed support/localization audit')
    cfg = json.loads((RUN/'config.json').read_text('utf8'))
    done = json.loads((RUN/'complete.json').read_text('utf8'))
    root = RUN/VARIANT; checkpoint = root/'final.pt'
    frozen = sha256(checkpoint); names = cfg['train']
    if frozen != done['variants'][VARIANT]['checkpoint_sha256'] or done['selected'] is not None:
        raise ValueError('Only the frozen failed near model may be analyzed')
    proof_path = Path('weld_pipeline/reports/near_event_runtime_156_20261001')/VARIANT/'complete.json'
    proof = json.loads(proof_path.read_text('utf8'))
    if not proof['all_checks_passed'] or proof['checkpoint_sha256'] != frozen:
        raise ValueError('Raw model input replay proof required')
    target_manifest = json.loads((TARGETS/'manifest.json').read_text('utf8'))
    if target_manifest != done['target_manifests']:
        raise ValueError('Frozen near targets changed')
    torch.set_num_threads(2); torch.backends.cudnn.benchmark = False
    base_config = json.loads((DATA/'config.json').read_text('utf8'))
    data = load_data(dict(base_config, pieces=names)); attach_context_and_targets(data, names)
    OUT.mkdir(parents=True)
    protocol = dict(train=names, variant=VARIANT, checkpoint_sha256=frozen,
        interventions=INTERVENTIONS, teachers_used_only_after_frozen_inference=True,
        unknown_offset_values_remain_model_values=True, offset_known_mask_is_explicit=True,
        original_runtime_proof_sha256=sha256(proof_path),
        target_manifest_sha256=sha256(TARGETS/'manifest.json'),
        sources={p:sha256(p) for p in [__file__, 'weld_pipeline/near_event_votes.py',
            'weld_pipeline/audit_event_offset_distance_bins.py', 'weld_pipeline/contract_child_evaluation.py']},
        held_sources_read=False, observed_check_read=False, diagnostic_only=True, full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf8')
    model = load_model(checkpoint); results = {key:{} for key, _, _ in INTERVENTIONS}
    metrics = {}; raws = {}; started = time.perf_counter()
    for name in names:
        record = target_manifest[name]; item = data[name]
        if (sha256(TARGETS/(name+'_targets.npy')) != record['targets_sha256']
                or sha256(TARGETS/(name+'_known.npy')) != record['known_sha256']
                or sha256(DATA/name/'manifest.json') != record['source_manifest_sha256']):
            raise ValueError('Frozen diagnostic arrays changed')
        target = np.load(TARGETS/(name+'_targets.npy'), mmap_mode='r')
        known = np.load(TARGETS/(name+'_known.npy'), mmap_mode='r')
        raw = raw_predictions(model, item)
        raw_path = OUT/(name+'_raw.npy'); np.save(raw_path, raw)
        raws[name] = dict(path=str(raw_path), sha256=sha256(raw_path), shape=list(raw.shape))
        piece = load_piece(item['manifest']['source_cache_root'], verify=True); sequences = piece['sequences']
        truth = [w for w in piece['truth'] if w['type'] == 'horizontal'
                 and all(s['kind'] == 'line' for s in w['segments'])]
        metrics[name] = sample_metrics(item, raw, target, known)
        for label, support, offset in INTERVENTIONS:
            p = probabilities(item, sequences, raw, target, known, support, offset)
            predictions = decode_intervals(sequences, p,
                threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
            score = score_children(predictions, truth); results[label][name] = score
            if label == 'learned':
                with np.load(root/('FIT_'+name+'_heatmaps.npz'), allow_pickle=False) as cache:
                    if (cache['lengths'].tolist() != [len(v) for v in p]
                            or not np.array_equal(np.concatenate(p), cache['probabilities'])):
                        raise ValueError('Raw forward pass does not reproduce frozen near heatmaps exactly')
                original = json.loads((root/('FIT_'+name+'.json')).read_text('utf8'))
                if predictions != original['predictions']:
                    raise ValueError('Unchanged baseline predictions differ')
            if label == 'teacher_both' and (score['valid_children'] != score['truth_count'] or score['false_predictions'] != 0):
                raise ValueError('Correct support/localization cannot express all source children')
            (OUT/(label+'_'+name+'.json')).write_text(json.dumps(dict(score=score, predictions=predictions), indent=2), encoding='utf8')
        print('NEAR_SUPPORT_LOCALIZATION', name,
              {label:(results[label][name]['valid_children'], results[label][name]['false_predictions'])
               for label, _, _ in INTERVENTIONS}, flush=True)
    total = {key:aggregate(scores) for key, scores in results.items()}
    if any(r['truth_count'] != 1221 for r in total.values()) or sha256(checkpoint) != frozen:
        raise ValueError('Frozen denominator or model changed')
    sample_totals = {}
    for head in ('start', 'end', 'cut'):
        rows = [m[head] for m in metrics.values()]
        sums = {k:sum(r[k] for r in rows) for k in ('positive_points', 'negative_points', 'support_TP',
            'support_FP', 'support_FN', 'known_offset_points', 'offset_error_sum_mm', 'offset_within_3mm')}
        sums.update(support_recall=sums['support_TP']/max(1, sums['positive_points']),
            support_negative_FP_rate=sums['support_FP']/max(1, sums['negative_points']),
            mean_offset_error_mm=sums['offset_error_sum_mm']/max(1, sums['known_offset_points']),
            offset_within_3mm_fraction=sums['offset_within_3mm']/max(1, sums['known_offset_points']))
        sample_totals[head] = sums
    receipt = dict(config_sha256=sha256(OUT/'config.json'), checkpoint_sha256=frozen,
        results=total, sample_metrics=metrics, sample_totals=sample_totals, raw_arrays=raws,
        exact_frozen_heatmap_and_prediction_replay=True, teacher_both_1221_FP0=True,
        all_known_offset_masks_explicit=True, diagnostic_only=True,
        no_teacher_assisted_runtime_or_new_model_adopted=True, held_or_observed_read=False,
        elapsed_seconds=time.perf_counter()-started, full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf8')
    print('NEAR_SUPPORT_LOCALIZATION_COMPLETE', {k:(v['valid_children'], v['false_predictions']) for k, v in total.items()}, flush=True)


if __name__ == '__main__':
    main()
