"""Read-only audit: votes for one known event should converge to one position."""
import json
from pathlib import Path

import numpy as np

from training.labels import sha256

AUDIT = Path('weld_pipeline/reports/near_event_support_localization_158_20261001')
TARGETS = Path('weld_pipeline/data/near_event_pilot156')
BASE = Path('weld_pipeline/data/joint_relation_v1')
OUT = Path('weld_pipeline/reports/near_vote_stationarity_159_20261001')


def summary(slopes, spreads):
    slopes = np.concatenate(slopes) if slopes else np.empty(0)
    spreads = np.concatenate(spreads) if spreads else np.empty(0)
    errors = np.abs(slopes+1.)
    return dict(same_event_adjacent_pairs=len(slopes),
        mean_offset_derivative=float(slopes.mean()) if len(slopes) else None,
        mean_derivative_error_from_minus_one=float(errors.mean()) if len(slopes) else None,
        median_derivative_error=float(np.median(errors)) if len(slopes) else None,
        derivative_within_0_25_fraction=float((errors<=.25).mean()) if len(slopes) else None,
        known_event_groups=len(spreads),
        median_predicted_center_spread_mm=float(np.median(spreads)) if len(spreads) else None,
        p95_predicted_center_spread_mm=float(np.quantile(spreads, .95)) if len(spreads) else None,
        event_center_spread_within_3mm_fraction=float((spreads<=3.).mean()) if len(spreads) else None,
        diagnostic_values_are_not_contract_geometry_scores=True)


def main():
    if OUT.exists():
        raise ValueError('Preserve frozen stationarity audit')
    done = json.loads((AUDIT/'complete.json').read_text('utf8'))
    cfg = json.loads((AUDIT/'config.json').read_text('utf8')); names = cfg['train']
    if not done['exact_frozen_heatmap_and_prediction_replay'] or not done['teacher_both_1221_FP0']:
        raise ValueError('Finished raw/support/localization proof required')
    target_manifest = json.loads((TARGETS/'manifest.json').read_text('utf8'))
    if sha256(TARGETS/'manifest.json') != cfg['target_manifest_sha256']:
        raise ValueError('Frozen near-target manifest changed')
    OUT.mkdir(parents=True)
    all_slopes = {head:[] for head in ('start', 'end', 'cut')}
    all_spreads = {head:[] for head in all_slopes}; results = {}; target_error = 0.
    for name in names:
        rr = done['raw_arrays'][name]; raw_path = Path(rr['path'])
        if sha256(raw_path) != rr['sha256']:
            raise ValueError('Frozen raw forward values changed')
        raw = np.load(raw_path, mmap_mode='r')
        target = np.load(TARGETS/(name+'_targets.npy'), mmap_mode='r')
        known = np.load(TARGETS/(name+'_known.npy'), mmap_mode='r')
        manifest = json.loads((BASE/name/'manifest.json').read_text('utf8'))
        if (sha256(BASE/name/'manifest.json') != target_manifest[name]['source_manifest_sha256']
                or sha256(TARGETS/(name+'_targets.npy')) != target_manifest[name]['targets_sha256']
                or sha256(TARGETS/(name+'_known.npy')) != target_manifest[name]['known_sha256']):
            raise ValueError('Stationarity data source changed')
        slopes = {head:[] for head in all_slopes}; spreads = {head:[] for head in all_slopes}
        for entry in manifest['entries']:
            if entry['type'] != 'horizontal':
                continue
            a, b = entry['start'], entry['stop']; step = 2.
            for c, head in enumerate(all_slopes):
                y = target[a:b, c+4].astype(float); k = known[a:b, c+4] > .5
                prediction = np.clip(raw[a:b, c+4].astype(float), -1., 1.)
                true_derivative = np.diff(y)*16./step
                same = k[:-1] & k[1:] & (np.abs(true_derivative+1.) < 5e-5)
                if np.any(same):
                    target_error = max(target_error, float(np.abs(true_derivative[same]+1.).max()))
                    slopes[head].append(np.diff(prediction)[same]*16./step)
                ids = np.flatnonzero(k)
                if not len(ids):
                    continue
                true_centers = ids*step+y[ids]*16.
                keys = np.round(true_centers, 4)
                pred_centers = ids*step+prediction[ids]*16.
                values = []
                for key in np.unique(keys):
                    take = keys == key
                    if take.sum() >= 2:
                        values.append(float(np.ptp(pred_centers[take])))
                if values:
                    spreads[head].append(np.asarray(values))
        results[name] = {head:summary(slopes[head], spreads[head]) for head in all_slopes}
        for head in all_slopes:
            all_slopes[head].extend(slopes[head]); all_spreads[head].extend(spreads[head])
    totals = {head:summary(all_slopes[head], all_spreads[head]) for head in all_slopes}
    receipt = dict(train=names, checkpoint_sha256=done['checkpoint_sha256'], results=results, totals=totals,
        parent_audit_sha256=sha256(AUDIT/'complete.json'), target_expected_derivative_max_error=target_error,
        rule='For one event: center = query_position + 16mm*offset; d(offset_mm)/d(query_mm) = -1.',
        same_event_identity_used_only_offline_from_explicit_known_target_pairs=True,
        clipping_matches_frozen_runtime=True, minimum_accepted_spread_not_changed=True,
        new_model_training=False, original_labels_changed=False, diagnostic_only=True,
        held_or_observed_read=False, full_contract_pass=False, source_sha256=sha256(__file__))
    (OUT/'complete.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf8')
    print('NEAR_VOTE_STATIONARITY_COMPLETE', json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
