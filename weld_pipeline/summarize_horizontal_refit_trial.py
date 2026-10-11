"""Collect completed trial evidence without changing acceptance criteria."""
import json
from pathlib import Path
import numpy as np
from training.labels import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    run = Path('weld_pipeline/runs/horizontal_refit_v1_seed120')
    complete = read(run / 'complete.json')
    best = read(run / 'best_validation.json')
    evaluation = read(run / 'final_evaluation/receipt.json')
    assert evaluation['checkpoint_sha256'] == sha256(run / 'best.pt')
    groups = evaluation['groups']
    dev = [g for g in groups if g['role'] == 'development']
    original = [g for g in groups if g['role'] == 'training_fit']
    errors, segment_errors, vertical, same, raw = [], [], [], True, {}
    for row in dev:
        score = read(run / 'final_evaluation' / (row['group'] + '_score.json'))
        old = read(Path('weld_pipeline/runs/parent_balanced_branch_v4_seed117/final_evaluation') / (row['group'] + '_score.json'))
        accepted = [r for r in score['per_truth'] if r['accepted']]
        prior = [r for r in old['per_truth'] if r['accepted']]
        key = lambda r: (r['truth_id'], r['start_mm'], r['end_mm'], r['segment_endpoint_max_mm'])
        same = same and sorted(map(key, accepted)) == sorted(map(key, prior))
        errors.extend(e for r in accepted for e in (r['start_mm'], r['end_mm']))
        segment_errors.extend(r['segment_endpoint_max_mm'] for r in accepted)
        if 'vertical' in score['by_type']:
            vertical.append(score['by_type']['vertical'])
    for group, folder in [('527-FR146A', 'raw_FR146A'), ('TD2-LB8A', 'raw_TD2')]:
        dest = run / folder
        score = read(dest / 'independent_score.json')
        cached = read(run / 'final_evaluation' / (group + '_score.json'))
        assert score['per_truth'] == cached['per_truth']
        assert score['input_guard']['status'] == 'PASS'
        inference = read(dest / 'inference_receipt.json')
        export = read(dest / 'export_qa.json')
        formula = read(dest / 'formula_check.json')
        assert export['csv_excel_ply_all_coordinates_and_ids'] == 'PASS'
        assert export['black_white_no_fill'] == 'PASS'
        assert formula['error_count'] == 0
        assert inference['checkpoint_sha256'] == sha256(run / 'best.pt')
        raw[group] = {
            'valid': score['valid_welds'], 'truth': score['truth_count'],
            'predictions': score['prediction_count'], 'input_guard': 'PASS',
            'per_truth_exact_to_cache': True,
            'inference_receipt_sha256': sha256(dest / 'inference_receipt.json'),
            'raw_pcd_to_csv_seconds': inference['raw_pcd_to_csv_seconds'],
            'timing_scope': 'Single development/training diagnostic run, not speed acceptance.',
            'feature_equivalence_receipt_sha256': inference['feature_equivalence_receipt_sha256'],
            'export_qa': export,
            'source_archive': read(dest / 'source_snapshot/verified_sources.json'),
            'formula_check': formula,
        }
    report = {
        'status': 'NOT_ACHIEVED', 'model': run.name,
        'completed_epochs': complete['completed_epochs'], 'epoch': best['epoch'],
        'elapsed_seconds': complete['elapsed_seconds'],
        'development_groups': dev,
        'development_valid': sum(r['valid'] for r in dev),
        'development_truth': sum(r['truth'] for r in dev),
        'development_false': sum(r['false'] for r in dev),
        'development_vertical': {k: sum(r[k] for r in vertical) for k in ['valid_welds', 'truth_count', 'prediction_count']},
        'accepted_geometry': {
            'scope': 'Accepted development welds only; not all truth or blind acceptance.',
            'accepted_welds': len(errors) // 2, 'endpoint_count': len(errors),
            'mean_endpoint_mm': float(np.mean(errors)), 'max_endpoint_mm': max(errors),
            'p95_endpoint_mm': float(np.percentile(errors, 95)),
            'max_segment_endpoint_mm': max(segment_errors),
            'accepted_truth_ids_and_errors_equal_to_117': same,
        },
        'training_pass_count': sum(r['gates_pass'] for r in original),
        'training_count': len(original),
        'training_failures': [r for r in original if not r['gates_pass']],
        'augmentation': [r for r in groups if r['role'].startswith('training_') and r['role'] != 'training_fit'],
        'raw_checks': raw,
        'checkpoint_sha256': sha256(run / 'best.pt'),
        'final_evaluation_receipt_sha256': sha256(run / 'final_evaluation/receipt.json'),
        'scope': 'Development experiment, not formal customer acceptance.',
        'remaining_requirements': ['Every development piece must pass >95% and <=5% invalid predictions',
            'Real single-segment vertical recovery', 'Mixed line/arc whole-weld recovery',
            'Independent arc validation', 'Eligible 90-110-weld speed piece and three runs <=10s'],
    }
    (run / 'goal_completion_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'epoch', 'development_valid', 'development_truth', 'development_false', 'development_vertical', 'accepted_geometry', 'training_pass_count']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
