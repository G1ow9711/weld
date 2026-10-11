"""Recall diagnostics for annotated welds in otherwise unlabeled context."""
from weld_pipeline.evaluation import score_welds
from weld_pipeline.curve_geometry_metrics import check_matched_curves


def score_partial_welds(predictions, known_truth):
    matched = score_welds(predictions, known_truth)
    curves = check_matched_curves(predictions, known_truth, matched)
    return {
        'role': 'training_partial_augmentation',
        'scope': 'Known-weld recall only. Unmatched predictions are unassessed; no whole-piece precision or acceptance claim.',
        'truth_count': len(known_truth),
        'prediction_count': len(predictions),
        'valid_welds': matched['valid_welds'],
        'known_weld_recall': matched['valid_welds'] / len(known_truth),
        'false_or_duplicate_predictions': None,
        'all_gates_pass': None,
        'unassessed_prediction_indices': matched['unmatched_prediction_indices'],
        'per_truth': matched['per_truth'],
        'curve_geometry': curves,
        'endpoint_limit_mm': matched['endpoint_limit_mm'],
        'segment_endpoint_limit_mm': matched['segment_endpoint_limit_mm'],
    }
