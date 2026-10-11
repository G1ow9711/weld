"""FIT-only diagnostic: feed true training heatmaps to the retained decoder."""
import json
import shutil
import time
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals


def main():
    root = Path('weld_pipeline/reports/decoder_oracle_20261001')
    if root.exists():
        raise ValueError('Oracle diagnostic already exists; do not overwrite')
    root.mkdir(parents=True)
    split_path = Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split = json.loads(split_path.read_text(encoding='utf-8'))
    names = [name for name in split['train'] if '__' not in name]
    config = {'role': 'FIT_ORACLE_DIAGNOSTIC_NOT_MODEL_PREDICTION', 'pieces': names,
              'input': 'exact_first_four_training_target_heatmaps',
              'decoder': {'threshold': .5, 'weld_penalty': 2., 'mask_weight': .25, 'event_gap_mm': 0.},
              'check_or_formal_test_read': False, 'split_sha256': sha256(split_path),
              'source_sha256': sha256(__file__),
              'decoder_sha256': sha256('weld_pipeline/event_gap_decoder.py'),
              'score_sha256': sha256('weld_pipeline/contract_child_evaluation.py'),
              'runtime_acceptance': False, 'full_contract_pass': False}
    (root / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(__file__, root / Path(__file__).name)
    started = time.perf_counter()
    scores = {}
    for name in names:
        piece = load_piece(split['piece_roots'][name], verify=True)
        values = []
        for sequence, target in zip(piece['sequences'], piece['targets']):
            probability = np.zeros((len(target), 7), dtype=np.float32)
            if np.any(target[:, :4] < 0) or not np.isfinite(target[:, :4]).all():
                raise ValueError('Unexpected unknown/invalid geometric target in real FIT: ' + name)
            probability[:, :4] = target[:, :4]
            values.append(probability)
        predictions = decode_intervals(piece['sequences'], values, **config['decoder'])
        score = score_children(predictions, piece['truth'])
        scores[name] = score
        (root / (name + '.json')).write_text(json.dumps({'score': score, 'oracle_predictions': predictions,
            'diagnostic_not_trained_model': True}, ensure_ascii=False, indent=2), encoding='utf-8')
        print('ORACLE', name, score['valid_children'], '/', score['truth_count'],
              'FP', score['false_predictions'], flush=True)
    summary = aggregate(scores)
    summary.update(role=config['role'], runtime_acceptance=False, full_contract_pass=False,
                   elapsed_seconds=time.perf_counter()-started,
                   all_truth_heatmaps_reconstructed=(summary['valid_children']==summary['truth_count'] and summary['false_predictions']==0))
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ORACLE_COMPLETE', summary['valid_children'], '/', summary['truth_count'],
          'FP', summary['false_predictions'], 'NOT_MODEL_ACCEPTANCE', flush=True)


if __name__ == '__main__':
    main()
