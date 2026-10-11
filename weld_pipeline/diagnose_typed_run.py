"""Post-scoring failure diagnostics. Reference files are never inference inputs."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from weld_pipeline.train_typed_pilot import load
from weld_pipeline.evaluation import endpoints
from weld_pipeline.mapped_cache import load_piece,STORAGE_FORMAT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--data', default='weld_pipeline/data/typed_pilot_v5')
    args = ap.parse_args()
    root = Path(args.run)
    best = json.loads((root / 'best_validation.json').read_text(encoding='utf-8'))
    folder = root / ('validation_epoch_%03d' % best['epoch'])
    report = {'epoch': best['epoch'], 'scope': 'Post-prediction diagnosis only', 'groups': {}}
    data_root=Path(args.data)
    split=json.loads((data_root/'split.json').read_text(encoding='utf-8'))
    for name in best['validation']:
        piece = (load_piece(data_root/split.get('piece_roots',{}).get(name,name))
                 if split.get('storage_format')==STORAGE_FORMAT else load(data_root / (name + '.pt')))
        pred = json.loads((folder / (name + '_predictions.json')).read_text())
        score = json.loads((folder / (name + '_score.json')).read_text())
        causes = Counter()
        rows = []
        for item in score['per_truth']:
            if item['accepted']:
                causes['valid'] += 1
                continue
            truth = piece['truth'][item['truth_index']]
            idx = item.get('nearest_prediction_index')
            e = max(item.get('start_mm', 1e9), item.get('end_mm', 1e9))
            if idx is None:
                cause = 'no_predictions'
            elif e <= 3 and not item['topology_correct']:
                cause = 'segment_count_or_kind'
            elif e <= 3:
                cause = 'internal_endpoint_or_duplicate'
            elif e <= 10:
                cause = 'outer_endpoint_3_to_10mm'
            else:
                cause = 'missing_or_large_endpoint_error'
            causes[cause] += 1
            rows.append({'id': truth['id'], 'type': truth['type'], 'cause': cause,
                         'reference_endpoints_mm': endpoints(truth).tolist(),
                         'reference_segment_count': len(truth['segments']),
                         'nearest_prediction_endpoints_mm': endpoints(pred[idx]).tolist() if idx is not None else None,
                         'nearest_prediction_segment_count': len(pred[idx]['segments']) if idx is not None else None,
                         'errors': item,
                         'reference_metadata': truth.get('metadata', {})})
        summary = dict(causes)
        report['groups'][name] = {'summary': summary, 'failures': rows}
        print(name, summary, flush=True)
    (root / 'failure_diagnostics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
