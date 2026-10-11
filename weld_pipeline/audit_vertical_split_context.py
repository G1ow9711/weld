"""Read-only diagnostic of reference splitting semantics, never inference input."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from weld_pipeline.curved_cache import load_piece


def main():
    split = json.loads(Path('weld_pipeline/data/contract_child_v1/split.json').read_text('utf8'))
    report = {'scope': 'Reference-assisted development diagnosis; labels not runtime input', 'pieces': {}}
    coverage = []
    for name in split['train'] + split['validation']:
        piece = load_piece(split['piece_roots'][name], verify=True)
        vertical = [(s, t) for s, t in zip(piece['sequences'], piece['targets']) if s['type'] == 'vertical']
        positive = [(s, t) for s, t in vertical if (t[:, 0] > .5).any()]
        split_counts = Counter()
        for s, t in positive:
            known_foreground = t[:, 0] > .5
            split_counts[str(int(np.median(t[known_foreground, 6]) > .5))] += 1
        coverage.append({'name': name, 'role': 'TRAIN' if name in split['train'] else 'DEV',
                         'vertical_candidates': len(vertical), 'positive_candidates': len(positive),
                         'positive_side_split_counts': dict(split_counts),
                         'positive_candidate_lengths_mm': [round(float(s['sample_step_mm'] * (len(s['xyz'])-1)), 3)
                                                           for s, t in positive]})
    report['coverage'] = coverage
    print('COVERAGE', json.dumps([{k: v for k, v in r.items() if k != 'positive_candidate_lengths_mm'} for r in coverage]))
    for name in ['224-GR24A1', '527-FR146A']:
        piece = load_piece(split['piece_roots'][name], verify=True)
        rows = [w for w in piece['truth'] if w['type'] == 'vertical']
        keys = sorted({k for w in rows for k in w['metadata']})
        fields = {k: dict(Counter(json.dumps(w['metadata'].get(k), ensure_ascii=False) for w in rows))
                  for k in keys if 'Matrix' not in k and k not in ['ObjectId', 'Name']}
        details = []
        for w in rows:
            segments = []
            for seg in w['segments']:
                pts = np.asarray(seg['points'], float)
                segments.append({'kind': seg['kind'], 'start': pts[0].tolist(), 'end': pts[-1].tolist(),
                                 'length_mm': float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())})
            details.append({'id': w['id'], 'name': w['name'], 'segments': segments, 'metadata': w['metadata']})
        fractions = [[round(seg['length_mm'] / sum(s['length_mm'] for s in w['segments']), 6) for seg in w['segments']]
                     for w in details]
        report['pieces'][name] = {'vertical_parents': len(rows), 'fields': fields, 'welds': details,
                                 'segment_length_fractions': dict(Counter(json.dumps(f) for f in fractions)),
                                 'record': piece['record']}
        print(name, json.dumps({'parents': len(rows), 'split_flags': {k: v for k, v in fields.items() if 'Split' in k},
                               'lengths': fields['Length'], 'fractions': report['pieces'][name]['segment_length_fractions']}, ensure_ascii=False))
    out = Path('weld_pipeline/reports/contract_child_20260930/vertical_split_context.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
