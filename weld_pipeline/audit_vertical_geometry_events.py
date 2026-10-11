"""Reference-assisted endpoint geometry inspection, never an inference rule."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--groups', nargs='+', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    root, out = Path(args.data), Path(args.outdir)
    split = json.loads((root / 'split.json').read_text(encoding='utf-8'))
    rows, sources = [], {}
    for group in args.groups:
        if group not in split['train'] + split['validation']:
            raise ValueError('Unknown split member')
        folder = (root / split.get('piece_roots', {}).get(group, group)).resolve()
        piece = load_piece(folder)
        sources[str(folder / 'manifest.json')] = sha256(folder / 'manifest.json')
        mapping = piece['record']['mapping']['line_mapping']['weld_to_sequence']
        line_ids = [i for i, s in enumerate(piece['sequences']) if s.get('curve_kind') != 'arc']
        lines = [w for w in piece['truth'] if all(s['kind'] == 'line' for s in w['segments'])]
        for wi, weld in enumerate(lines):
            if weld['type'] != 'vertical' or str(wi) not in mapping:
                continue
            si = line_ids[mapping[str(wi)]]
            seq = piece['sequences'][si]
            feature = np.asarray(seq['features'])[:, np.r_[0:49, 70:84]] * 50.
            change = np.zeros(len(feature))
            change[1:-1] = np.sqrt(np.mean((feature[2:] - feature[:-2]) ** 2, axis=1))
            xyz, direction = np.asarray(seq['xyz']), np.asarray(seq['direction'])
            ends = np.array([weld['segments'][0]['points'][0], weld['segments'][-1]['points'][-1]])
            for label, point in zip(['start', 'end'], ends):
                along = (xyz - point) @ direction
                window = np.flatnonzero(np.abs(along) <= 20.)
                if not len(window):
                    continue
                index = window[np.argmax(change[window])]
                near = np.abs(along) <= 3.
                rows.append({'group': group, 'role': 'development' if group in split['validation'] else 'training',
                             'number': weld['metadata'].get('WeldNumber (f/v + Name number)'),
                             'endpoint': label, 'reference_xyz_mm': point.tolist(), 'sequence': si,
                             'strongest_change_offset_mm': float(along[index]),
                             'strongest_change_rms_mm': float(change[index]),
                             'max_change_within_3mm': float(change[near].max()) if near.any() else None})
    report = {'scope': 'Reference-centered feature variation diagnosis; largest variation need not be a weld endpoint. Not a prediction or acceptance result.',
              'window_mm': 20., 'difference_span_mm': 4., 'sources': sources,
              'script_sha256': sha256(__file__), 'rows': rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'geometry_events.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for group in args.groups:
        for end in ['start', 'end']:
            selected = [r for r in rows if r['group'] == group and r['endpoint'] == end]
            print(json.dumps({'group': group, 'endpoint': end, 'count': len(selected),
                              'median_abs_peak_offset_mm': float(np.median([abs(r['strongest_change_offset_mm']) for r in selected])) if selected else None,
                              'diagnostic_only': True}), flush=True)


if __name__ == '__main__':
    main()
