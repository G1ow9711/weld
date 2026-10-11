"""Measure local point spacing near reference endpoints; diagnostic only."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from training.dataio import read_xyz_pcd
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
            raise ValueError('Unknown group')
        folder = (root / split.get('piece_roots', {}).get(group, group)).resolve()
        piece = load_piece(folder)
        source = Path(piece['record']['input'])
        if sha256(source) != piece['record']['pcd_sha256']:
            raise ValueError('Raw source changed')
        xyz = read_xyz_pcd(source) * 1000.
        tree = cKDTree(xyz)
        sources[str(source)] = sha256(source)
        sources[str(folder / 'manifest.json')] = sha256(folder / 'manifest.json')
        mapping = piece['record']['mapping']['line_mapping']['weld_to_sequence']
        line_ids = [i for i, s in enumerate(piece['sequences']) if s.get('curve_kind') != 'arc']
        lines = [w for w in piece['truth'] if all(s['kind'] == 'line' for s in w['segments'])]
        for wi, weld in enumerate(lines):
            if weld['type'] != 'vertical' or str(wi) not in mapping:
                continue
            sequence = piece['sequences'][line_ids[mapping[str(wi)]]]
            normals = sequence['features'][0, 60:66].reshape(2, 3)
            for endpoint, point in zip(['start', 'end'], [weld['segments'][0]['points'][0], weld['segments'][-1]['points'][-1]]):
                point = np.asarray(point)
                local = xyz[tree.query_ball_point(point, 30.)]
                for index, normal in enumerate(normals):
                    face = local[np.abs((local - point) @ normal) <= .05]
                    # Merge coincident export samples before measuring spacing.
                    face = np.unique(np.round(face, 4), axis=0)
                    spacing = cKDTree(face).query(face, k=2)[0][:, 1] if len(face) >= 2 else np.array([])
                    rows.append({'group': group, 'endpoint': endpoint, 'plane': index,
                                 'number': weld['metadata'].get('WeldNumber (f/v + Name number)'),
                                 'unique_face_points': len(face),
                                 'median_spacing_mm': float(np.median(spacing)) if len(spacing) else None,
                                 'p90_spacing_mm': float(np.quantile(spacing, .9)) if len(spacing) else None})
        del piece, xyz, tree
    summary = []
    for group in args.groups:
        for endpoint in ['start', 'end']:
            selected = [r for r in rows if r['group'] == group and r['endpoint'] == endpoint and r['median_spacing_mm'] is not None]
            values = [r['median_spacing_mm'] for r in selected]
            summary.append({'group': group, 'endpoint': endpoint, 'face_neighborhoods': len(selected),
                            'median_of_local_medians_mm': float(np.median(values)) if values else None,
                            'median_range_mm': [min(values), max(values)] if values else None})
    report = {'scope': 'Reference-centered local sampling diagnosis. Sampling differences alone do not establish the cause of model failure.',
              'radius_mm': 30., 'plane_tolerance_mm': .05, 'dedup_rounding_mm': .0001,
              'sources': sources, 'script_sha256': sha256(__file__), 'summary': summary, 'rows': rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sampling.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
