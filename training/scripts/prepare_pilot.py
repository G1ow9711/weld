"""Audit all labeled workpieces and generate traceable pilot arrays. Raw data read-only."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from training.dataio import read_xyz_pcd
from training.labels import read_curves, read_excel, point_labels, sha256


def main():
    out = ROOT/'training/data/pilot_v1'
    out.mkdir(parents=True, exist_ok=True)
    frozen = ROOT/'training/configs/pilot_split.frozen.json'
    if frozen.exists():
        split = json.loads(frozen.read_text(encoding='utf-8'))
        assert sha256(out/'manifest.json') == split['manifest_sha256'], 'Frozen manifest changed; use a new data version.'
        saved = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        for group in saved['groups']:
            assert all(sha256(ROOT/p)==h for p,h in group['source_hashes'].items()), 'Raw source changed; use a new data version.'
            assert all((out/group['name']/f).exists() for f in ['xyz_world_mm.npy','labels.npy','distance_mm.npy','instance.npy','voxel_indices.npy','voxel_grid.npy','inverse.npy','voxel_labels.npy','voxel_weights.npy'])
        print('FROZEN_DATASET_VERIFIED: existing pilot arrays retained; create a new version for changes.',flush=True)
        return
    report = {'version': 'pilot_v1', 'pcd_scale_to_mm': 1000, 'voxel_mm': 2,
        'positive_band_mm': 4, 'background_min_mm': 6,
        'unit_basis': 'PCD unit not declared; 1000 scaling checked against every JSON exportToWorld and Excel source endpoint.',
        'scope': 'Customer-reference-derived labels; no customer signoff or final weld acceptance.',
        'groups': [], 'pending': []}
    for group in sorted((ROOT/'DataSet/9月20日').iterdir()):
        if not group.is_dir(): continue
        jf = list(group.glob('*_welds.json'))
        if not jf:
            report['pending'].append(group.name); continue
        target = out/group.name
        target.mkdir(exist_ok=True)
        cached = target/'audit.json'
        if cached.exists() and (target/'voxel_grid.npy').exists():
            prior = json.loads(cached.read_text(encoding='utf-8'))
            if all(sha256(ROOT/p)==h for p,h in prior['source_hashes'].items()):
                report['groups'].append(prior)
                print('CACHED', group.name, flush=True); continue
        start = time.perf_counter()
        pcd = next(group.glob('*_AllFaces.pcd'))
        xf = next(group.glob('*.xlsx'))
        xyz = read_xyz_pcd(pcd)*1000
        curves = read_curves(jf[0])
        excel = read_excel(xf)
        assert {w['id'] for w in curves['welds']} == set(excel)
        endpoint_errors = []
        for w in curves['welds']:
            refs = excel[w['id']]
            endpoints = np.array([p for s in w['segments'] for p in [s['start_world_mm'], s['end_world_mm']]])
            endpoint_errors.extend(cKDTree(endpoints).query(np.array([refs['start'], refs['end']]))[0].tolist())
            w['number'], w['type'] = refs['number'], refs['type']
        assert max(endpoint_errors) < 0.05, (group.name, max(endpoint_errors))
        tree = cKDTree(xyz)
        curve_distance = tree.query(curves['samples'], workers=2)[0]
        per_segment = []
        for si in range(len(curves['starts'])):
            ds = curve_distance[curves['sample_segment'] == si]
            per_segment.append({'segment': si, 'weld': int(curves['instance'][si]),
                'coverage_4mm': float(np.mean(ds <= 4)), 'coverage_6mm': float(np.mean(ds <= 6)),
                'p95_mm': float(np.quantile(ds, .95)), 'max_mm': float(ds.max())})
        rng = np.random.default_rng(42)
        spacing = tree.query(xyz[rng.choice(len(xyz), min(4096, len(xyz)), replace=False)], k=2, workers=2)[0][:, 1]
        labels, distance, instance, checks = point_labels(xyz, curves)
        origin = xyz.min(axis=0)
        grid = np.floor((xyz-origin)/2).astype(np.int32)
        _, first, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
        counts = np.bincount(inverse)
        # Mixed foreground/background voxels are ignored; never silently majority-vote a seam away.
        fg = np.bincount(inverse, weights=(labels==1), minlength=len(first))
        bg = np.bincount(inverse, weights=(labels==0), minlength=len(first))
        vl = labels[first].copy()
        vl[(fg>0) & (bg>0)] = -1
        np.save(target/'xyz_world_mm.npy', xyz.astype(np.float32))
        np.save(target/'labels.npy', labels)
        np.save(target/'distance_mm.npy', distance)
        np.save(target/'instance.npy', instance)
        np.save(target/'voxel_indices.npy', first.astype(np.int32))
        np.save(target/'voxel_grid.npy', grid[first])
        np.save(target/'inverse.npy', inverse.astype(np.int32))
        np.save(target/'voxel_labels.npy', vl)
        np.save(target/'voxel_weights.npy', counts.astype(np.int32))
        np.savez(target/'geometry_sample.npz', xyz=xyz[rng.choice(len(xyz), min(8000, len(xyz)), replace=False)])
        (target/'curves_world_mm.json').write_text(json.dumps(curves['welds'], ensure_ascii=False), encoding='utf-8')
        audit = {'name': group.name, 'input': str(pcd.relative_to(ROOT)),
            'source_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in [pcd, jf[0], xf]},
            'export_to_world': curves['matrix'].tolist(), 'source_points': len(xyz), 'voxel_points': len(first),
            'bbox_mm': np.ptp(xyz,axis=0).tolist(), 'grid_origin_mm': origin.tolist(),
            'reference_welds': len(curves['welds']), 'reference_segments': len(curves['starts']),
            'excel_json_endpoint_max_mm': max(endpoint_errors),
            'curve_to_cloud_mm': {str(q): float(np.quantile(curve_distance,q)) for q in [.5,.9,.95,.99,1]},
            'curve_coverage_4mm': float(np.mean(curve_distance<=4)),
            'curve_coverage_6mm': float(np.mean(curve_distance<=6)),
            'point_spacing_mm': {str(q): float(np.quantile(spacing,q)) for q in [.1,.5,.9]},
            'positive_points_by_band': {str(b): int((distance<=b).sum()) for b in [2,4,6]},
            'label_counts': {str(i): int((labels==i).sum()) for i in [-1,0,1]},
            'voxel_label_counts': {str(i): int((vl==i).sum()) for i in [-1,0,1]},
            'candidate_checks': checks, 'segments': per_segment,
            'preprocess_seconds': time.perf_counter()-start}
        cached.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
        report['groups'].append(audit)
        print(f"AUDITED {group.name}: {len(xyz)} -> {len(first)} voxels; foreground={(vl==1).sum()}; curve coverage@4mm={audit['curve_coverage_4mm']:.3f}; p95={audit['curve_to_cloud_mm']['0.95']:.2f}mm; {audit['preprocess_seconds']:.1f}s", flush=True)
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED',out/'manifest.json',flush=True)


if __name__ == '__main__': main()
