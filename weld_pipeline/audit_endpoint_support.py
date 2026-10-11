"""Reference-centered diagnosis of wall/base support; never an inference input."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.dataio import read_xyz_pcd
from training.labels import sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True)
    ap.add_argument('--numbers', nargs='+', required=True)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args()
    manifest = Path('weld_pipeline/data/mixed_surface_v1') / a.group / 'manifest.json'
    data = json.loads(manifest.read_text(encoding='utf-8'))
    source = Path(data['record']['input'])
    if sha256(source) != data['record']['pcd_sha256']:
        raise ValueError('Source cloud changed')
    xyz = read_xyz_pcd(source) * 1000
    records = []
    for weld in data['truth']:
        number = weld['metadata']['WeldNumber (f/v + Name number)']
        if number not in a.numbers:
            continue
        if weld['type'] != 'horizontal' or len(weld['segments']) != 1 or weld['segments'][0]['kind'] != 'line':
            raise ValueError('This diagnosis requires a single horizontal line')
        p = np.array(weld['segments'][0]['points'])[[0, -1]]
        u = p[1] - p[0]
        u /= np.linalg.norm(u)
        normal = np.cross(u, [0., 0., 1.])
        rows = []
        for endpoint, q in enumerate(p):
            delta = xyz - q
            t, cross, z = delta @ u, delta @ normal, delta[:, 2]
            area = (abs(t) < 180) & (abs(cross) < 100) & (abs(z) < 180)
            t, cross, z = t[area], cross[area], z[area]
            for along in range(-150, 151, 30):
                wall = (abs(cross) < .15) & (abs(t - along) < 5) & (abs(z) > 3)
                base = (abs(z) < .15) & (abs(t - along) < 5) & (abs(cross) > 3)
                rows.append({'endpoint': endpoint, 'along_mm': along, 'wall_points': int(wall.sum()),
                             'wall_z_range': [float(z[wall].min()), float(z[wall].max())] if wall.any() else None,
                             'base_points': int(base.sum()),
                             'base_cross_range': [float(cross[base].min()), float(cross[base].max())] if base.any() else None})
        records.append({'number': number, 'endpoints_mm': p.tolist(), 'metadata': weld['metadata'], 'bins': rows})
    if len(records) != len(set(a.numbers)):
        raise ValueError('Requested weld not found')
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    report = {'group': a.group, 'scope': 'Reference-centered point geometry diagnosis only; not predicted endpoints or an acceptance result.',
              'source_cloud_sha256': sha256(source), 'manifest_sha256': sha256(manifest),
              'source_code_sha256': sha256(__file__), 'rows': records}
    (out / 'endpoint_support.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'group': a.group, 'welds': len(records), 'output': str(out / 'endpoint_support.json')}))


if __name__ == '__main__':
    main()
