"""Geometry-only near-duplicate screening before freezing the pilot split."""
import sys, json, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from training.labels import sha256


def canonical(x):
    x = x-x.mean(axis=0)
    _, axes = np.linalg.eigh(x.T@x)
    return x@axes[:, ::-1]


def main():
    base = ROOT/'training/data/pilot_v1'
    manifest = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    geometry = {g['name']: canonical(np.load(base/g['name']/'geometry_sample.npz')['xyz']) for g in manifest['groups']}
    records = []
    for a, b in itertools.combinations(geometry, 2):
        x, y = geometry[a], geometry[b]
        extx, exty = np.ptp(x,axis=0), np.ptp(y,axis=0)
        relative = float(np.max(np.abs(extx-exty)/np.maximum(extx,exty)))
        if relative > .30:
            records.append({'a':a,'b':b,'extent_relative_difference':relative,'screen':'different_extent'}); continue
        best = None
        for signs in itertools.product([-1,1],repeat=3):
            z = y*np.asarray(signs)
            for _ in range(8):
                d, idx = cKDTree(x).query(z,workers=2)
                keep = d <= np.quantile(d,.85)
                zz, xx = z[keep], x[idx[keep]]
                zc, xc = zz.mean(axis=0), xx.mean(axis=0)
                u, _, vt = np.linalg.svd((zz-zc).T@(xx-xc))
                rot = u@vt
                z = (z-zc)@rot+xc
            ds = np.r_[cKDTree(x).query(z,workers=2)[0],cKDTree(z).query(x,workers=2)[0]]
            score = float(np.quantile(ds,.90))
            if best is None or score < best['p90_mm']:
                best={'p50_mm':float(np.median(ds)),'p90_mm':score,'coverage_40mm':float(np.mean(ds<=40)), 'coverage_80mm':float(np.mean(ds<=80))}
        records.append({'a':a,'b':b,'extent_relative_difference':relative,**best,
            'screen':'near_shape' if relative<.12 and best['p90_mm']<80 else 'not_near_at_screen_resolution'})
    (base/'geometry_pairs.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in records:
        if r['screen']!='different_extent': print(json.dumps(r,ensure_ascii=False),flush=True)
    print('Manifest SHA256:',sha256(base/'manifest.json'))


if __name__=='__main__': main()
