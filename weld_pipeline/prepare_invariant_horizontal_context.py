"""No labels are used to calculate point centroid/covariance."""
import json
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256


def main():
    source=Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split=json.loads(source.read_text('utf8'))
    out=Path('weld_pipeline/data/invariant_horizontal_context_v1')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        raise ValueError('Frozen centroid/covariance context exists')
    names=[n for n in split['train'] if '__' not in n]+split['validation']
    rows=[]
    for name in names:
        manifest=json.loads((Path(split['piece_roots'][name])/'manifest.json').read_text('utf8'))
        source_pcd=Path(manifest['record']['input'])
        if sha256(source_pcd)!=manifest['record']['pcd_sha256']:
            raise ValueError('PCD changed')
        xyz=read_xyz_pcd(source_pcd)*1000.
        mean=xyz.mean(0);centered=xyz-mean
        covariance=centered.T@centered/len(xyz)
        np.savez(out/(name+'.npz'),mean=mean,covariance=covariance)
        rows.append({'name':name,'pcd_sha256':sha256(source_pcd),'statistics_sha256':sha256(out/(name+'.npz'))})
    (out/'complete.json').write_text(json.dumps({'parent_split_sha256':sha256(source),
        'sources':{p:sha256(p) for p in [__file__,'weld_pipeline/invariant_horizontal_context.py']},
        'rows':rows,'point_only':True},ensure_ascii=False,indent=2),encoding='utf8')
    print('INVARIANT_POINT_CONTEXT_READY',len(rows),flush=True)


if __name__=='__main__':
    main()
