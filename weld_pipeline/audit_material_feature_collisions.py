"""Exact indistinguishable physical feature groups; never change original labels."""
import json
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.signed_material_support import frame_views

DATA=Path('weld_pipeline/data/material_support_v1')
OUT=Path('weld_pipeline/reports/material_feature_collisions_148_20261001.json')


def main():
    if OUT.exists():raise ValueError('Preserve exact feature collision evidence')
    config=json.loads((DATA/'config.json').read_text('utf8'))
    xs=[];control=[];material=[];business=[];records=[]
    for name in config['pieces']:
        root=DATA/name;manifest=json.loads((root/'manifest.json').read_text('utf8'))
        for file,digest in manifest['files'].items():
            if sha256(root/file)!=digest:raise ValueError('Prepared feature evidence changed')
        xs.append(np.load(root/'signed.npy',allow_pickle=False));control.append(np.load(root/'side_sorted.npy',allow_pickle=False))
        with np.load(root/'labels.npz',allow_pickle=False) as targets:
            material.append(targets['material']);business.append(targets['business'])
        records.extend(dict(piece=name,**row) for row in manifest['entries'])
    x=np.concatenate(xs);views=frame_views(x);canonical=np.empty_like(x)
    for i in range(len(x)):canonical[i]=min((v[i] for v in views),key=lambda a:tuple(a))
    ys={'material':np.concatenate(material),'business':np.concatenate(business)}
    results={}
    for name,features in [('signed_eight_frame_orbit',canonical),('side_sorted_control',np.concatenate(control))]:
        packed=np.ascontiguousarray(features).view(np.dtype((np.void,features.shape[1]*features.dtype.itemsize))).ravel()
        _,inverse=np.unique(packed,return_inverse=True)
        groups={}
        for i,g in enumerate(inverse):groups.setdefault(int(g),[]).append(i)
        heads={}
        for target,y in ys.items():
            conflicts=[];lower_bound=0
            for indices in groups.values():
                indices=[i for i in indices if y[i]>=0]
                values,counts=np.unique(y[indices],return_counts=True)
                if len(values)<2:continue
                errors=int(counts.sum()-counts.max());lower_bound+=errors
                conflicts.append(dict(class_values=values.tolist(),class_counts=counts.tolist(),
                    minimum_sample_errors=errors,rows=[dict(records[i],reference_class=int(y[i])) for i in indices]))
            heads[target]=dict(conflicting_exact_feature_groups=len(conflicts),
                minimum_unavoidable_sample_classification_errors=lower_bound,conflicts=conflicts)
        results[name]=dict(unique_equivalence_groups=len(groups),targets=heads)
    final=dict(rows=len(x),results=results,data_config_sha256=sha256(DATA/'config.json'),
        feature_source_sha256=sha256('weld_pipeline/signed_material_support.py'),verifier_sha256=sha256(__file__),
        interpretation='A deterministic invariant classifier cannot separate identical descriptors; this is a descriptor limit, not a proof of full-PCD or model-class limits.',
        original_labels_changed=False,observed_check_read=False,full_contract_pass=False)
    OUT.write_text(json.dumps(final,indent=2,ensure_ascii=False),encoding='utf8')
    for name,row in results.items():
        print('MATERIAL_FEATURE_COLLISIONS',name,{h:(s['conflicting_exact_feature_groups'],s['minimum_unavoidable_sample_classification_errors'])
                                                   for h,s in row['targets'].items()},flush=True)


if __name__=='__main__':main()
