"""Read-back checks of every child value, unknown, curve and profile reference."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from weld_pipeline.contract_export import ATTRIBUTES


def audit(out):
    out=Path(out);pred=json.loads((out/'predictions.json').read_text('utf8'))
    curves=json.loads((out/'child_curves.json').read_text('utf8'))
    profiles=json.loads((out/'attribute_profiles.json').read_text('utf8'))['children']
    with (out/'predicted_children.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    expected=[(w,i,s) for w in pred for i,s in enumerate(w['segments'])]
    assert len(rows)==len(expected)==len(curves)==len(profiles)
    estimated=0
    for row,(w,i,s) in zip(rows,expected):
        key=w['id']+'_'+str(i+1);p=np.array(s['points'])
        assert row['weld_id']==key and row['parent_id']==w['id'] and int(row['segment_index'])==i
        assert row['curve_reference']=='child_curves.json#'+key
        assert row['attribute_profile_reference']=='attribute_profiles.json#'+key
        assert curves[key]['ordered_points_mm']==s['points']
        np.testing.assert_allclose([float(row[k]) for k in ['start_x','start_y','start_z','end_x','end_y','end_z']],
            np.r_[p[0],p[-1]],rtol=0,atol=1e-8)
        assert abs(float(row['length_mm'])-np.linalg.norm(np.diff(p,axis=0),axis=1).sum())<1e-8
        prof=profiles[key]
        if prof['state']=='estimated_candidate':
            estimated+=1;assert len(prof['samples'])==5
            assert prof['sample_fractions']==[0.,.25,.5,.75,1.]
            tangent=p[-1]-p[0];tangent/=np.linalg.norm(tangent)
            for sample in prof['samples']:
                assert sample['state']=='estimated_candidate' and sample['confidence']>=.8
                n=np.array(sample['material_outward_normals']);t=np.array(sample['thickness_mm'])
                assert n.shape==(2,3) and t.shape==(2,) and np.isfinite(n).all() and np.isfinite(t).all() and (t>0).all()
                assert np.abs(np.linalg.norm(n,axis=1)-1.).max()<=1e-3
                assert np.cross(n[0],n[1])@tangent>0.
            for side in [1,2]:
                for field in ['thickness','normal']:
                    prefix=field+'_'+str(side)
                    assert row[prefix+'_status']=='estimated_candidate' and row[prefix+'_source']
                    value=json.loads(row[prefix+'_value'])
                    wanted=prof['samples'][2]['thickness_mm' if field=='thickness' else 'material_outward_normals'][side-1]
                    np.testing.assert_allclose(value,wanted,atol=1e-10,rtol=0.)
        for field in ATTRIBUTES:
            if row[field+'_status']=='unknown':
                assert row[field+'_value']=='' and row[field+'_reason']
            else:
                assert field in ['thickness_1','thickness_2','normal_1','normal_2']
        assert row['weld_side']=='unknown'
    result={'status':'PASS_CHILD_COORDINATES_LENGTHS_UNKNOWN_STATES_AND_PROFILE_REFERENCES',
            'children':len(rows),'estimated_attribute_candidates':estimated,'full_contract_output_compliant':False}
    (out/'child_attribute_export_qa.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--outdir',required=True);args=parser.parse_args()
    print(json.dumps(audit(args.outdir)),flush=True)
