"""Compare point-only estimates to independently CAD-derived candidates."""
import json
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.point_plate_attributes import PointPlateAttributes
from weld_pipeline.truth import load_truth


def main():
    root=Path('weld_pipeline/reports/step_alignment_20260930')
    for name in ['224-GR24A1','527-FR146A']:
        folder=Path('DataSet/9月20日')/name
        pcd=next(folder.glob('*AllFaces.pcd'))
        xyz=read_xyz_pcd(pcd)*1000.
        estimator=PointPlateAttributes(xyz)
        reference_file=root/(name+'_attribute_candidates.json')
        reference=json.loads(reference_file.read_text('utf8'))
        refs={(r['weld_id'],r['child_index']):r for r in reference['children']}
        truth,excluded=load_truth(folder)
        rows=[]
        for w in truth:
            for i,segment in enumerate(w['segments']):
                if segment['kind']!='line':
                    continue
                measured=estimator.line(segment['points'])
                ref=refs[w['id'],i]
                comparisons=[]
                for m,r in zip(measured['samples'],ref['samples']):
                    sides=[]
                    if m['both_estimated'] and r['unique_pair']:
                        for got,wanted in zip(m['sides'],r['candidate_pairs'][0]):
                            thickness_error=abs(got['thickness_mm']-wanted['thickness_candidate_mm'])
                            n=np.array(got['normal_material_outward']);target=np.array(wanted['outward_normal_world'])
                            angle=float(np.degrees(np.arccos(np.clip(n@target,-1.,1.))))
                            sides.append({'thickness_error_mm':thickness_error,'directed_normal_error_deg':angle,
                                          'thickness_pass':thickness_error<=1.,'normal_pass':angle<=3.})
                    comparisons.append({'fraction':m['fraction'],'side_errors':sides})
                full_ref=ref['state']=='geometric_candidate_all_samples'
                flags={f'{field}_{s+1}':bool(full_ref and len(comparisons)==5 and
                       all(len(c['side_errors'])==2 and c['side_errors'][s][field+'_pass'] for c in comparisons))
                       for field in ['thickness','normal'] for s in [0,1]}
                rows.append({'parent':w['id'],'child':i,'type':w['type'],'point_estimate':measured,
                             'derived_reference_full':full_ref,'comparison':comparisons,'all_samples_pass':flags})
        comparable=[r for r in rows if r['derived_reference_full']]
        report={'piece':name,'input_sha256':sha256(pcd),'derived_reference_sha256':sha256(reference_file),
                'point_estimator_source_sha256':sha256('weld_pipeline/point_plate_attributes.py'),
                'scope':'Known geometry positions are used only for attribute audit. Estimates use XYZ alone. Derived CAD candidates are not reviewed customer truth. This is not E2 acceptance.',
                'children':len(rows),'complete_derived_reference_children':len(comparable),
                'comparable_pass_counts':{k:sum(r['all_samples_pass'][k] for r in comparable)
                    for k in ['thickness_1','thickness_2','normal_1','normal_2']},
                'full_contract_pass':False,'results':rows}
        (root/(name+'_point_attribute_comparison_v2.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        print('POINT_ATTRIBUTES',name,len(rows),'comparable',len(comparable),report['comparable_pass_counts'],flush=True)


if __name__=='__main__':
    main()
