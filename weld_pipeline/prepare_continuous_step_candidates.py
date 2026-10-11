"""Resolve endpoint contact ambiguity only through common CAD face identities."""
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from training.labels import sha256


def identity(pair):
    return tuple((int(p['volume']),*np.round(p['outward_normal_world'],6)) for p in pair)


def resolve(child):
    result=deepcopy(child)
    tables=[{identity(pair):pair for pair in s['candidate_pairs']} for s in child['samples']]
    common=set.intersection(*(set(t) for t in tables))
    if len(common)==1:
        key=next(iter(common))
        for sample,table in zip(result['samples'],tables):
            sample['raw_candidate_pairs']=sample['candidate_pairs']
            sample['raw_unique_pair']=sample['unique_pair']
            sample['candidate_pairs']=[table[key]];sample['unique_pair']=True
        result['state']='geometric_candidate_all_samples'
        result['pair_at_mid_or_nearest_sample']=tables[2][key]
        result['resolution']='UNIQUE_SHARED_SOLID_AND_OUTWARD_FACE_NORMAL_IDENTITIES_AT_ALL_FIVE_SAMPLES'
    else:
        result['resolution']='NO_UNIQUE_COMMON_FACE_IDENTITY_AT_ALL_SAMPLES'
    result['reviewed_customer_truth']=False
    return result


def main():
    source=Path('weld_pipeline/reports/all_step_references_20260930')
    receipt=json.loads((source/'complete.json').read_text('utf8'))
    out=Path('weld_pipeline/reports/step_contact_continuity_candidates_v1');out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        raise ValueError('Frozen continuity references already exist')
    summaries=[];csv_rows=[]
    for piece in receipt['pieces']:
        name=piece['name'];path=source/(name+'_attribute_candidates.json')
        ref=json.loads(path.read_text('utf8'))
        repair=Path('weld_pipeline/reports/step_reference_pose_repair_v1')/(name+'_attribute_candidates.json')
        if not ref['pose_verified_for_reference'] and repair.exists():
            repaired=json.loads(repair.read_text('utf8'))
            if repaired['pose_verified_for_reference']:
                ref=repaired
                ref['repair_reference_sha256']=sha256(repair)
        if not ref['pose_verified_for_reference']:
            summaries.append({'name':name,'status':'UNRESOLVED_REFERENCE_POSE','children':0,'complete_candidates':0})
            continue
        records=[resolve(r) for r in ref['children']]
        report={**ref,'children':records,'parent_reference_sha256':sha256(path),'resolution_source_sha256':sha256(__file__),
            'all_five_sample_unique_consistent':sum(r['state']=='geometric_candidate_all_samples' for r in records),
            'independent_customer_review':False,'full_contract_pass':False}
        (out/(name+'_attribute_candidates.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        summaries.append({'name':name,'children':len(records),'complete_candidates':report['all_five_sample_unique_consistent'],
                          'original_complete_candidates':ref['all_five_sample_unique_consistent']})
        for r in records:
            pair=r['pair_at_mid_or_nearest_sample']
            row={'piece':name,'parent_id':r['weld_id'],'child_index':r['child_index'],'weld_number':r['weld_number'],
                 'type':r['type'],'reference_state':r['state'],'resolution':r['resolution'],'reviewed_customer_truth':False}
            for i in [0,1]:
                side=pair[i] if pair else {}
                row.update({f'board_{i+1}_candidate':side.get('board_name'),
                    f'thickness_{i+1}_candidate_mm':side.get('thickness_candidate_mm'),
                    f'normal_{i+1}_outward_candidate':json.dumps(side.get('outward_normal_world')),
                    f'thickness_{i+1}_reviewed_mm':'',f'normal_{i+1}_reviewed_xyz':''})
            csv_rows.append(row)
    with (out/'STEP_板厚与法向候选审核样表.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    (out/'complete.json').write_text(json.dumps({'pieces':summaries,'sources':{__file__:sha256(__file__)},
        'parent_receipt_sha256':sha256(source/'complete.json'),'reviewed_customer_truth':False,'full_contract_pass':False,
        'scope':'Reference candidates only. Endpoints resolved only by one common material-face identity pair over all five samples.'},
        ensure_ascii=False,indent=2),encoding='utf8')
    print('CONTINUOUS_STEP_REFERENCES',json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
