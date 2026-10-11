"""FIT-only comparisons establish measurement failure modes, never predictions."""
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.endpoint_plate_observations import EndpointPlanes,physical_height_candidates,VERSION
from weld_pipeline.prepare_contract_child import child_attributes
from weld_pipeline.train_candidate_tree import mapped_references
from weld_pipeline.train_endpoint_process_attributes import known_label


def main():
    split_path=Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split=json.loads(split_path.read_text('utf8'));names=[n for n in split['train'] if '__' not in n]
    out=Path('weld_pipeline/reports/endpoint_height_physics_20260930');out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        raise ValueError('Frozen physics audit exists')
    summaries=[];combined=defaultdict(lambda:Counter());hashes={}
    for name in names:
        piece=load_piece(split['piece_roots'][name],verify=True)
        path=piece['record']['input'];assert sha256(path)==piece['record']['pcd_sha256']
        xyz=read_xyz_pcd(path)*1000.;context=EndpointPlanes(xyz);rows=[]
        mapping={w['id']:i for i,refs in mapped_references(piece).items() for w in refs}
        for weld in piece['truth']:
            if weld['id'] not in mapping:
                continue
            seq=piece['sequences'][mapping[weld['id']]]
            normals=seq['features'][len(seq['features'])//2,60:66].reshape(2,3)
            for ci,segment in enumerate(weld['segments']):
                attrs=child_attributes(weld,ci)
                for field,index in [('start_height',0),('end_height',-1)]:
                    target=known_label(field,attrs[field])
                    if target is None:
                        continue
                    q=np.asarray(segment['points'][index]);obs=context.observe(q,normals)
                    candidates=physical_height_candidates(obs)
                    error={k:abs(v-target['value']) for k,v in candidates.items()}
                    rows.append({'parent':weld['id'],'child':ci,'type':weld['type'],'field':field,
                        'target_mm':target['value'],'xyz_mm':q.tolist(),'observations':obs,'candidates_mm':candidates,'error_mm':error})
                    for key,e in error.items():
                        combined[weld['type']][key+'_known']+=1;combined[weld['type']][key+'_within_3']+=e<=3.
        summary={'name':name,'endpoints':len(rows),'target_range':([min(r['target_mm'] for r in rows),max(r['target_mm'] for r in rows)] if rows else None),
            'min_oracle_candidate_within_3':sum(bool(r['error_mm']) and min(r['error_mm'].values())<=3. for r in rows),
            'diagnostic_oracle_only':True,'full_contract_pass':False}
        summaries.append(summary);hashes[name]=piece['record']['pcd_sha256']
        (out/(name+'.json')).write_text(json.dumps({'summary':summary,'rows':rows},indent=2,ensure_ascii=False),encoding='utf8')
        print('FIT_PHYSICS',name,json.dumps(summary,ensure_ascii=False),flush=True)
    report={'version':VERSION,'fit_sources':names,'split_sha256':sha256(split_path),'pcd_hashes':hashes,'pieces':summaries,
        'fixed_candidate_counts_by_type':dict(combined),'untrained_or_test_labels_used':False,
        'sources':{p:sha256(p) for p in [__file__,'weld_pipeline/endpoint_plate_observations.py']},'full_contract_pass':False}
    (out/'complete.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('HEIGHT_PHYSICS_COMPLETE',sum(r['endpoints'] for r in summaries),flush=True)


if __name__=='__main__':
    main()
