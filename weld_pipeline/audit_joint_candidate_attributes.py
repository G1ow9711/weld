"""Conservative geometry-inclusive candidate comparison, explicitly not E2."""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.learned_plate_attributes import estimate_children

FIELDS=['thickness_1','thickness_2','normal_1','normal_2']


def main():
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    geometry=Path('weld_pipeline/runs/contract_full_fit_hybrid_v1')
    refs=Path('weld_pipeline/reports/step_contact_continuity_candidates_v1')
    attrs=Path('weld_pipeline/runs/expanded_plate_attributes_seed132')
    bundle=joblib.load(attrs/'heads.joblib')
    out=Path('weld_pipeline/reports/joint_candidate_attributes_20260930');out.mkdir(parents=True,exist_ok=True)
    aggregate={};start=time.perf_counter()
    for role,names in [('FIT',[n for n in split['train'] if '__' not in n]),('UNSEEN_CHECK',split['validation'])]:
        all_counts={k:0 for k in FIELDS};g=0;coverage=0;valid_total=0;pieces=[]
        for name in names:
            piece=load_piece(split['piece_roots'][name],verify=True)
            cached=json.loads((geometry/(role+'_'+name+'.json')).read_text('utf8'))
            pred,score=cached['predictions'],cached['score']
            attributes=estimate_children(bundle,piece['sequences'],pred)
            indexed={(r['parent_id'],r['child_index']):r for r in attributes}
            path=refs/(name+'_attribute_candidates.json')
            reference=json.loads(path.read_text('utf8'))['children'] if path.exists() else []
            reference={(r['weld_id'],r['child_index']):r for r in reference if r['state']=='geometric_candidate_all_samples'}
            counts={k:0 for k in FIELDS};comparisons=[]
            for match in score['matches']:
                wanted_parent,wanted_index=match['truth_id'].rsplit(':',1)
                got_parent,got_index=match['prediction_id'].rsplit(':',1)
                ref=reference.get((wanted_parent,int(wanted_index)))
                got=indexed.get((got_parent,int(got_index)))
                flags={k:False for k in FIELDS};errors=[]
                if ref and got and got['state']=='estimated_candidate':
                    samples=got['samples'][::-1] if match['reversed'] else got['samples']
                    for observed,wanted in zip(samples,ref['samples']):
                        order=[1,0] if match['reversed'] else [0,1]
                        t=np.array(observed['thickness_mm'])[order]
                        n=np.array(observed['material_outward_normals'])[order]
                        pair=wanted['candidate_pairs'][0]
                        errors.append({'thickness':[abs(t[i]-pair[i]['thickness_candidate_mm']) for i in [0,1]],
                            'normal':[float(np.degrees(np.arccos(np.clip(n[i]@np.array(pair[i]['outward_normal_world']),-1.,1.)))) for i in [0,1]]})
                    for side in [0,1]:
                        flags['thickness_'+str(side+1)]=all(e['thickness'][side]<=1. for e in errors)
                        flags['normal_'+str(side+1)]=all(e['normal'][side]<=3. for e in errors)
                for key in FIELDS:
                    counts[key]+=int(flags[key])
                comparisons.append({'truth_id':match['truth_id'],'prediction_id':match['prediction_id'],
                    'reverse_adjustment':match['reversed'],'complete_reference':ref is not None,
                    'attribute_candidate_state':got['state'] if got else 'unknown','flags':flags,'errors':errors})
            for key in FIELDS:
                all_counts[key]+=counts[key]
            g+=score['truth_count'];coverage+=len(reference);valid_total+=score['valid_children']
            row={'name':name,'role':role,'truth_count':score['truth_count'],'geometry_valid':score['valid_children'],
                'complete_reference_count':len(reference),'joint_candidate_counts':counts,'comparisons':comparisons,
                'unknown_reference_or_attribute_and_geometry_miss_not_counted_as_success':True,
                'reviewed_customer_truth':False,'full_contract_pass':False}
            (out/(role+'_'+name+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf8')
            pieces.append({k:v for k,v in row.items() if k!='comparisons'})
            print('JOINT_REFERENCE',name,counts,flush=True)
        aggregate[role]={'truth_count':g,'geometry_valid':valid_total,'complete_candidate_reference_count':coverage,
            'joint_candidate_counts':all_counts,'conservative_diagnostic_rates_all_children':{k:all_counts[k]/g for k in FIELDS},
            'pieces':pieces,'reviewed_customer_truth':False,'full_contract_pass':False,
            'scope':'Conservative diagnostic; numerator requires valid geometry and every sampled candidate field. Unreviewed reference and missing reference cannot provide formal E2 acceptance.'}
    report={'roles':aggregate,'elapsed_seconds':time.perf_counter()-start,'source_sha256':sha256(__file__),
        'attribute_head_sha256':sha256(attrs/'heads.joblib'),'full_contract_pass':False}
    (out/'complete.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('JOINT_CANDIDATE_COMPLETE',json.dumps({k:v['joint_candidate_counts'] for k,v in aggregate.items()}),flush=True)


if __name__=='__main__':
    main()
