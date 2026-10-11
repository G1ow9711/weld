"""One checkpoint, raw XYZ geometry and partial learned attribute outputs."""
import argparse
import csv
import json
import time
from pathlib import Path

import torch

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.contract_export import save_children
from weld_pipeline.contract_horizontal_model import from_checkpoint
from weld_pipeline.export import save_all
from weld_pipeline.full_fit_hybrid import infer_from_sequences,point_structure
from weld_pipeline.learned_plate_attributes import estimate_children
from weld_pipeline.mixed_sequences import make_mixed_sequences

MODEL_NAME='full_available_fit_hybrid_attributes_v1'


def export_attributes(out,attributes):
    profile=out/'attribute_profiles.json'
    profile.write_text(json.dumps({'source':'Learned point profile estimator trained on unreviewed CAD-derived candidates',
        'side_convention':'cross(normal_1,normal_2) dot directed weld tangent positive; customer freezing pending',
        'samples':'Five fractions 0,0.25,0.5,0.75,1 of every predicted straight child',
        'full_contract_pass':False,'children':attributes},ensure_ascii=False,indent=2),encoding='utf8')
    indexed={(r['parent_id'],r['child_index']):r for r in attributes}
    path=out/'子段结果.csv'
    # Locate the current child CSV by its exact header, keeping ordinary weld
    # and legacy attribute workbooks untouched.
    if not path.exists():
        candidates=[p for p in out.glob('*.csv') if p.name not in ['预测焊缝.csv']]
        path=next(p for p in candidates if 'segment_index' in p.read_text('utf-8-sig').splitlines()[0])
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);rows=list(reader);fields=reader.fieldnames
    fields=list(fields)+['attribute_profile_reference']
    estimated=0
    for row in rows:
        r=indexed.get((row['parent_id'],int(row['segment_index'])))
        row['attribute_profile_reference']='attribute_profiles.json#'+row['weld_id']
        if r and r['state']=='estimated_candidate':
            estimated+=1
            mid=r['samples'][2]
            for i in [0,1]:
                for attribute,value in [('thickness',mid['thickness_mm'][i]),('normal',mid['material_outward_normals'][i])]:
                    prefix=attribute+'_'+str(i+1)
                    row[prefix+'_value']=json.dumps(value)
                    row[prefix+'_status']='estimated_candidate'
                    row[prefix+'_source']=mid['source']
                    row[prefix+'_confidence']=min(s['confidence'] for s in r['samples'])
                    row[prefix+'_reason']='CUSTOMER_SIDE_CONVENTION_AND_INDEPENDENT_ATTRIBUTE_ACCEPTANCE_PENDING'
            row['status']='geometry_and_partial_attribute_prediction'
            row['reason_code']='REMAINING_ATTRIBUTES_AND_SIDE_SCHEMA_PENDING'
        else:
            for i in [0,1]:
                for attribute in ['thickness','normal']:
                    row[attribute+'_'+str(i+1)+'_reason']=r.get('reason','LOW_CONFIDENCE_ALONG_LINE') if r else 'CONTACT_CANDIDATE_NOT_ASSOCIATED'
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return estimated


def predict(pcd,checkpoint,outdir):
    torch.set_num_threads(4)
    started=time.perf_counter()
    state=torch.load(checkpoint,map_location='cpu')
    if state['model_name']!=MODEL_NAME:
        raise ValueError('Wrong attribute checkpoint')
    verify_cached_inference_sources(state,checkpoint)
    verify_cached_inference_sources(state['horizontal'],checkpoint)
    model=from_checkpoint(state['horizontal']).cuda().eval()
    torch.cuda.synchronize();load_seconds=time.perf_counter()-started
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();xyz=read_xyz_pcd(pcd)*1000.
    read_seconds=time.perf_counter()-started
    before=time.perf_counter();sequences,geometry=make_mixed_sequences(xyz)
    structural,process=point_structure(xyz,sequences)
    feature_seconds=time.perf_counter()-before
    before=time.perf_counter();pred,diagnostic=infer_from_sequences(state,model,sequences,structural,process)
    attributes=estimate_children(state['attribute_heads'],sequences,pred)
    torch.cuda.synchronize();inference_seconds=time.perf_counter()-before
    (out/'predictions.json').write_text(json.dumps(pred,indent=2),encoding='utf8')
    (out/'vertical_diagnostic.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf8')
    count=save_children(pred,out,'input_'+sha256(pcd)[:12]);save_all(pred,out)
    estimated=export_attributes(out,attributes)
    torch.cuda.synchronize()
    receipt={'input':str(pcd),'input_sha256':sha256(pcd),'points':len(xyz),'children':count,
        'attribute_candidate_children':estimated,'single_checkpoint':True,'runtime_step_used':False,
        'runtime_labels_used':False,'preprocessing_cache_used':False,'checkpoint_sha256':sha256(checkpoint),
        'model_load_seconds':load_seconds,'read_seconds':read_seconds,'feature_seconds':feature_seconds,
        'inference_attribute_seconds':inference_seconds,'raw_read_to_all_exports_seconds':time.perf_counter()-started,
        'geometry':geometry,'speed_gate':'DEFERRED_BY_USER','full_contract_pass':False,
        'attribute_training_truth_reviewed':False,'source_sha256':sha256(__file__)}
    (out/'inference_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print('RAW_ATTRIBUTE_HYBRID',json.dumps({k:receipt[k] for k in ['points','children','attribute_candidate_children','raw_read_to_all_exports_seconds']}),flush=True)
    return pred,receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pcd',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--outdir',required=True)
    args=p.parse_args();predict(args.pcd,args.checkpoint,args.outdir)
