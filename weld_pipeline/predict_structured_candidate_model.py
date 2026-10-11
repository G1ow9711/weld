"""Raw PCD geometry plus structural thickness regression in one checkpoint."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.contract_export import save_children
from weld_pipeline.contract_horizontal_model import from_checkpoint
from weld_pipeline.export import save_all
from weld_pipeline.full_fit_hybrid import infer_from_sequences
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline.point_plate_context import PlateContext
from weld_pipeline.predict_hybrid_with_attributes import export_attributes
from weld_pipeline.structured_plate_attributes import estimate_children, VERSION

MODEL_NAME='full_fit_structural_plate_attributes_v4'


def predict(pcd,checkpoint,outdir):
    torch.set_num_threads(4);begin=time.perf_counter()
    state=torch.load(checkpoint,map_location='cpu')
    if state['model_name']!=MODEL_NAME:
        raise ValueError('Wrong structured candidate model')
    verify_cached_inference_sources(state,checkpoint);verify_cached_inference_sources(state['horizontal'],checkpoint)
    model=from_checkpoint(state['horizontal']).cuda().eval();torch.cuda.synchronize()
    load_seconds=time.perf_counter()-begin
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    xyz=read_xyz_pcd(pcd)*1000.;read_seconds=time.perf_counter()-begin;t=time.perf_counter()
    sequences,geometry=make_mixed_sequences(xyz);context=PlateContext(xyz)
    structural=np.zeros((len(sequences),50),np.float32);process=np.zeros((len(sequences),6),np.float32)
    for i,seq in enumerate(sequences):
        if seq.get('curve_kind')!='arc':
            structural[i],process[i],_=context.describe(seq)
        seq['plate_context']=structural[i]
    feature_seconds=time.perf_counter()-t;t=time.perf_counter()
    pred,diagnostic=infer_from_sequences(state,model,sequences,structural,process)
    attrs=estimate_children(state['attribute_heads'],sequences,pred);torch.cuda.synchronize()
    inference_seconds=time.perf_counter()-t
    (out/'predictions.json').write_text(json.dumps(pred,indent=2),encoding='utf8')
    (out/'vertical_diagnostic.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf8')
    count=save_children(pred,out,'input_'+sha256(pcd)[:12]);save_all(pred,out)
    estimated=export_attributes(out,attrs)
    profile=out/'attribute_profiles.json';data=json.loads(profile.read_text('utf8'))
    data['children']={r['parent_id']+'_'+str(r['child_index']+1):r for r in data['children']}
    profile.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    path=out/'output_schema.json';schema=json.loads(path.read_text('utf8'))
    schema.update(output_schema_version='structured_candidate_attributes_v4',attribute_estimator_version=VERSION,
        attribute_profile_reference='attribute_profiles.json#<weld_id>; keyed child dictionary',
        candidate_scalar_definition='CSV midpoint value; all five along-line samples in the referenced profile.',
        material_direction='Material outward unit normals; right handed side order with the directed child tangent.',
        confidence_definition='Minimum material-direction class confidence over five samples; not calibrated thickness uncertainty.',
        customer_side_schema_frozen=False,full_contract_output_compliant=False)
    schema['columns'].append('attribute_profile_reference');path.write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf8')
    torch.cuda.synchronize()
    receipt={'input':str(pcd),'input_sha256':sha256(pcd),'points':len(xyz),'children':count,
        'attribute_candidate_children':estimated,'checkpoint_sha256':sha256(checkpoint),'single_checkpoint':True,
        'runtime_step_used':False,'runtime_labels_used':False,'preprocessing_cache_used':False,
        'model_load_seconds':load_seconds,'read_seconds':read_seconds,'feature_seconds':feature_seconds,
        'inference_attribute_seconds':inference_seconds,'raw_read_to_all_exports_seconds':time.perf_counter()-begin,
        'geometry':geometry,'speed_gate':'DEFERRED_BY_USER','full_contract_pass':False,
        'attribute_training_truth_reviewed':False,'output_schema_version':'structured_candidate_attributes_v4',
        'source_sha256':sha256(__file__)}
    (out/'inference_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print('STRUCTURED_RAW',json.dumps({k:receipt[k] for k in ['children','attribute_candidate_children','raw_read_to_all_exports_seconds']}),flush=True)
    return pred,receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pcd',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--outdir',required=True)
    args=p.parse_args();predict(args.pcd,args.checkpoint,args.outdir)
