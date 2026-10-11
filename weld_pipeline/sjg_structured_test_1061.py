"""Label-free final checkpoint inference, or separate frozen-output scoring."""
import argparse,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources

def infer(run,training,ablate_process=False):
    import torch
    from weld_pipeline.sjg_contact_tcn_1052 import neighbours
    from weld_pipeline.sjg_structured_contact_1057 import StructuredContactTCN as ContactTCN,decode
    from weld_pipeline.sjg_hierarchical_infer_1032 import representation
    dest=OUT/run;dest.mkdir(exist_ok=True);folder=dest/'source_21';folder.mkdir(exist_ok=False);start=time.perf_counter();check_resources()
    torch.set_num_threads(2);base=OUT/training;receipt=read(base/'receipt.json');assert receipt['status']=='CONTACT_TRAIN_COMPLETE'
    assert digest(base/'final.pt')==receipt['checkpoint_sha256'];checkpoint=torch.load(base/'final.pt',map_location='cpu')
    data=OUT/checkpoint['data_run'];freeze=read(data/'input_freeze.json');assert freeze['target_reads']==0
    for name,sha in freeze['files'].items():assert digest(data/name)==sha
    assert digest(base/'normalization.npz')==checkpoint['normalization_sha256']
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='21')
    assert digest(OUT/case['input_file'])==case['input_sha256']==freeze['source_input_sha256']
    db=freeze['provenance']['domain'];assert digest(OUT/'run201'/db['json_file'])==db['json_sha256']
    domain=read(OUT/'run201'/db['json_file']);meta=read(data/'input_domain.json')
    with np.load(data/'input_domain.npz') as z:cells=z['cells'];family=z['family']
    with np.load(base/'normalization.npz') as z:mean=z['mean'];scale=z['scale']
    raw=np.load(data/'features.npy',mmap_mode='r');x=torch.from_numpy((raw-mean)/scale).cuda()
    index=torch.from_numpy(neighbours(meta['sequences'],len(raw))).cuda();model=ContactTCN(raw.shape[1]).cuda()
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    if ablate_process:
        with torch.no_grad():model.process_fusion.weight.zero_()
    with torch.inference_mode():out={k:v.cpu().numpy() for k,v in model(x,index).items()}
    np.savez(folder/'raw_outputs.npz',**out);leaves=decode(out,cells,meta['sequences'],family);pred=[]
    for i,leaf in enumerate(leaves):
        f=domain['families'][leaf['family']]
        pred.append(dict(**leaf,representation=representation(f,leaf['lo'],leaf['hi'],leaf['forward']),
                         prediction_id=i,relation=f['relation'],parent_assignment_status='predicted_by_connection_marks',process_attributes=predicted_process(out,cells,family,leaf)))
    write(folder/'predictions.json',pred)
    write(folder/'prediction_freeze.json',dict(status='PREDICTIONS_FROZEN',source='21',predicted=len(pred),predictions_sha256=digest(folder/'predictions.json'),
        raw_outputs_sha256=digest(folder/'raw_outputs.npz'),checkpoint_sha256=digest(base/'final.pt'),input_sha256=case['input_sha256'],
        target_reads=0,STEP_reads=0,process_fusion_ablation=ablate_process,checkpoint_file_unchanged=True,all_relations=len(domain['relations']),all_families=len(domain['families']),neural_tokens=len(raw),
        elapsed_seconds=time.perf_counter()-start,cached_upstream_and_feature_time_excluded=True,is_contract_timing=False,full_contract_pass=False,utc=utc()))
    print('ACTUAL_CONTACT_PREDICTIONS',len(pred),'SECONDS',round(time.perf_counter()-start,3),flush=True)

def predicted_process(out,cells,family,leaf):
    from weld_pipeline.sjg_process_rule_heads_1063 import FIELDS
    mask=(family==leaf['family']) & (cells[:,1]>leaf['lo']) & (cells[:,0]<leaf['hi'])
    if not mask.any():return dict(state='no_local_support')
    flags=out['process_flags'][mask].mean(0);states=out['process_state'][mask].mean(0).argmax(-1);numeric=out['process_numeric'][mask].mean(0)*50
    if not leaf['forward']:
        order=[1,0,3,2,5,4,7,6];states=states[order];numeric=numeric[order]
    result=dict(has_eave_probability=float(1/(1+np.exp(-flags[0]))),side_plate_split_probability=float(1/(1+np.exp(-flags[1]))),source='model_prediction_only',formal_E16_pass=False,passage_clearance_mm=None)
    for i,k in enumerate(FIELDS):
        state='exact' if states[i]==0 else ('no_wall_or_over_50' if k.endswith('wall') else 'over_50' if k.endswith('base_edge') else 'no_hole' if k.endswith('hole_height') else 'unconfirmed_state')
        result[k]=dict(state=state,value_mm=float(numeric[i]) if state=='exact' else None)
    return result


def score(run):
    from weld_pipeline.sjg_local_event_score_932 import geometric_match
    folder=OUT/run/'source_21';freeze=read(folder/'prediction_freeze.json')
    assert freeze['target_reads']==0 and digest(folder/'predictions.json')==freeze['predictions_sha256']
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='21')
    assert digest(OUT/case['target_file'])==case['target_sha256'];target=read(OUT/case['target_file']);pred=read(folder/'predictions.json')
    s=geometric_match(pred,target)
    # Directed result is conservative. D3 permission is not inferred from a
    # missing target field. Full parent correspondence stays explicitly open.
    s.update(source='21',role='TRAIN',prediction_sha256=freeze['predictions_sha256'],target_sha256=case['target_sha256'],
        elapsed_seconds=freeze['elapsed_seconds'],maximum_endpoint_error_mm=max((max(m['endpoint_errors_mm']) for m in s['matches']),default=None),
        maximum_bidirectional_error_mm=max((m['bidirectional_sampled_maximum_mm'] for m in s['matches']),default=None),
        D3_parent_correspondence='directed child diagnosis only; formal parent matching outstanding',full_contract_pass=False,utc=utc())
    write(folder/'geometry_score.json',s);score_parent_structure(folder,pred,target);print('ACTUAL_SCORE TP',s['TP'],'/',s['G'],'FP',s['FP'],'FN',s['FN'],'MAX_MM',s['maximum_bidirectional_error_mm'],flush=True)

def score_parent_structure(folder,pred,target):
    from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
    from weld_pipeline.sjg_native_prediction_adapter_1064 import decoded_parents
    decoded=decoded_parents(pred)
    result=score_native_parents(decoded,target,[False]*len(target['parents']))
    result['direction_scope']='development whole-parent reversal allowance; formal specified direction still required'
    result['parent_scope']='predicted within-sequence connections; cross-sequence assembly not certified'
    write(folder/'parent_d3_score.json',result)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--training-run',default='run235');ap.add_argument('--score',action='store_true');ap.add_argument('--ablate-process',action='store_true');a=ap.parse_args()
    try:score(a.run) if a.score else infer(a.run,a.training_run,a.ablate_process)
    except BaseException:
        folder=OUT/a.run;folder.mkdir(exist_ok=True);write(folder/'failure.json',dict(error=traceback.format_exc(),utc=utc()));raise
if __name__=='__main__':main()
