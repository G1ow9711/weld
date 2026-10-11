"""Fresh-process two-model inference, then independent frozen-output scoring."""
import argparse,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources

def infer(run,training):
    import torch
    from weld_pipeline.sjg_contact_tcn_1052 import neighbours
    from weld_pipeline.sjg_structured_contact_1057 import decode
    from weld_pipeline.sjg_dual_models_1066 import ProcessModel,ExtractionModel
    from weld_pipeline.sjg_hierarchical_infer_1032 import representation
    from weld_pipeline.sjg_structured_test_1061 import predicted_process
    start=time.perf_counter();dest=OUT/run;dest.mkdir(exist_ok=False);folder=dest/'source_21';folder.mkdir();check_resources();torch.set_num_threads(2)
    base=OUT/training;receipt=read(base/'receipt.json');assert receipt['status']=='CONTACT_TRAIN_COMPLETE'
    assert digest(base/'final.pt')==receipt['checkpoint_sha256'];assert digest(base/'model_A.pt')==receipt['model_A_sha256']
    ck=torch.load(base/'final.pt',map_location='cpu');ak=torch.load(base/'model_A.pt',map_location='cpu');assert ck['model_A_sha256']==receipt['model_A_sha256']
    data=OUT/ck['data_run'];freeze=read(data/'input_freeze.json');assert freeze['target_reads']==0
    for name,sha in freeze['files'].items():assert digest(data/name)==sha
    pre=OUT/ak['preprocessing'];afreeze=read(pre/'input_freeze.json');assert afreeze['target_reads']==0
    ab=next(b for b in afreeze['cases'] if b['source']=='21');assert digest(pre/ab['file'])==ab['sha256']
    assert digest(base/'normalization.npz')==ck['normalization_sha256'];assert digest(base/'process_normalization.npz')==ak['normalization_sha256']
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='21');assert digest(OUT/case['input_file'])==case['input_sha256']==freeze['source_input_sha256']
    db=freeze['provenance']['domain'];assert digest(OUT/'run201'/db['json_file'])==db['json_sha256'];domain=read(OUT/'run201'/db['json_file'])
    meta=read(data/'input_domain.json')
    with np.load(data/'input_domain.npz') as z:cells=z['cells'];family=z['family']
    with np.load(base/'normalization.npz') as z:x=torch.from_numpy((np.load(data/'features.npy',mmap_mode='r')-z['mean'])/z['scale']).cuda()
    with np.load(base/'process_normalization.npz') as z:ax=torch.from_numpy((np.load(pre/ab['file'],mmap_mode='r')-z['mean'])/z['scale']).cuda()
    ma=ProcessModel().cuda();mb=ExtractionModel().cuda();ma.load_state_dict(ak['state_dict']);mb.load_state_dict(ck['state_dict']);ma.eval();mb.eval()
    index=torch.from_numpy(neighbours(meta['sequences'],len(x))).cuda();ids=torch.tensor(family,dtype=torch.long,device='cuda')
    with torch.inference_mode():
        process={k:v[ids] for k,v in ma(ax).items()};out=mb(x,index,process['process_code']);out.update(process);out={k:v.cpu().numpy() for k,v in out.items()}
    np.savez(folder/'raw_outputs.npz',**out);leaves=decode(out,cells,meta['sequences'],family);pred=[]
    for i,leaf in enumerate(leaves):
        f=domain['families'][leaf['family']]
        pred.append(dict(**leaf,representation=representation(f,leaf['lo'],leaf['hi'],leaf['forward']),prediction_id=i,relation=f['relation'],
            parent_assignment_status='predicted_by_connection_marks',process_attributes=predicted_process(out,cells,family,leaf)))
    write(folder/'predictions.json',pred)
    write(folder/'prediction_freeze.json',dict(status='PREDICTIONS_FROZEN',source='21',predicted=len(pred),predictions_sha256=digest(folder/'predictions.json'),
        raw_outputs_sha256=digest(folder/'raw_outputs.npz'),checkpoint_sha256=receipt['checkpoint_sha256'],model_A_sha256=receipt['model_A_sha256'],input_sha256=case['input_sha256'],
        target_reads=0,STEP_reads=0,all_relations=len(domain['relations']),all_families=len(domain['families']),neural_tokens=len(x),
        elapsed_seconds=time.perf_counter()-start,cached_upstream_and_feature_time_excluded=True,is_contract_timing=False,
        longest_output_mm=max((p['hi']-p['lo'] for p in pred),default=0),full_contract_pass=False,utc=utc()))
    print('DUAL_ACTUAL_PREDICTIONS',len(pred),'SECONDS',round(time.perf_counter()-start,3),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run246');ap.add_argument('--training-run',default='run245');ap.add_argument('--score',action='store_true');a=ap.parse_args()
    if a.score:
        from weld_pipeline.sjg_structured_test_1061 import score
        score(a.run)
    else:infer(a.run,a.training_run)
if __name__=='__main__':main()
