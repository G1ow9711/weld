"""TRAIN positive-query endpoint fit; never complete detection accuracy."""
from pathlib import Path
import argparse
import numpy as np
import torch
from weld_pipeline.server_raw_event_endpoint_selector import RawEventEndpointSelector
from weld_pipeline.server_fixed_fragment_queries import FixedQueries,GRID
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 batch02_keys={v["key"] for v in read(BASE/"additional_batch02_protocol.json")["cases"]}
 parser=argparse.ArgumentParser();parser.add_argument('--raw-events',action='store_true');args=parser.parse_args();folder=BASE/('bounded_raw_selector_v1' if args.raw_events else 'bounded_raw_control_v1');r=read(folder/'receipt.json');assert digest(folder/'model.pt')==r['model_sha256'];s=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model=RawEventEndpointSelector().cuda().eval();model.load_state_dict(s['state_dict']);torch.set_num_threads(2);tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32);reports=[]
 out=BASE/'endpoint_set_mass_argmax_audit_v1';out.mkdir(exist_ok=False)
 for key in ('new_07380fe2eac4','new_0d355d5d0cbe'):
  source=(BASE/'frozen_features/legacy_runtime')/('source_'+key);inp=FixedQueries(source,key);data=BASE/'exhaustive_carrier_supervision_v1/positive_query'/key;assert digest(data/'receipt.json')==r['bindings'][key]['query'];records=read(data/'records.json');cache=BASE/('bounded_raw_training_inputs_v1' if args.raw_events else ('batch02_boundary_anchor_training_v1' if key in batch02_keys else 'sampled_boundary_anchor_training_v1'))/key;assert digest(cache/'receipt.json')==r['bindings'][key]['events'];ev=np.load(cache/'events.npz');provfolder=cache if args.raw_events else BASE/('batch02_boundary_event_provenance_v1' if key in batch02_keys else 'boundary_event_provenance_v1')/key;pr=read(provfolder/'receipt.json');assert digest(provfolder/'receipt.json')==r['bindings'][key]['provenance'];pvfile=provfolder/('provenance.npy' if args.raw_events else 'features.npy');assert digest(pvfile)==pr['provenance_sha256' if args.raw_events else 'features_sha256'];pv=np.load(pvfile);provenance=pv if args.raw_events else np.column_stack((pv,np.zeros(len(pv),np.float32)));positions=ev['positions_mm'];featureids=ev['feature_indices'];offsets=ev['offsets'];sup=BASE/('bounded_raw_training_supervision_v1' if args.raw_events else ('batch02_boundary_anchor_supervision_v1' if key in batch02_keys else 'boundary_anchor_supervision_v1'))/key;assert digest(sup/'receipt.json')==r['bindings'][key]['supervision'];target=np.load(sup/'targets.npz');rows=target['query_rows'];lo=target['left_offsets'];ro=target['right_offsets'];left=target['left_event_ids'];right=target['right_event_ids'];queryids=np.array([records[i]['feature_rows'][0]//2*4+records[i]['query_index'] for i in rows]);pid=inp.pid[queryids//4];chosen=np.full((len(rows),2),-1,np.int64)
  heads=[]
  with torch.no_grad():
   for index,profile in enumerate(ev['profile_ids']):
    a,b=offsets[index:index+2];q=positions[a:b];length=float(ev['profile_length_mm'][index]);raw=np.asarray(inp.raw[np.column_stack((2*featureids[a:b],2*featureids[a:b]+1))],np.float64);xx=np.column_stack((raw.mean(1),raw.min(1),raw.max(1),q/length)).astype(np.float32);xx=tensor(np.clip((xx-s['mean'])/s['std'],-20,20))[None];er=tensor(q/length)[None];qp=torch.as_tensor(q,device='cuda',dtype=torch.float64)[None];group=np.flatnonzero(pid==profile)
    for begin in range(0,len(group),16):
     local=group[begin:begin+16];ids=queryids[local];x=inp.features(ids);spans=inp.spans[inp.canonical[ids//4],:2];qm=torch.as_tensor(spans[:,0]+GRID[ids%4]*(spans[:,1]-spans[:,0]),device='cuda',dtype=torch.float64);legal=torch.stack((qp<=qm[:,None],qp>=qm[:,None]),2);B=len(local)
     with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(tensor(np.clip((x-s['mean'])/s['std'],-20,20)),xx.expand(B,-1,-1),tensor(x[:,456]),er.expand(B,-1),tensor(np.full(B,length)),tensor(np.clip((provenance[a:b]-s['provenance_mean'])/s['provenance_std'],-20,20))[None].expand(B,-1,-1))
     chosen[local]=a+logits.float().masked_fill(~legal,-torch.inf).argmax(1).cpu().numpy()
     probability=logits.float().masked_fill(~legal,-torch.inf).softmax(1).cpu().numpy()
     for j,row in enumerate(local):
      for side,(off,ids_) in enumerate(((lo,left),(ro,right))):
       positive=ids_[off[row]:off[row+1]]
       if not len(positive):continue
       pp=probability[j,:,side];mass=float(pp[positive-a].sum());correct_=bool(np.any(positive==chosen[row,side]))
       heads.append(dict(query_row=int(rows[row]),side=side,positive_events=len(positive),legal_events=int(legal[j,:,side].sum()),positive_probability_mass=mass,chosen_probability=float(pp[chosen[row,side]-a]),argmax_correct=correct_))
  original=np.load(folder/(key+'_training_event_predictions.npz'));assert np.array_equal(original['query_rows'],rows) and np.array_equal(original['chosen_event_ids'],chosen)
  wrong=[h for h in heads if not h['argmax_correct']]
  report=dict(key=key,known_heads=len(heads),wrong_argmax_heads=len(wrong),wrong_argmax_positive_mass_ge_half=sum(h['positive_probability_mass']>=.5 for h in wrong),wrong_argmax_positive_mass_ge_point9=sum(h['positive_probability_mass']>=.9 for h in wrong),wrong_mass_quantiles=np.quantile([h['positive_probability_mass'] for h in wrong],[0,.25,.5,.75,1]).tolist() if wrong else [],original_chosen_bitwise_equal=True,model_sha256=r['model_sha256'],scope='Conditional TRAIN mass-vs-argmax diagnosis only; no fit, inference changes or full detection claims.')
  write(out/(key+'.json'),dict(report=report,heads=heads));reports.append(report);print(report,flush=True);inp.close()
 write(out/'receipt.json',dict(cases=reports,scope='Conditional TRAIN fit only',source_sha256=digest(__file__)))
if __name__=='__main__':main()
