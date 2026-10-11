"""TRAIN positive-query endpoint fit; never complete detection accuracy."""
from pathlib import Path
import numpy as np
import torch
from weld_pipeline.server_boundary_event_selector import BoundaryEventSelector
from weld_pipeline.server_fixed_fragment_queries import FixedQueries,GRID
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 folder=BASE/'boundary_event_selector_continuation_v1';r=read(folder/'receipt.json');assert digest(folder/'model.pt')==r['model_sha256'];s=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model=BoundaryEventSelector().cuda().eval();model.load_state_dict(s['state_dict']);torch.set_num_threads(2);tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32);reports=[]
 for key in s['keys']:
  source=BASE/'frozen_features/legacy_runtime'/('source_'+key);inp=FixedQueries(source,key);data=BASE/'exhaustive_carrier_supervision_v1/positive_query'/key;assert digest(data/'receipt.json')==r['bindings'][key]['query'];records=read(data/'records.json');cache=BASE/'sampled_boundary_anchor_training_v1'/key;assert digest(cache/'receipt.json')==r['bindings'][key]['events'];ev=np.load(cache/'events.npz');positions=ev['positions_mm'];featureids=ev['feature_indices'];offsets=ev['offsets'];sup=BASE/'boundary_anchor_supervision_v1'/key;assert digest(sup/'receipt.json')==r['bindings'][key]['supervision'];target=np.load(sup/'targets.npz');rows=target['query_rows'];lo=target['left_offsets'];ro=target['right_offsets'];left=target['left_event_ids'];right=target['right_event_ids'];queryids=np.array([records[i]['feature_rows'][0]//2*4+records[i]['query_index'] for i in rows]);pid=inp.pid[queryids//4];chosen=np.full((len(rows),2),-1,np.int64)
  with torch.no_grad():
   for index,profile in enumerate(ev['profile_ids']):
    a,b=offsets[index:index+2];q=positions[a:b];length=float(ev['profile_length_mm'][index]);raw=np.asarray(inp.raw[np.column_stack((2*featureids[a:b],2*featureids[a:b]+1))],np.float64);xx=np.column_stack((raw.mean(1),raw.min(1),raw.max(1),q/length)).astype(np.float32);xx=tensor(np.clip((xx-s['mean'])/s['std'],-20,20))[None];er=tensor(q/length)[None];qp=torch.as_tensor(q,device='cuda',dtype=torch.float64)[None];group=np.flatnonzero(pid==profile)
    for begin in range(0,len(group),16):
     local=group[begin:begin+16];ids=queryids[local];x=inp.features(ids);spans=inp.spans[inp.canonical[ids//4],:2];qm=torch.as_tensor(spans[:,0]+GRID[ids%4]*(spans[:,1]-spans[:,0]),device='cuda',dtype=torch.float64);legal=torch.stack((qp<=qm[:,None],qp>=qm[:,None]),2);B=len(local)
     with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(tensor(np.clip((x-s['mean'])/s['std'],-20,20)),xx.expand(B,-1,-1),tensor(x[:,456]),er.expand(B,-1),tensor(np.full(B,length)))
     chosen[local]=a+logits.float().masked_fill(~legal,-torch.inf).argmax(1).cpu().numpy()
  assert (chosen>=0).all();known=np.column_stack((np.diff(lo)>0,np.diff(ro)>0));correct=np.zeros_like(known)
  for i in range(len(rows)):correct[i]=[np.any(left[lo[i]:lo[i+1]]==chosen[i,0]),np.any(right[ro[i]:ro[i+1]]==chosen[i,1])]
  both=correct.all(1)&(positions[chosen[:,0]]<positions[chosen[:,1]]);owners={tuple(o) for row in rows for o in records[row]['owners']};hits={tuple(o) for i,row in enumerate(rows) if both[i] for o in records[row]['owners']};np.savez(folder/(key+'_training_event_predictions.npz'),query_rows=rows,chosen_event_ids=chosen,known_head_mask=known,correct_head_mask=correct,nonempty_both_correct=both);report=dict(key=key,line_positive_queries=len(rows),known_endpoint_heads=int(known.sum()),correct_known_endpoint_heads=int(correct.sum()),represented_line_children=len(owners),line_children_with_both_endpoint_fit=len(hits),both_correct_queries=int(both.sum()),model_sha256=r['model_sha256'],scope='TRAIN positive query endpoint fit, ignores runtime detection, type and parent assembly; not full extraction or acceptance',fit_calls=0,heldout_reads=0,DEV_reads=0);reports.append(report);print(report,flush=True);inp.close()
 write(folder/'training_fit_summary.json',dict(cases=reports,scope='Conditional TRAIN fit only',source_sha256=digest(__file__)))
if __name__=='__main__':main()
