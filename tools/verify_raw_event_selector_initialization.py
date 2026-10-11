"""Check warm-start preservation on an actual TRAIN query/event batch."""
from pathlib import Path
import numpy as np
import torch
from weld_pipeline.server_raw_event_endpoint_selector import RawEventEndpointSelector
from weld_pipeline.server_boundary_event_provenance_selector import BoundaryEventProvenanceSelector
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 torch.set_num_threads(2);folder=BASE/'boundary_provenance_batch02_expanded8_v1';r=read(folder/'receipt.json');assert digest(folder/'model.pt')==r['model_sha256'];s=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);old=BoundaryEventProvenanceSelector().cuda().eval();old.load_state_dict(s['state_dict']);new=RawEventEndpointSelector().cuda().eval();new.load_predecessor_state_dict(s['state_dict']);key=s['keys'][0];source=BASE/'frozen_features/legacy_runtime'/('source_'+key);inp=FixedQueries(source,key);cache=BASE/'sampled_boundary_anchor_training_v1'/key;ev=np.load(cache/'events.npz');pid=int(ev['profile_ids'][0]);a,b=ev['offsets'][:2];q=ev['positions_mm'][a:b];rows=ev['feature_indices'][a:b];length=ev['profile_length_mm'][0];raw=np.asarray(inp.raw[np.column_stack((2*rows,2*rows+1))],np.float64);x=np.column_stack((raw.mean(1),raw.min(1),raw.max(1),q/length)).astype(np.float32);queryids=np.flatnonzero(inp.pid==pid)[:4]*4;query=inp.features(queryids);B=len(queryids);assert B>0;pv=np.load(BASE/'boundary_event_provenance_v1'/key/'features.npy')[a:b];tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32);args=(tensor(np.clip((query-s['mean'])/s['std'],-20,20)),tensor(np.clip((x-s['mean'])/s['std'],-20,20))[None].expand(B,-1,-1),tensor(query[:,456]),tensor(q/length)[None].expand(B,-1),tensor(np.full(B,length)));p=tensor(np.clip((pv-s['provenance_mean'])/s['provenance_std'],-20,20))[None].expand(B,-1,-1);extended=torch.cat((p,torch.ones((*p.shape[:2],1),device='cuda')),2)
 with torch.no_grad():
  reference=old(*args,p);actual=new(*args,extended);torch.testing.assert_close(actual,reference,atol=1e-5,rtol=1e-5)
  with torch.autocast('cuda',dtype=torch.bfloat16):reference_bf=old(*args,p);actual_bf=new(*args,extended)
 out=BASE/'raw_event_selector_initialization_controls_v1';out.mkdir(exist_ok=False);write(out/'receipt.json',dict(status='PASS',actual_TRAIN_queries=B,actual_events=len(q),FP32_max_error=float((actual-reference).abs().max()),BF16_max_error=float((actual_bf-reference_bf).abs().max()),BF16_bitwise_asserted=False,new_flag_one_also_zero_effect_before_training=True,source_model_sha256=r['model_sha256'],module_sha256=digest('weld_pipeline/server_raw_event_endpoint_selector.py'),scope='Pre-optimization logits on same original candidates, not equivalence of enlarged candidate-set softmax or argmax'));inp.close()
if __name__=='__main__':main()
