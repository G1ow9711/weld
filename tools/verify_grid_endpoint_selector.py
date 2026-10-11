from pathlib import Path
import numpy as np
import torch
from weld_pipeline.server_grid_endpoint_selector import GridEndpointSelector, StreamingEndpointTopK
from weld_pipeline.server_raw_event_endpoint_selector import RawEventEndpointSelector
from weld_pipeline.server_stable_endpoint_pairs import endpoint_pairs
from weld_pipeline.sjg_local_runtime_904 import write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 torch.set_num_threads(2);torch.manual_seed(1031)
 out=BASE/'grid_endpoint_selector_controls_v1';out.mkdir(exist_ok=False)
 old=RawEventEndpointSelector();new=GridEndpointSelector()
 checkpoint=BASE/'bounded_raw_selector_v1/model.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 old.load_state_dict(state['state_dict']);new.load_raw_state_dict(state['state_dict'])
 b,e=7,73;x=torch.randn(b,457);features=torch.randn(b,e,457);qr=torch.rand(b);er=torch.rand(b,e);length=torch.ones(b)*100;pv=torch.randn(b,e,13);flag=torch.zeros(b,e,dtype=torch.bool)
 with torch.no_grad():
  a=old(x,features,qr,er,length,pv);z=new(x,features,qr,er,length,pv,flag)
 assert torch.equal(a,z)
 flag[:,::2]=True;new(x,features,qr,er,length,pv,flag).square().mean().backward()
 assert new.grid_offset.grad.abs().sum()>0
 cases=0
 for e in [1,3,8,9,73]:
  logits=torch.randint(-2,3,(b,e,2)).float();pos=torch.arange(e,dtype=torch.float64)[None].expand(b,-1)
  legal=torch.rand(b,e,2)>.3;legal[0]=False
  dense=endpoint_pairs(logits,legal,pos)
  for chunk in [1,3,17,100]:
   stream=StreamingEndpointTopK()
   for i in range(0,e,chunk):stream.add(logits[:,i:i+chunk],legal[:,i:i+chunk],pos[:,i:i+chunk],torch.arange(i,min(i+chunk,e)))
   got=stream.pairs()
   assert torch.equal(got['valid'],dense['valid'])
   # Invalid -inf placeholders need not retain the same IDs; they never output.
   mask=got['valid']
   for k in ['left_ids','right_ids','base_logits']:assert torch.equal(got[k][mask],dense[k][mask]),k
   assert not got['valid'][0].any();cases+=1
 assert StreamingEndpointTopK().pairs() is None
 write(out/'receipt.json',dict(status='PASS',raw_checkpoint_sha256=digest(checkpoint),legacy_logits_bitwise_equal=True,grid_parameter_gradient=True,stream_dense_tied_score_cases=cases,empty_sides_no_output=True,module_sha256=digest('weld_pipeline/server_grid_endpoint_selector.py'),scope='Interface and selection controls only, no model fitting or quality claim'))
 print('PASS',cases,flush=True)
if __name__=='__main__':main()
