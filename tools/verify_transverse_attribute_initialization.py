from pathlib import Path
import numpy as np
import torch
from weld_pipeline.server_endpoint_attribute_head import EndpointAttributeHead
from weld_pipeline.server_transverse_attribute_head import TransverseAttributeHead
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 torch.set_num_threads(2);f=BASE/'predicted_endpoint_attribute_head_v1';r=read(f/'receipt.json');assert digest(f/'model.pt')==r['model_sha256'];s=torch.load(f/'model.pt',map_location='cpu',weights_only=False);key=s['keys'][0];d=BASE/'predicted_endpoint_attribute_features_v1'/key;c=BASE/'transverse_endpoint_training_context_v1'/key;cr=read(c/'receipt.json');assert cr['source_feature_receipt_sha256']==digest(d/'receipt.json') and digest(c/'features.npy')==cr['features_sha256'];x=np.load(d/'features.npy')[:64];extra=np.load(c/'features.npy')[:64];old=EndpointAttributeHead().eval();old.load_state_dict(s['state_dict']);new=TransverseAttributeHead().eval();new.load_predecessor_state_dict(s['state_dict']);normal=np.clip((x-s['mean'])/s['std'],-20,20)
 with torch.no_grad():
  a,b=old(torch.from_numpy(normal));c1,d1=new(torch.from_numpy(np.column_stack((normal,extra))));torch.testing.assert_close(a,c1,atol=1e-5,rtol=1e-5);torch.testing.assert_close(b,d1,atol=1e-5,rtol=1e-5)
 out=BASE/'transverse_attribute_initialization_controls_v1';out.mkdir(exist_ok=False);write(out/'receipt.json',dict(status='PASS',TRAIN_rows=len(x),state_max_error=float(abs(a-c1).max()),numeric_max_error_mm=float(abs(b-d1).max()),source_model_sha256=r['model_sha256'],module_sha256=digest('weld_pipeline/server_transverse_attribute_head.py'),scope='Warm-start forward equivalence on actual TRAIN inputs before optimization, not generalization'))
if __name__=='__main__':main()
