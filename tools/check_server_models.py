"""Trusted migrated weights: loading and numerical smoke controls, not accuracy."""
import copy,hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from weld_pipeline.sjg_joint_pair_model_1248 import PairModel
r=ROOT/'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776'
manifest=json.loads((ROOT/'_source_manifest.json').read_text())['files']
for name in ['run432/pair_model.pt','run453/unified_model.npz','run456/unified_model.npz','run456/feature_scale.npz']:
 p=r/name;expected=next(v for v in manifest if v['relative_path']==str(p.relative_to(ROOT)))
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256']
# Explicitly trusted project checkpoint, verified against original transfer manifest.
payload=torch.load(r/'run432/pair_model.pt',map_location='cpu',weights_only=False)
m=PairModel(payload['classes']).eval();m.load_state_dict(payload['state_dict'],strict=True)
torch.manual_seed(458);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
x=torch.randn(32,180);c=torch.randn(160)
with torch.no_grad():
 a=m.encode(x,c);b=copy.deepcopy(m).cuda().encode(x.cuda(),c.cuda()).cpu();error=float((a-b).abs().max())
assert torch.isfinite(a).all() and error<1e-5
npz={}
for name in ['run453/unified_model.npz','run456/unified_model.npz','run456/feature_scale.npz']:
 with np.load(r/name,allow_pickle=False) as z:
  npz[name]={k:dict(shape=list(z[k].shape),finite=bool(np.isfinite(z[k]).all())) for k in z.files}
result=dict(status='MODEL_LOAD_AND_SYNTHETIC_ENCODER_PARITY_PASS',torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),encoder_max_error=error,npz=npz,real_input_parity=False,DP_identity_parity=False,accuracy_test=False,fit_calls=0)
(ROOT/'migration_checks/model_load_controls.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
