"""Check zero-column feature extension preserves the original first-family head."""
from pathlib import Path
import numpy as np
import torch
from weld_pipeline.server_arc_boundary_pointer import ArcBoundaryPointer
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from tools.train_joint_arc_boundary_pointer import first_family
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    torch.set_num_threads(2);cases,_,binding=first_family();source=BASE/'arc_parent_feedback_model_valid_v1';sr=read(source/'receipt.json');assert digest(source/'model.pt')==sr['model_sha256'];old=torch.load(source/'model.pt',map_location='cpu',weights_only=False);newpath=BASE/'curved_group_comparison_grouped_v1/model.pt';assert digest(newpath)==read(newpath.parent/'receipt.json')['model_sha256'];new=torch.load(newpath,map_location='cpu',weights_only=False)
    assert np.array_equal(old['mean'],new['mean'][:76]);assert np.array_equal(old['std'],new['std'][:76]);a=ArcBoundaryPointer().cuda().eval();a.load_state_dict(old['state_dict']);state=dict(old['state_dict']);state['encoder.0.weight']=torch.cat((state['encoder.0.weight'],torch.zeros(128,7)),1);b=ArcBoundaryPointer(input_dim=83).cuda().eval();b.load_state_dict(state);rows=[]
    with torch.no_grad():
        for c in cases:
            x=torch.as_tensor(np.clip((c['x']-new['mean'])/new['std'],-20,20),dtype=torch.float32,device='cuda');q=torch.as_tensor(c['q'],dtype=torch.float64,device='cuda');ea=a.encoder(x[:,:76]);eb=b.encoder(x);assert torch.allclose(ea,eb,atol=1e-5,rtol=1e-5);pa=a.decode(x[:,:76],q);pb=b.decode(x,q);assert pa==pb;rows.append(dict(face=c['face'],candidate=c['candidate'],prediction_equal=True,max_embedding_difference=float((ea-eb).abs().max())))
    out=BASE/'arc_retention_feature_adapter_audit_v1';out.mkdir(exist_ok=False);write(out/'report.json',dict(rows=rows,original_normalization_first76_exactly_equal=True,old_model_sha256=sr['model_sha256'],new_model_sha256=digest(newpath),first_family_bindings=binding,scope='Zero-extra-weight83 interface preserves old76 predictions on18 originally supervised carriers. Does not assert new trained model preserves weights or full-pool predictions.',fit_calls=0,DEV_reads=0));print('PASS',len(rows),'zero-column prediction parity')

if __name__=='__main__':main()
