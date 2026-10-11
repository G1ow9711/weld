"""Original-weight binding, structural forward parity and gradient smoke check."""
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from weld_pipeline.server_legacy_runtime import install
ADAPTER=install()
from weld_pipeline.server_reference_distance_model import ReferenceDistanceModel,bounded_log_distance_loss
from weld_pipeline.sjg_material_model_859 import SurfaceMaterialModel
from weld_pipeline.sjg_ptv3_801 import voxelize
from weld_pipeline.sjg_local_runtime_904 import read,write,digest,check_resources
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'


def main():
    torch.set_num_threads(2);torch.manual_seed(1031);split=read(BASE/'reference_distance_auxiliary_protocol_v1/split.json');case=next(r for r in split['cases'] if r['role']=='train' and (BASE/'reference_distance_fields_v1'/r['key']/'receipt.json').exists());key=case['key'];folder=BASE/'reference_distance_fields_v1'/key;fr=read(folder/'receipt.json');path=BASE/'inputs'/key/'xyz_mm.npy';assert digest(path)==fr['input_sha256'];assert digest(folder/'lower_mm.npy')==fr['lower_sha256'];assert digest(folder/'upper_mm.npy')==fr['upper_sha256'];xyz=np.load(path,mmap_mode='r');anchor=np.random.default_rng(1031).integers(len(xyz));ids=cKDTree(xyz).query(xyz[anchor],k=min(4096,len(xyz)),workers=2)[1];block=voxelize(xyz[ids]);lower=torch.as_tensor(np.load(folder/'lower_mm.npy',mmap_mode='r')[ids],device='cuda');upper=torch.as_tensor(np.load(folder/'upper_mm.npy',mmap_mode='r')[ids],device='cuda')
    warm=ROOT/'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776/run162';r=read(warm/'receipt.json');assert digest(warm/'last.pt')==r['checkpoint_sha256'];state=torch.load(warm/'last.pt',map_location='cpu',weights_only=False)['model'];student=ReferenceDistanceModel();missing=student.load_state_dict(state,strict=False);assert set(missing.missing_keys)=={'distance_head.0.weight','distance_head.0.bias','distance_head.2.weight','distance_head.2.bias'} and not missing.unexpected_keys;teacher=SurfaceMaterialModel();teacher.load_state_dict(state,strict=True);student=student.cuda().eval();teacher=teacher.cuda().eval();check_resources()
    with torch.no_grad():
        old=teacher(block);new=student(block);errors={}
        for name,value in zip(['embedding','normal','boundary','material'],old):
            assert torch.allclose(value,new[name],atol=1e-5,rtol=1e-4);errors[name]=float((value-new[name]).abs().max())
    # Gradient only: no optimizer update, no altered original checkpoint.
    prediction=student(block);loss=bounded_log_distance_loss(prediction['log_distance'],lower,upper);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.distance_head.parameters());backbone_grad=sum(float(p.grad.square().sum()) for p in student.backbone.parameters() if p.grad is not None);assert backbone_grad>0
    v=torch.tensor([0.,1.,2.],requires_grad=True);lo=torch.expm1(torch.tensor([0.,.5,2.5]));hi=torch.expm1(torch.tensor([.2,1.5,3.]));control=bounded_log_distance_loss(v,lo,hi);assert torch.allclose(control,torch.tensor(.25/3));control.backward();assert torch.allclose(v.grad,torch.tensor([0.,0.,-1./3]))
    out=BASE/'reference_distance_model_controls_v1';out.mkdir(exist_ok=False);write(out/'receipt.json',dict(status='ORIGINAL_WEIGHT_FORWARD_AND_GRADIENT_CONTROLS_PASS',smoke_train_key=key,points=len(ids),original_checkpoint_sha256=r['checkpoint_sha256'],distance_field_receipt_sha256=digest(folder/'receipt.json'),split_sha256=digest(BASE/'reference_distance_auxiliary_protocol_v1/split.json'),module_sha256=digest(ROOT/'weld_pipeline/server_reference_distance_model.py'),structural_max_absolute_differences=errors,initial_smoke_loss=float(loss.detach()),backbone_squared_gradient_norm=backbone_grad,optimizer_updates=0,scope='One uniform input-selected TRAIN crop. Checks initialization/interface/gradient only, not trained performance or full weld acceptance.',DEV_reads=0));print('PASS',key,errors,float(loss.detach()),flush=True)

if __name__=='__main__':main()
