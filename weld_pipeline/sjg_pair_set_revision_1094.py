"""Warm-start extension and objective controls for the RoboWeldAR pair trial."""
import copy
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from weld_pipeline.sjg_parent_set_network_1075 import ParentSetModel
from weld_pipeline.sjg_parent_set_loss_1084 import assignment


def widen_bins(tensor):
    if tensor.ndim==2 and tensor.shape[1]==16*242:
        result=tensor.new_zeros((tensor.shape[0],16,247))
        result[:,:,:242]=tensor.reshape(tensor.shape[0],16,242)
        return result.flatten(1)
    return tensor


def warm_start(checkpoint,device):
    model=ParentSetModel(bin_channels=247).to(device)
    state=copy.deepcopy(checkpoint['state_dict'])
    state['bin_encoder.weight']=widen_bins(state['bin_encoder.weight'])
    model.load_state_dict(state)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    opt=copy.deepcopy(checkpoint['optimizer_state'])
    for entry in opt['state'].values():
        for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if name in entry:entry[name]=widen_bins(entry[name])
    optimizer.load_state_dict(opt)
    return model,optimizer


def loss_controls():
    records=[]
    for p in (0,1,2,124):
        q=256 if p==124 else 4
        torch.manual_seed(1031+p)
        positive=torch.randn(q,p,dtype=torch.float64)
        negative=F.softplus(torch.randn(q,dtype=torch.float64))
        rows,cols=assignment(positive,negative,True)
        mask=torch.ones(q,dtype=torch.bool);mask[rows]=False
        actual=(positive[rows,cols].sum()+negative[mask].sum())/max(p,1)
        rebuilt=negative.sum()/max(p,1)+((positive-negative[:,None])/max(p,1))[rows,cols].sum()
        assert torch.allclose(actual,rebuilt,atol=1e-12,rtol=1e-12)
        if p<=2:
            brute=min((sum(float(positive[i,j]) for j,i in enumerate(indices))+
                       sum(float(negative[i]) for i in range(q) if i not in indices))/max(p,1)
                      for indices in itertools.permutations(range(q),p))
            assert abs(float(actual)-brute)<1e-12
        if 0<p<q:
            z=torch.full((q,),np.log(p/(q-p)),dtype=torch.float64,requires_grad=True)
            loss=(F.softplus(-z[:p]).sum()+F.softplus(z[p:]).sum())/p
            grad=torch.autograd.grad(loss,z)[0]
            assert float(grad[:p].max())<0 and float(grad[p:].min())>0 and abs(float(grad.sum()))<1e-12
        records.append(dict(P=p,Q=q,total_loss=float(actual),cost_identity_error=float(abs(actual-rebuilt))))
    return dict(status='COMMON_DENOMINATOR_CONTROLS_PASSED',cases=records,
                P_zero_and_double_identity_supported=True,optimizer_updates=0)
