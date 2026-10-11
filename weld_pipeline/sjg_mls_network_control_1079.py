"""Actual complete-21 input-path control, not training or weld validation."""
import hashlib
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_parent_input_1078 import ParentInput
from weld_pipeline.sjg_parent_set_network_1075 import ParentSetModel


def parameter_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    dest=OUT/'run261';dest.mkdir(exist_ok=False);start=time.perf_counter()
    try:
        torch.set_num_threads(2);torch.manual_seed(1031);reclaim_own_idle_pages();check_resources()
        frozen=read(OUT/'run260/input_freeze.json')
        assert digest(OUT/'run260/input_normalization.npz')==frozen['normalization_sha256']
        with np.load(OUT/'run260/input_normalization.npz') as z:normalization={k:z[k] for k in z.files}
        data=ParentInput('21',normalization)
        context,bins=data.arrays(slice(None))
        assert len(context)==5250 and bins.shape==(5250,16,242)
        c=torch.from_numpy(context);b=torch.from_numpy(bins).requires_grad_()
        model=ParentSetModel(bin_channels=242)
        before=parameter_hash(model)
        result=model(c,b,chunk=1024)
        loss=result['existence'].square().mean()+result['profile_query'].square().mean()
        loss.backward()
        gradient_norm=float(b.grad[...,188:].norm())
        assert gradient_norm>0 and np.isfinite(gradient_norm)
        with torch.no_grad():
            disabled=b.detach().clone()
            disabled[...,188:]=torch.from_numpy(-normalization['bin_mean'][:,188:]/normalization['bin_scale'][:,188:])
            ablated=model(c,disabled,chunk=1024)
            change=float((result['parent_latent']-ablated['parent_latent']).abs().max())
        assert change>0 and np.isfinite(change)
        assert parameter_hash(model)==before
        write(dest/'receipt.json',dict(status='MLS_FEATURES_CONNECTED_TO_PARENT_NETWORK',source='21',
            complete_profiles=5250,input_shape=list(b.shape),geometry_channels=54,
            geometry_input_gradient_norm=gradient_norm,geometry_disabled_parent_latent_max_change=change,
            model_parameters_unchanged=True,parameter_sha256=before,model_parameter_count=sum(p.numel() for p in model.parameters()),
            input_freeze_sha256=digest(OUT/'run260/input_freeze.json'),
            target_reads=0,STEP_reads=0,optimizer_updates=0,
            extraction_correct_rate=None,extraction_precision_mm=None,
            accuracy_status='NOT_TESTED_NO_TRAINED_SET_DECODER',
            does_not_prove_accuracy_improvement=True,seconds=time.perf_counter()-start,utc=utc()))
        print('MLS_NETWORK_CONNECTED',gradient_norm,change,'OPTIMIZER_UPDATES',0,flush=True)
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),optimizer_updates=0,utc=utc()));raise


if __name__=='__main__':main()
