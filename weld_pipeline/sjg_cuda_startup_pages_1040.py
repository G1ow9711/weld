"""One measured CUDA startup page release, after temporary objects are gone."""
import gc
import torch
from weld_pipeline.sjg_local_runtime_904 import resources,reclaim_own_idle_pages,check_resources

def startup_pages():
    before=check_resources();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
    model=torch.nn.Sequential(torch.nn.Linear(419,96),torch.nn.GELU(),torch.nn.Linear(96,1)).cuda()
    x=torch.zeros((2692,419),device='cuda');loss=model(x).sum();loss.backward();torch.cuda.synchronize()
    warm=resources();del model,x,loss;gc.collect();torch.cuda.empty_cache()
    released=resources();page_reclaim=reclaim_own_idle_pages()
    torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
    after=check_resources()
    return dict(before=before,after_kernel_warmup=warm,after_object_release=released,page_reclaim=page_reclaim,
        after=after,optimizer_updates=0,one_call_only=True,rng_restored=True,private_commit_not_claimed_freed=True)
