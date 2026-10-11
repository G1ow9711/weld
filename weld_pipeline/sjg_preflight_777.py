"""Permutation and finite-gradient controls; zero optimizer updates."""
import time
import torch
from weld_pipeline.sjg_train_777 import load_source,check_resources,OUT
from weld_pipeline.sjg_model_777 import SurfaceJointNet777 as SurfaceJointNet
from weld_pipeline.sjg_geometry_776 import write

def main():
    torch.set_num_threads(2);torch.manual_seed(776)
    a=load_source(4);n=len(a["nodes"])
    model=SurfaceJointNet(a["nodes"].shape[1]).cuda()
    q=torch.arange(512)
    p=torch.from_numpy(a["patches"][q]).cuda()
    x=torch.from_numpy(a["query"][q]).cuda()
    index=torch.from_numpy(a["query_node"][q]).cuda()
    before=check_resources();start=time.perf_counter()
    out,offset=model(a["nodes_gpu"],a["edge_index_gpu"],a["edge_type_gpu"],p,x,index)
    loss=out.square().mean()+offset.square().mean();loss.backward()
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
    assert model.point[0].weight.grad.abs().sum()>0
    assert model.messages[0][0].weight.grad.abs().sum()>0
    model.eval()
    with torch.no_grad():
        perm=torch.randperm(n,device="cuda");inverse=torch.argsort(perm)
        orig=model(a["nodes_gpu"],a["edge_index_gpu"],a["edge_type_gpu"],p,x,index)
        changed=model(a["nodes_gpu"][perm],inverse[a["edge_index_gpu"]],a["edge_type_gpu"],p,x,inverse[index])
        error=max(float((u-v).abs().max()) for u,v in zip(orig,changed))
        assert error<1e-5,error
    torch.cuda.synchronize()
    result=dict(status="GRAPH_AND_RAW_POINT_GRADIENTS_AND_NODE_RENUMBERING_PASSED",
        node_permutation_max_error=error,seconds=time.perf_counter()-start,
        gpu_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        actual_optimizer_updates=0,resources_before=before,resources_after=check_resources())
    write(OUT/"preflight_777.json",result);print(result,flush=True)
if __name__=="__main__":main()
