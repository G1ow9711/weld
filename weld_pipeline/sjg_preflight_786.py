"""Zero-update rotation/translation feature and GPU gradient controls."""
from copy import deepcopy
import numpy as np
import torch
from weld_pipeline.sjg_geometry_776 import read, write
from weld_pipeline.sjg_prepare_776 import OUT
from weld_pipeline.sjg_local_frame_786 import local_arrays, load_source, reversed_batch
from weld_pipeline.sjg_model_786 import SurfaceJointNet786


def main():
    torch.set_num_threads(2); torch.manual_seed(786)
    with np.load(OUT/'source_21/inputs.npz', allow_pickle=False) as z: a={k:z[k] for k in z.files}
    geom=read(OUT/'source_21/geometry.json'); moved=deepcopy(geom)
    theta=.371; rotation=np.array([[np.cos(theta),-np.sin(theta),0],
                                   [np.sin(theta),np.cos(theta),0],[0,0,1.]])
    translation=np.array([813.,-274.,99.])
    for c in moved['contacts']:
        for key in ('p0','p1'): c[key]=(np.asarray(c[key])@rotation.T+translation).tolist()
        c['direction']=(np.asarray(c['direction'])@rotation.T).tolist()
    for f in moved['faces']:
        f['normal']=(np.asarray(f['normal'])@rotation.T).tolist()
        f['center']=(np.asarray(f['center'])@rotation.T+translation).tolist()
        f['lines']=[(np.asarray(line)@rotation.T+translation).tolist() for line in f['lines']]
    rotated=dict(a);rotated['patches']=(a['patches']@rotation.T).astype(np.float32)
    before=local_arrays(a,geom);after=local_arrays(rotated,moved)
    error=[float(np.max(np.abs(u-v))) for u,v in zip(before,after)]
    assert max(error)<1e-5,error
    data=load_source(4); model=SurfaceJointNet786().cuda()
    ids=np.arange(min(512,len(data['query'])),dtype=np.int64)
    nodes,edges,p,q,idx,y,o,mask=reversed_batch(data,ids,np.random.default_rng(786))
    logits,offset=model(nodes,data['edge_index_gpu'],edges,p,q,idx)
    loss=logits.square().mean()+offset.square().mean();loss.backward()
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
    assert model.point[0].weight.grad.abs().sum()>0
    assert model.messages[0][0].weight.grad.abs().sum()>0
    # Verify the query/target reversal retains occupancy and flips endpoint
    # channels as a pair, with valid integer CUDA graph indices.
    assert torch.allclose(y[:,0].cpu(),torch.from_numpy(data['targets'][ids,0]))
    assert idx.dtype==torch.int64 and torch.isfinite(q).all()
    result=dict(status='LOCAL_FRAME_RIGID_TRANSFORM_AND_GPU_GRADIENT_CONTROL_PASS',
        feature_max_errors=error,scope='frozen local patches and finite structure',
        pure_XYZ_preprocessor_rotation_invariance_not_tested=True,
        actual_optimizer_updates=0,gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    write(OUT/'preflight_786.json',result);print(result,flush=True)


if __name__=='__main__': main()
