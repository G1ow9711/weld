from pathlib import Path
import torch
from weld_pipeline.server_arc_boundary_pointer import ArcBoundaryPointer
from weld_pipeline.sjg_local_runtime_904 import write,digest

def main():
    torch.manual_seed(1031);torch.set_num_threads(2);model=ArcBoundaryPointer();x=torch.randn(7,76);q=torch.tensor([0.,1.,1.,2.,3.,4.,5.])
    embedded,state=model.prepare(x);scores,_=model.step(embedded,state,2,q)
    assert torch.isneginf(scores[:3]).all() and torch.isfinite(scores[3:]).all()
    # Supervision and STOP support different segment counts, including no parent.
    for path in ([],[0,6],[0,3,5,6]):
        loss=model.set_teacher_loss(x,q,[[i] for i in path],path);assert torch.isfinite(loss);model.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    try:model.set_teacher_loss(x,q,[[1],[2]],[1,2])
    except ValueError:pass
    else:raise AssertionError('Equal-position teacher boundaries were accepted')
    selected=model.decode(x,q);assert all(q[b]>q[a] for a,b in zip(selected,selected[1:]))
    with torch.no_grad():model.stop.weight.zero_();model.stop.bias.fill_(100.)
    assert model.decode(x,q)==[]
    out=Path('server_rebuild/expansion_v1/arc_boundary_pointer_controls_v1');out.mkdir(exist_ok=False)
    write(out/'receipt.json',dict(status='PASS',checks=['strict_coordinate_mask_including_duplicates','variable_boundary_count_loss','finite_gradients','equal_position_teacher_rejected','ordered_decode','immediate_STOP'],module_sha256=digest('weld_pipeline/server_arc_boundary_pointer.py'),scope='Pointer mechanics only; not trained quality, parent selection, or generalization.'))
    print('PASS')

if __name__=='__main__':main()
