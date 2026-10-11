"""One-parent latent complete-path responsibility and structured training loss."""
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_structured_contact_1057 import log_partition,target_score
from weld_pipeline.sjg_process_rule_heads_1063 import process_loss

def objective(out,y,target,layout,length,witnesses,process_target=None):
    z=log_partition(out,layout);scores=target_score(out,layout,target)
    costs=[];posloss=[]
    for seq in witnesses:
        s=layout.sequences[seq];a,b=s['start'],s['end'];mask=target['position_mask'][a:b]
        delta=(out['positions'][a:b].double()-target['positions'][a:b])*length[a:b,None]
        p=F.smooth_l1_loss(delta[mask],torch.zeros_like(delta[mask]),beta=1.) if mask.any() else delta.sum()*0
        posloss.append(p);costs.append(-scores[seq]+2*p)
    selected=int(torch.stack(costs).detach().argmin());seq=witnesses[selected];s=layout.sequences[seq];a,b=s['start'],s['end']
    structure=z.double().sum()-scores[seq].double();position=posloss[selected]
    dm=y['direction_mask'][a:b];known=y['weld_type'][a:b]>=0
    direction=F.binary_cross_entropy_with_logits(out['direction'][a:b][dm],y['direction'][a:b][dm]) if dm.any() else out['direction'].sum()*0
    types=F.cross_entropy(out['weld_type'][a:b][known],y['weld_type'][a:b][known]) if known.any() else out['weld_type'].sum()*0
    # Connection is already a marked transition in logZ and target_score.
    canonical_process=process_target
    if process_target is not None and dm.any() and float(y['direction'][a:b][dm].mean())<.5:
        order=torch.tensor([1,0,3,2,5,4,7,6],device=out['occupancy'].device)
        canonical_process={k:(v[order] if k in ('state','numeric','numeric_mask') else v) for k,v in process_target.items()}
    pl,pp=process_loss(out,canonical_process,s,target) if canonical_process is not None else (structure*0,{})
    loss=structure+2*position+direction+types+pl
    return loss,dict(structure=float(structure.detach()),position_mm=float(position.detach()),direction=float(direction.detach()),type=float(types.detach()),witness_sequence=seq,witness_family=s['family'],connection_in_structured_loss=True,process_loss=float(pl.detach()),**pp)
