"""Dimensionless geometry, global class denominators and detached pair auxiliary."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .world_set_loss_285 import MODE_NAMES,semantics,reference_samples
from .world_set_geometry_285 import parent_samples

SCALE_MM=100.
BETA=.03
CLASS_COUNTS={("horizontal","line"):1443,("vertical","line"):138,("horizontal","arc"):24}

def _cost(output,target):
    q=len(output["presence_logits"]); device=output["presence_logits"].device
    semantic=semantics(target,q,device)
    points,_=parent_samples(output,torch.arange(q),**semantic)
    error=torch.linalg.vector_norm(points-reference_samples(target,output["origin_mm"],device)[None],dim=-1)
    endpoint=error[:,:,[0,-1]].mean(dim=(1,2)); curve=error.mean(dim=(1,2))
    n=target["original_child_count"]
    kind=output["sequence_kind_logits"][:,:n].log_softmax(-1)
    kind_cost=-kind.gather(2,semantic["child_kinds"][:,:,None]).mean(dim=(1,2))
    mode=-output["mode_logits"].log_softmax(-1)[:,semantic["mode"]]
    stop=-output["sequence_kind_logits"][:,n].log_softmax(-1)[:,0]
    return (endpoint+curve)/(2*SCALE_MM)+kind_cost+mode+stop-F.logsigmoid(output["presence_logits"])

def geometry_terms(output,query,target):
    device=output["presence_logits"].device
    points,_=parent_samples(output,torch.tensor([query]),**semantics(target,1,device))
    reference=reference_samples(target,output["origin_mm"],device)
    points=points[0]
    endpoint=F.smooth_l1_loss(points[:,[0,-1]]/SCALE_MM,reference[:,[0,-1]]/SCALE_MM,reduction="none",beta=BETA).mean(dim=(1,2))
    error=torch.linalg.vector_norm(points-reference,dim=-1)
    smooth=F.smooth_l1_loss(error/SCALE_MM,torch.zeros_like(error),reduction="none",beta=BETA)
    curve=smooth.mean(dim=1)+smooth.max(dim=1).values
    return endpoint,curve,error

def local_world_loss(output,targets):
    parents=targets["parents"]; q=len(output["presence_logits"]); device=output["presence_logits"].device
    assert len(parents)==targets["original_parents"] and sum(p["original_child_count"] for p in parents)==targets["original_children"]
    with torch.no_grad():
        costs=torch.stack([_cost(output,parent) for parent in parents],dim=1)
        if not torch.isfinite(costs).all(): raise ValueError("Nonfinite complete matching")
        queries,truth_rows=linear_sum_assignment(costs.numpy())
    presence_target=torch.zeros(q); presence_target[torch.from_numpy(queries)]=1
    presence=F.binary_cross_entropy_with_logits(output["presence_logits"],presence_target)
    count=F.smooth_l1_loss((output["presence_logits"].sigmoid().sum()-len(parents))/max(1,len(parents)),torch.tensor(0.),beta=1.)
    zero=output["presence_logits"].sum()*0
    terms={name:zero.clone() for name in ("mode","kind","stop","tail","weld","gap","closed","turns","direction","mixed_direction","endpoint","curve","coarse_geometry")}
    errors=[]; represented=0; matched=[]
    for query,parent_row in zip(queries.tolist(),truth_rows.tolist()):
        parent=parents[parent_row]; n=parent["original_child_count"]
        sem=semantics(parent,1,device)
        weights=torch.tensor([25/(3*CLASS_COUNTS[(c["weld_type"],c["original_kind"])]) for c in parent["children"]],dtype=torch.float64)
        endpoint,curve,error=geometry_terms(output,query,parent)
        coarse_endpoint,coarse_curve,_=geometry_terms(output["coarse_prediction"],query,parent)
        terms["endpoint"]+=sum(endpoint*weights); terms["curve"]+=sum(curve*weights)
        terms["coarse_geometry"]+=sum((coarse_endpoint+coarse_curve)*weights)
        terms["mode"]+=F.cross_entropy(output["mode_logits"][query:query+1],torch.tensor([sem["mode"]]))
        terms["kind"]+=sum(F.cross_entropy(output["sequence_kind_logits"][query,:n],sem["child_kinds"][0],reduction="none")*weights)
        weld=torch.tensor([int(c["weld_type"]=="vertical") for c in parent["children"]])
        terms["weld"]+=sum(F.cross_entropy(output["weld_type_logits"][query,:n],weld,reduction="none")*weights)
        terms["stop"]+=F.cross_entropy(output["sequence_kind_logits"][query,n:n+1],torch.tensor([0]))
        if n+1<output["sequence_kind_logits"].shape[1]:
            tail=output["sequence_kind_logits"][query,n+1:]
            terms["tail"]+=F.cross_entropy(tail,torch.zeros(len(tail),dtype=torch.long))
        if n>1: terms["gap"]+=F.cross_entropy(output["gap_logits"][query,:n-1],sem["positive_gaps"][0].long())
        closure=torch.tensor([int(parent["is_closed"])])
        terms["closed"]+=F.cross_entropy(output["closed_logits"][query:query+1],closure)
        terms["turns"]+=F.cross_entropy(output["turns_logits"][query:query+1],closure)
        if sem["mode"]==1:
            terms["direction"]+=F.cross_entropy(output["direction_logits"][query:query+1],((sem["directions"]+1)//2).long())
        if sem["mode"]==2:
            for child,c in enumerate(parent["children"]):
                if c["original_kind"]=="arc":
                    signed=int(c["world_representation"]["arc_parameters"]["sweep_rad"]>0)
                    terms["mixed_direction"]+=weights[child]*F.cross_entropy(output["mixed_direction_logits"][query,child:child+1],torch.tensor([signed]))
        errors.extend(error.detach().max(dim=1).values.tolist()); represented+=n
        matched.append(dict(query=query,parent=parent_row))
    assert represented==targets["original_children"]
    for name in ("mode","stop","tail","gap","closed","turns","direction"):
        terms[name]=terms[name]/max(1,len(parents))
    discrete=sum(terms[name] for name in ("mode","kind","stop","weld","gap","closed","turns","direction","mixed_direction"))
    loss=10*presence+.1*count+2*discrete+.05*terms["tail"]+terms["endpoint"]+terms["curve"]+.25*terms["coarse_geometry"]
    if not torch.isfinite(loss): raise ValueError("Nonfinite full geometry loss")
    receipt=dict(original_parents=len(parents),original_children=represented,matched=matched,all_original_denominators_retained=True,
        geometry_scale_mm=SCALE_MM,huber_beta_dimensionless=BETA,class_normalization="FIXED25_OVER3_GLOBAL_CLASS_G_NO_PER_SOURCE_CANCELLATION",
        presence_BCE_mean_all_queries=float(presence.detach()),count_loss=float(count.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
        total_geometry_loss=float(loss.detach()),conditional_child_max_errors_mm=errors,conditional_result_not_formal_accuracy=True)
    return loss,receipt

def unordered_material_loss(model,source,output,targets,auxiliary,matching):
    by_id={child["original_child_id"]:child for child in auxiliary["children"]}
    embeddings=[]; positions=[]; records=[]
    for match in matching:
        parent=targets["parents"][match["parent"]]
        for child_index,truth in enumerate(parent["children"]):
            record=by_id[truth["original_child_id"]]
            if any(sample["training_point_mask"] for sample in record["samples"]):
                embeddings.append(output["child_embeddings"][match["query"],child_index])
                positions.append([s["original_reference_xyz_mm_label_only"] for s in record["samples"]]); records.append(record)
    zero=sum(p.sum()*0 for p in model.material_head.parameters())+sum(p.sum()*0 for p in model.material_encoder.parameters())
    if not records:
        return zero,dict(valid_children=0,valid_points=0,formal_E_supervision_cells=0,geometry_detached=True,actual_aux_loss=0.)
    prediction=model.predict_materials(source,torch.stack(embeddings),np.asarray(positions))
    thickness=torch.tensor([[[m["thickness_mm"] for m in s["unordered_material_pair"]] for s in record["samples"]] for record in records],dtype=torch.float32)
    normals=torch.tensor([[[m["CAD_outward_unit_normal_world"] for m in s["unordered_material_pair"]] for s in record["samples"]] for record in records],dtype=torch.float32)
    mask=torch.tensor([[s["training_point_mask"] for s in record["samples"]] for record in records],dtype=torch.float32)
    pair_losses=[]
    diagnostic=[]
    for permutation in ((0,1),(1,0)):
        target_t=thickness[:,:,permutation]; target_n=normals[:,:,permutation]
        t=F.smooth_l1_loss(prediction["thickness_mm"],target_t,reduction="none",beta=1.)
        cosine=(prediction["CAD_outward_unit_normals_world"]*target_n).sum(dim=-1)
        normal=(1-cosine)/(1-math.cos(math.radians(3)))
        point_loss=(t+normal).sum(dim=-1)*mask
        pair_losses.append(point_loss.sum(dim=-1)/(2*mask.sum(dim=-1).clamp(min=1)))
        diagnostic.append(((prediction["thickness_mm"]-target_t).abs(),torch.acos(cosine.clamp(-1,1))*180/math.pi))
    selected=torch.stack(pair_losses,dim=1).argmin(dim=1)
    pair=torch.stack(pair_losses,dim=1).gather(1,selected[:,None]).mean()
    unit=((prediction["raw_normal_lengths"]-1)**2*mask[:,:,None]).sum()/(2*mask.sum()).clamp(min=1)
    loss=pair+.1*unit
    idx=torch.arange(len(records))
    dt=torch.stack([d[0] for d in diagnostic],dim=1)[idx,selected]
    dn=torch.stack([d[1] for d in diagnostic],dim=1)[idx,selected]
    point_valid=mask.bool()[:,:,None].expand_as(dt)
    receipt=dict(valid_children=len(records),valid_points=int(mask.sum()),valid_materials=int(2*mask.sum()),
        actual_aux_loss=float(loss.detach()),whole_child_pair_permutation=selected.tolist(),pair_permutation_shared_between_thickness_and_signed_normals=True,
        conditional_thickness_abs_mm_median=float(dt.detach()[point_valid].median()),conditional_signed_normal_deg_median=float(dn.detach()[point_valid].median()),
        conditional_all_valid_points_pass_1mm_3deg=int((((dt<=1)&(dn<=3))|~point_valid).all(dim=-1).all(dim=-1).sum()),
        diagnostic_not_official_E_accuracy=True,normal_abs_dot_or_flip_used=False,geometry_detached=True,formal_E_supervision_cells=0)
    return loss,receipt
