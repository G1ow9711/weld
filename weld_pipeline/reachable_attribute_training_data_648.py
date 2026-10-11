"""Complete TRAIN labels, current-XYZ hypotheses and own attribute transaction."""
import gc
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from weld_pipeline.aligned_transaction_training_data_583 import prepare_aligned_events_source
from weld_pipeline.legal_allocation_training_data_520 import ROOT,HERE,prepare_inputs
from weld_pipeline.legacy_incremental_supervision_388 import read,digest
from weld_pipeline.endpoint_local_finite_support_645 import VERSION
from weld_pipeline.endpoint_local_support_memory_646 import build_finite_surface_index
from weld_pipeline.reachable_set_attribute_head_637 import attribute_loss,attribute_totals,legal_attribute_output_check,PHYSICAL_FIELDS
from weld_pipeline.physical_trial_transactions_585 import finite_independent_transaction
from weld_pipeline.legal_allocation_finite_update_521 import rng_state,restore_rng,trial_submit
from weld_pipeline.legacy_scale_count_training_loop_488 import analytic_adamw_proposal


def strict_attribute_transaction(optimizer,parameters,check,initial_rng,commit):
    before=[p.detach().clone() for p in parameters]
    state=deepcopy(optimizer.state_dict());native_calls=0
    def restore():
        with torch.no_grad():
            for p,old in zip(parameters,before):p.copy_(old)
        optimizer.load_state_dict(state);restore_rng(initial_rng)
    try:
        proposal=analytic_adamw_proposal(optimizer,parameters)
        assert bool(torch.isfinite(proposal).all())
        trials=[];chosen=None
        for fraction in (1.,.5,.25,.125):
            trial_submit(parameters,before,proposal,fraction)
            nonzero=any(not torch.equal(p,old) for p,old in zip(parameters,before))
            result=check() if nonzero else dict(pass_all=False,actual_parameter_change_zero=True)
            trials.append(dict(fraction=fraction,result=result))
            if result["pass_all"]:chosen=fraction;break
        restore()
        evidence=dict(trials=trials,chosen_fraction=chosen)
        if chosen is None or not commit:
            return dict(**evidence,accepted=chosen is not None,committed_optimizer_steps=0,native_proposal_step_calls=0,
                normal_four_trial_rejection=chosen is None,own_parameters_optimizer_RNG_restored=True)
        optimizer.step();native_calls=1
        native=torch.cat([(p.detach().double()-old.double()).reshape(-1) for p,old in zip(parameters,before)])
        trial_submit(parameters,before,native,chosen)
        final=check();nonzero=any(not torch.equal(p,old) for p,old in zip(parameters,before))
        if not final["pass_all"] or not nonzero:
            restore()
            return dict(**evidence,accepted=False,committed_optimizer_steps=0,native_proposal_step_calls=1,
                final_native_guard=final,stop_branch=True,own_parameters_optimizer_RNG_restored=True)
        return dict(**evidence,accepted=True,committed_optimizer_steps=1,native_proposal_step_calls=1,
            final_actual_guard=final,actual_minus_analytic_maximum=float((native-proposal).abs().max()))
    except BaseException as error:
        restore();error.native_proposal_step_calls=native_calls
        error.transaction_parameters_optimizer_RNG_restored=True
        raise


def attach_current_physical_hypotheses(index,source,prior,metadata,cache_path):
    from training.dataio import read_xyz_pcd
    cache_path=Path(cache_path)
    raw=metadata["raw"][index]
    original=ROOT/raw["original_pcd_path"]
    assert digest(original)==raw["original_pcd_sha256"]
    module_sha=digest(ROOT/"weld_pipeline/endpoint_local_finite_support_645.py")
    special=HERE/"actual_endpoint_support_647_attempt1/TRAIN_source0_hypotheses.json"
    existing=cache_path if cache_path.is_file() else special if index==0 else None
    if existing is not None:
        stored=read(existing)
        assert stored["version"]==VERSION and stored["physical_module_sha256"]==module_sha
        assert stored["original_PCD_sha256"]==raw["original_pcd_sha256"]
        assert stored["TRAIN_target_location_queries_only_not_runtime_lookup"]
        if existing==special:
            terminal=read(special.parent/"controller_tool_terminal.json")
            assert terminal["actual_terminal"]["exit_code"]==0
    else:
        xyz=read_xyz_pcd(original)*1000.
        assert len(xyz)==raw["original_points"]
        sequences,_,_,mapping=prepare_inputs(index,prior,metadata)
        canonical_path=ROOT/mapping["E16_canonical_payload_path"]
        assert digest(canonical_path)==mapping["E16_canonical_payload_sha256"]
        children=read(canonical_path)["children"]
        indexer=build_finite_surface_index(xyz)
        for sequence in sequences:
            center=(sequence["xyz"][0]+sequence["xyz"][-1])/2.
            axes=np.asarray(sequence["features"])[len(sequence["features"])//2,60:66].reshape(2,3)
            for normal in axes:
                if np.linalg.norm(normal)>1e-8:indexer.register_lazy_plane(normal,center)
        by_id={cid:(row,ci) for row in mapping["mapped_parents"] for ci,cid in enumerate(row["original_child_ids"])}
        path,local=[],[]
        for child in children:
            owner=by_id.get(child["original_child_id"])
            if owner is None:
                result=dict(fields=[[] for _ in range(16)])
            else:
                row,ci=owner
                sequence=sequences[row["sequence_index"]]
                a,z=row["child_parameter_endpoints_mm"][ci]
                result=indexer.ordered_hypotheses(sequence,a,z)
            path.append(dict(fields=result["fields"],unresolved_fields_explicit=True))
            for sample in child["ordered_auxiliary_material_samples"]:
                if owner is None or not sample["pointwise_payload_valid"]:
                    item=dict(fields=[[] for _ in range(16)])
                else:
                    position=a+float(sample["fraction"])*(z-a)
                    item=indexer.ordered_hypotheses(sequence,position,position)
                local.append(dict(fields=item["fields"],unresolved_fields_explicit=True))
        stored=dict(version=VERSION,original_PCD_sha256=raw["original_pcd_sha256"],
            path=path,local=local,TRAIN_target_location_queries_only_not_runtime_lookup=True,
            full_original_rows_retained=indexer.original_rows,physical_module_sha256=module_sha,
            canonical_payload_sha256=mapping["E16_canonical_payload_sha256"],
            first_proposal_ownership_preserved=True,maximum_actual_resident_planes=indexer.maximum_actual_resident_planes)
        del xyz,sequences,indexer
        gc.collect()
    assert len(stored["path"])==len(source["path_features"])==source["original_G"]
    assert len(stored["local"])==source["original_local_points"]==len(source["local_features"])
    if not cache_path.is_file():
        cache_path.parent.mkdir(parents=True,exist_ok=True)
        def plain(value):
            if isinstance(value,np.ndarray):return value.tolist()
            if isinstance(value,np.generic):return value.item()
            raise TypeError(type(value).__name__)
        with cache_path.open("x",encoding="utf-8") as stream:
            json.dump(stored,stream,ensure_ascii=False,default=plain,allow_nan=False)
    source["path_plate_hypotheses"],source["local_plate_hypotheses"]=stored["path"],stored["local"]
    source["physical_hypothesis_cache_sha256"]=digest(cache_path)
    source["physical_hypothesis_facts"]={prefix:dict(
        total=sum(len(field) for r in stored[prefix] for field in r["fields"]),
        valid=sum(bool(h["field_valid"]) for r in stored[prefix] for field in r["fields"] for h in field))
        for prefix in ("path","local")}
    return source


def prepare_complete_source(index,prior,metadata,cache_path):
    source=prepare_aligned_events_source(index,prior,metadata)
    return attach_current_physical_hypotheses(index,source,prior,metadata,cache_path)


def attribute_objective(model,source,device="cuda",*,backward=False):
    value,effective,all_finite=0.,0,True
    facts={}
    for prefix,name in (("path","path_attributes"),("local","attributes")):
        chosen=np.flatnonzero(source[prefix+"_available"])
        target=source[prefix+"_targets"]
        totals=attribute_totals({key:torch.as_tensor(value[chosen]) for key,value in target.items()})
        coverage={str(fi):0 for fi in PHYSICAL_FIELDS}
        associated={str(fi):0 for fi in PHYSICAL_FIELDS}
        state_correct=np.zeros((16,5),np.int64)
        boolean_correct=np.zeros((16,2),np.int64)
        for offset in range(0,len(chosen),128):
            rows=chosen[offset:offset+128]
            output=getattr(model,name)(torch.as_tensor(source[prefix+"_features"][rows],device=device),
                [source[prefix+"_plate_hypotheses"][int(i)] for i in rows])
            loss,report=attribute_loss(output,{key:torch.as_tensor(v[rows],device=device) for key,v in target.items()},totals)
            guard=legal_attribute_output_check(output)
            predicted_states=output["state_logits"].detach().argmax(-1).cpu().numpy()
            predicted_boolean=(output["scalar_logits"].detach()>=0.).cpu().numpy()
            for fi in range(16):
                for label in range(5):
                    known=np.asarray(target["state_known"][rows,fi],bool) & (target["state_class"][rows,fi]==label)
                    state_correct[fi,label]+=int((known & (predicted_states[:,fi]==label)).sum())
                if fi in (12,14):
                    for label in range(2):
                        known=np.asarray(target["scalar_known"][rows,fi],bool) & (target["scalar_values"][rows,fi]==label)
                        boolean_correct[fi,label]+=int((known & (predicted_boolean[:,fi]==bool(label))).sum())
            all_finite &= guard["pass_all"]
            if backward and report["effective_loss_terms"]:
                (loss*.25).backward()
            value+=float(loss.detach())*.25
            effective+=report["effective_loss_terms"]
            for fi in PHYSICAL_FIELDS:
                coverage[str(fi)]+=report["physical_candidate_coverage_miss"][fi]
                associated[str(fi)]+=report["compatible_set_supervised_rows"][fi]
            del output,loss
        facts[prefix]=dict(available_reference_rows=len(chosen),original_rows=len(source[prefix+"_features"]),
            state_known_denominators=list(map(int,totals["state_known"])),
            state_class_counts=totals["state_class_counts"],boolean_class_counts=totals["boolean_class_counts"],
            state_class_correct=state_correct.tolist(),
            state_class_recall=[[float(state_correct[j,c]/n) if n else None for c,n in enumerate(counts)] for j,counts in enumerate(totals["state_class_counts"])],
            boolean_false_true_correct=boolean_correct.tolist(),
            boolean_false_true_recall={str(j):[float(boolean_correct[j,c]/n) if n else None for c,n in enumerate(totals["boolean_class_counts"][j])] for j in (12,14)},
            scalar_known_denominators=list(map(int,totals["scalar_known"])),
            normal_known_denominators=list(map(int,totals["normal_known"])),
            physical_candidate_coverage_miss=coverage,compatible_set_supervised_rows=associated)
    assert math.isfinite(value)
    return value,dict(pass_all=all_finite,effective_loss_terms=effective,supervision=facts,
        original_G=source["original_G"],original_E16_records=source["original_E16_records"],
        original_local_points=source["original_local_points"],no_formal_truth_created=True,
        physical_candidate_facts=source["physical_hypothesis_facts"])


def attribute_transaction(model,optimizer,source,device="cuda",*,commit=True):
    initial_rng=rng_state()
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    parameters=list(model.attributes.parameters())+list(model.path_attributes.parameters())
    before,details=attribute_objective(model,source,device,backward=True)
    if not details["effective_loss_terms"]:
        return dict(accepted=False,committed_optimizer_steps=0,native_proposal_step_calls=0,
            no_effective_supervision=True,branch="attributes",details=details)
    norms={name:float(torch.linalg.vector_norm(torch.cat([p.grad.reshape(-1) for p in getattr(model,name).parameters() if p.grad is not None])))
           if any(p.grad is not None for p in getattr(model,name).parameters()) else 0.
           for name in ("attributes","path_attributes")}
    preclip=float(torch.nn.utils.clip_grad_norm_(parameters,5.))
    assert math.isfinite(preclip)

    @torch.no_grad()
    def check():
        after,report=attribute_objective(model,source,device)
        report.update(pass_all=report["pass_all"] and after<before,
            attribute_loss_before=before,attribute_loss_after=after)
        return report

    result=strict_attribute_transaction(optimizer,parameters,check,initial_rng=initial_rng,commit=commit)
    result.update(branch="attributes",attribute_loss_before=before,preclip_norm=preclip,
                  per_head_preclip_norms=norms,details=details,
                  event_geometry_parameters_are_separate=True,physical_hypothesis_cache_sha256=source["physical_hypothesis_cache_sha256"])
    model.verify_frozen()
    return result

