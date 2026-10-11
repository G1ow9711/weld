"""Registered full25 local/aux round with transactional state and finite budget."""
from pathlib import Path
import io,json,random,shutil,time
import numpy as np
import torch
from .world_set_training_285 import read,file_sha,atomic_bytes,utc,cursor
from .source_checkpoint_232 import finite_state,tree_hash,gradient_receipt,parameter_binding,rng_payload
from .live_json_snapshot_282 import write_json_snapshot as dump
from .raw_point_groups_285 import RawPointSource
from .world_set_loss_295 import local_world_loss,unordered_material_loss
from .world_set_runtime_295 import decode_local_predictions
from .world_set_evaluation_295 import score_source,aggregate_sources
from .world_set_evaluation_285 import score_source as old_score_source,aggregate_sources as old_aggregate
from .reports.full25_source_geometry_263_20261003.run_fresh_source_geometry_263 import process_identity,memory
from .reports.local_observation_auxiliary_295_20261004.precheck_real_source_295 import initialize_model

VERSION="COUNTED_ALL25_LOCAL_ANALYTIC_DETACHED_UNORDERED_AUX_TRAINING_295_v1"

def rng_state(model):
    state=rng_payload(model,{},{})
    state["source_order_generator_policy"]="REGISTERED80_PRECOMPUTED_FULL25_SEED293_PERMUTATIONS"
    return state

def commit(directory,model,optimizer,current,protocol,source_receipt=None):
    state=finite_state(model,optimizer); rng=rng_state(model)
    protocol_hash=tree_hash(protocol)
    payload=dict(version=VERSION,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),cursor=current,protocol=protocol,
        protocol_content_sha256=protocol_hash,RNG=rng,RNG_content_sha256=tree_hash(rng),state_receipt=state,model_training_mode=model.training,
        source_receipt=source_receipt)
    stream=io.BytesIO(); torch.save(payload,stream); contents=stream.getvalue(); completed=current["complete_sources"]
    checkpoint=atomic_bytes(directory/("latest_A.pt" if completed%2==0 else "latest_B.pt"),contents)
    pinned=None
    if completed in (0,1) or completed%25==0:
        label="initial_zero" if completed==0 else "first_real_update" if completed==1 else "epoch_%03d"%(completed//25)
        pinned=atomic_bytes(directory/(label+".pt"),contents)
    receipt=dict(version=VERSION,utc=utc(),status="COMMITTED_COMPLETE_SOURCE_MODEL_ADAMW_RNG",cursor=current,checkpoint=checkpoint,
        pinned_checkpoint=pinned,protocol_content_sha256=protocol_hash,state_receipt=state,RNG_content_sha256=tree_hash(rng),source_receipt=source_receipt,
        full_contract_pass=False)
    history=directory/"history"/("source_%04d_commit.json"%completed)
    if history.exists(): raise ValueError("Cannot overwrite committed update history")
    dump(history,dict(receipt,historical_rotating_checkpoint_may_be_replaced=True)); dump(directory/"latest_commit.json",receipt)
    return receipt

def restore(directory,model,optimizer,protocol):
    descriptor=read(directory/"latest_commit.json"); path=descriptor["checkpoint"]["path"]
    assert file_sha(path)==descriptor["checkpoint"]["sha256"]
    saved=torch.load(path,map_location="cpu")
    assert saved["version"]==VERSION and saved["protocol"]==protocol and saved["cursor"]==descriptor["cursor"]
    assert parameter_binding(model,optimizer)==saved["state_receipt"]["parameter_binding"]
    model.load_state_dict(saved["state_dict"],strict=True); optimizer.load_state_dict(saved["optimizer"])
    assert finite_state(model,optimizer)==saved["state_receipt"]
    rng=saved["RNG"]; assert rng["CUDA"] is None and not rng["independent_numpy"] and not rng["independent_torch"]
    random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch"])
    assert tree_hash(rng_state(model))==saved["RNG_content_sha256"]
    model.train(saved["model_training_mode"])
    current=saved["cursor"]; assert current==cursor(current["complete_sources"],current["actual_committed_optimizer_updates"],protocol)
    intent=directory.parent/"optimizer_step_intent.json"
    if intent.exists() and read(intent)["complete_sources_after"]>current["complete_sources"]:
        raise ValueError("Uncommitted update intent: preserve and reconcile, never silently replay")
    return current,descriptor

def validate(root,protocol):
    assert protocol["version"]==VERSION and protocol["epochs"]==80 and protocol["maximum_complete_sources"]==2000
    assert protocol["formal_E_reviewed_cells"]==0 and protocol["seed"]==293
    for name,digest in {**protocol["source_code_sha256"],**protocol["artifacts_sha256"]}.items():
        if file_sha(root/name)!=digest: raise ValueError("Registered code/input/evidence changed: "+name)
    inputs=read(root/protocol["input_scope_path"]); targets=read(root/protocol["supervision_scope_path"])
    aux=read(root/protocol["auxiliary_scope_path"])
    assert len(inputs)==len(targets)==len(aux)==25 and sum(t["original_children"] for t in targets)==1605
    for index,(entry,target,auxentry) in enumerate(zip(inputs,targets,aux)):
        assert entry["source_index_metadata_only"]==target["source_index_metadata_only"]==auxentry["source_index"]==index
        assert file_sha(root/target["loss_target_path"])==target["loss_target_sha256"]
        assert file_sha(root/auxentry["path"])==auxentry["sha256"]
    assert len(protocol["global_epoch_source_orders"] )==80
    assert all(sorted(order)==list(range(25)) for order in protocol["global_epoch_source_orders"])
    return inputs,targets,aux

def branch_norms(model):
    values={}
    for name,param in model.named_parameters():
        if param.grad is not None:
            prefix=name.split(".")[0]; values[prefix]=values.get(prefix,0.)+float(param.grad.double().square().sum())
    return {name:value**.5 for name,value in values.items()}

def train(root,protocol_path,output,resume=False):
    root,output=Path(root),Path(output); protocol=read(protocol_path)
    if resume: assert output.is_dir()
    else: output.mkdir(exist_ok=False)
    identity=process_identity(); dump(output/"actual_compute_process_identity.json",identity)
    current=cursor(0,0,protocol); stage="REGISTERED_VALIDATION"; start=time.monotonic()
    def progress(**extra):
        dump(output/"progress.json",dict(version=VERSION,utc=utc(),stage=stage,cursor=current,actual_native_identity=identity,
            elapsed_seconds=time.monotonic()-start,resources=memory(),full_contract_pass=False,**extra))
    try:
        torch.set_num_threads(1); torch.set_num_interop_threads(1); assert torch.version.cuda is None
        progress(); inputs,target_entries,aux_entries=validate(root,protocol)
        model,initial=initialize_model(); assert initial==protocol["initialization"]
        model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4,foreach=False)
        geometry_parameters=[p for name,p in model.named_parameters() if not name.startswith(("material_encoder.","material_head."))]
        auxiliary_parameters=[p for name,p in model.named_parameters() if name.startswith(("material_encoder.","material_head."))]
        checkpoints=output/"checkpoints"
        if resume:
            current,descriptor=restore(checkpoints,model,optimizer,protocol)
            dump(output/("actual_resume_%s.json"%identity["creation_filetime_ticks"]),dict(restored_cursor=current,checkpoint=descriptor["checkpoint"],budget_reset=False))
        else:
            for directory in (checkpoints,checkpoints/"history",output/"source_receipts",output/"optimizer_executions",output/"epoch_diagnostics"):
                directory.mkdir()
            descriptor=commit(checkpoints,model,optimizer,current,protocol)
        while current["complete_sources"]<2000:
            epoch,slot=current["epoch"],current["slot"]; index=protocol["global_epoch_source_orders"][epoch][slot]
            entry,target_entry,aux_entry=inputs[index],target_entries[index],aux_entries[index]
            stage="REAL_FULL_SOURCE_INPUT_OPEN"; progress(source_index_metadata_only=index)
            source=RawPointSource(root,root/entry["group_directory"]); targets=read(root/target_entry["loss_target_path"]); auxiliary=read(root/aux_entry["path"])
            optimizer.zero_grad(set_to_none=True); stage="REAL_ALL_ORIGINAL_POINTS_TWO_LOCAL_REFINEMENTS"; progress(source_index_metadata_only=index)
            prediction=model(source)
            stage="REAL_COMPLETE_MATCH_GEOMETRY_AND_DETACHED_MATERIAL_LOSS"; progress(source_index_metadata_only=index,input_receipt=prediction["input_receipt"])
            geometry,geometry_receipt=local_world_loss(prediction,targets)
            material,material_receipt=unordered_material_loss(model,source,prediction,targets,auxiliary,geometry_receipt["matched"])
            loss=geometry+.1*material
            with torch.no_grad():
                decoded=decode_local_predictions(prediction)
                free_score=score_source(decoded,targets)
                probabilities=prediction["presence_logits"].sigmoid().numpy(); kinds=prediction["sequence_kind_logits"].argmax(-1).numpy()
                stops=[int(np.flatnonzero(row==0)[0]) if (row==0).any() else 17 for row in kinds]
                free=dict(present_parent_count=decoded["present_parent_count"],published_children=decoded["total_prediction_children"],categories=free_score["categories"],
                    presence_min=float(probabilities.min()),presence_median=float(np.median(probabilities)),presence_max=float(probabilities.max()),
                    presence_sum=float(probabilities.sum()),first_STOP_histogram={str(k):stops.count(k) for k in sorted(set(stops))},
                    free_matched_mm_errors=[m["bidirectional_sampled_maximum_mm"] for m in free_score["matches"]],
                    free_unmatched_original_children=targets["original_children"]-len(free_score["matches"]),sampled_before_current_update=True)
                del decoded,free_score
            stage="REAL_FULL_SOURCE_BACKWARD"; progress(source_index_metadata_only=index,loss=float(loss.detach()))
            loss.backward(); gradients=gradient_receipt(model); assert gradients["nonzero"]
            norms=branch_norms(model)
            # Independent clipping keeps unreviewed aux gradients from altering
            # the geometric step through a shared global clipping factor.
            geom_norm=torch.nn.utils.clip_grad_norm_(geometry_parameters,100.)
            aux_norm=torch.nn.utils.clip_grad_norm_(auxiliary_parameters,100.)
            assert torch.isfinite(geom_norm) and torch.isfinite(aux_norm)
            clipped=gradient_receipt(model); before=tree_hash(model.state_dict()); completed=current["complete_sources"]+1
            intent=dict(version=VERSION,utc=utc(),actual_identity=identity,complete_sources_before=current["complete_sources"],complete_sources_after=completed,
                optimizer_updates_before=current["actual_committed_optimizer_updates"],source_index_metadata_only=index,before_model_sha256=before,gradient_sha256=tree_hash(clipped))
            dump(output/"optimizer_step_intent.json",intent); stage="REAL_ADAMW_CALL"; progress(source_index_metadata_only=index)
            parameter_before={name:param.detach().clone() for name,param in model.named_parameters()}
            optimizer.step(); after=tree_hash(model.state_dict()); assert before!=after
            update_norms={}
            for name,param in model.named_parameters():
                prefix=name.split(".")[0]; difference=param.detach()-parameter_before[name]
                update_norms[prefix]=update_norms.get(prefix,0.)+float(difference.double().square().sum())
            update_norms={k:v**.5 for k,v in update_norms.items()}; del parameter_before
            executed=dict(intent,optimizer_step_returned=True,after_model_sha256=after,parameter_change_verified=True,actual_optimizer_invocations_this_source=1,returned_utc=utc())
            dump(output/"optimizer_executions"/("source_%04d.json"%completed),executed)
            next_cursor=cursor(completed,current["actual_committed_optimizer_updates"]+1,protocol)
            record=dict(version=VERSION,utc=utc(),cursor_before=current,cursor_after=next_cursor,source_index_metadata_only=index,input_receipt=prediction["input_receipt"],
                loss_receipt=geometry_receipt,auxiliary_loss_receipt=material_receipt,auxiliary_fixed_weight=.1,total_loss=float(loss.detach()),
                free_prediction_diagnostic=free,gradient_receipt_before_clip=gradients,gradient_receipt_used=clipped,branch_gradient_norms=norms,
                geometric_preclip_norm=float(geom_norm),auxiliary_preclip_norm=float(aux_norm),geometric_clip_factor=min(1.,100/max(float(geom_norm),1e-12)),
                auxiliary_clip_factor=min(1.,100/max(float(aux_norm),1e-12)),branch_parameter_update_norms=update_norms,
                optimizer_execution=executed,formal_E_supervision_cells=0,accuracy_measured=False,full_contract_pass=False)
            stage="COMPLETE_SOURCE_STATE_TRANSACTION"; progress(source_index_metadata_only=index,optimizer_returned_not_committed=True)
            descriptor=commit(checkpoints,model,optimizer,next_cursor,protocol,record)
            dump(output/"source_receipts"/("source_%04d.json"%completed),record); current=next_cursor
            if completed%25==0:
                rows=[read(output/"source_receipts"/("source_%04d.json"%number)) for number in range(completed-24,completed+1)]
                dump(output/"epoch_diagnostics"/("epoch_%03d.json"%(completed//25)),dict(complete_epoch=completed//25,actual_full25_source_order=[r["source_index_metadata_only"] for r in rows],
                    originals=1605,free_prediction_diagnostics=[r["free_prediction_diagnostic"] for r in rows],
                    conditional_errors_mm=[r["loss_receipt"]["conditional_child_max_errors_mm"] for r in rows],
                    branch_gradient_norms=[r["branch_gradient_norms"] for r in rows],clip_factors=[(r["geometric_clip_factor"],r["auxiliary_clip_factor"]) for r in rows],
                    branch_parameter_update_norms=[r["branch_parameter_update_norms"] for r in rows],auxiliary_masks_and_errors=[r["auxiliary_loss_receipt"] for r in rows],
                    epoch_diagnostic_not_independent_or_final_accuracy=True,full_contract_pass=False))
            stage="COUNTED_REAL_SOURCE_UPDATE_COMMITTED"; progress(source_index_metadata_only=index,last_total_loss=record["total_loss"],latest_checkpoint=descriptor["checkpoint"])
            print("REAL_COMMITTED_LOCAL_AUX_UPDATE",completed,"EPOCH",current["epoch"],"SOURCE",index,"AUX_POINTS",material_receipt["valid_points"],"LOSS",record["total_loss"],flush=True)
            del source,targets,auxiliary,prediction,geometry,material,loss
            if shutil.disk_usage(root).free<protocol["minimum_disk_free_bytes"]: raise RuntimeError("Registered disk floor breached at committed boundary")
        stage="FULL25_FREE_RUNTIME_FIT_EVALUATION"; evaluation=output/"actual_full25_fit_evaluation"; evaluation.mkdir(exist_ok=False)
        model.eval(); scores=[]; old_scores=[]
        with torch.no_grad():
            for index,(entry,target_entry) in enumerate(zip(inputs,target_entries)):
                progress(source_index_metadata_only=index,evaluation_sources_completed=len(scores))
                source=RawPointSource(root,root/entry["group_directory"]); prediction=model(source)
                decoded=decode_local_predictions(prediction,model,source); targets=read(root/target_entry["loss_target_path"])
                dump(evaluation/("source_%02d_predictions.json"%index),decoded)
                score=score_source(decoded,targets); old_score=old_score_source(decoded,targets)
                dump(evaluation/("source_%02d_score.json"%index),score); dump(evaluation/("source_%02d_old285_diagnostic.json"%index),old_score)
                scores.append(score); old_scores.append(old_score); del source,prediction,decoded,targets
        aggregate=aggregate_sources(scores); aggregate.update(actual_committed_optimizer_updates=2000,actual_complete_sources=2000,checkpoint=descriptor["checkpoint"],
            Pro_before_further_training_required=True,further_training_budget=0)
        dump(evaluation/"aggregate.json",aggregate); dump(evaluation/"old285_diagnostic_aggregate.json",old_aggregate(old_scores))
        stage="REGISTERED_FULL2000_LOCAL_AUX_ROUND_AND_FULL25_FIT_COMPLETE"; progress(actual_fit_result=aggregate,further_training_budget=0)
        dump(output/"receipt.json",dict(status=stage,utc=utc(),version=VERSION,cursor=current,aggregate=aggregate,actual_identity=identity,full_contract_pass=False))
        print("REAL_LOCAL_AUX_ROUND_COMPLETE",aggregate["categories"],flush=True)
    except BaseException as exc:
        dump(output/"failed_actual_training_attempt.json",dict(version=VERSION,utc=utc(),stage=stage,last_committed_cursor=current,
            actual_identity=identity,exception=type(exc).__name__,message=str(exc),optimizer_intent_requires_explicit_reconciliation=True,full_contract_pass=False))
        raise
