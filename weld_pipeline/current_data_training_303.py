"""Current65 development training: pure XYZ, new process rules, masked labels.

The older registered 295 round remains immutable. Native STEP measurements
are detached auxiliary supervision, never an inference input or final truth.
"""
from pathlib import Path
import argparse
import io
import json
import math
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .world_set_model_295 import LocalWorldModel
from .world_set_loss_295 import _cost, geometry_terms
from .world_set_loss_285 import semantics
from .world_set_geometry_285 import parent_samples
from .contract_attribute_heads_283 import (FIELDS, CONTINUOUS_FIELDS, BINARY_FIELDS, STATES, attribute_supervised_loss)
from .raw_point_groups_285 import RawPointSource
from .current_process_runtime_303 import decode_current_predictions
from .current_data_evaluation_303 import score_current_source
from .world_set_evaluation_285 import CATEGORIES
from .world_set_training_285 import read, file_sha, atomic_bytes, utc
from .source_checkpoint_232 import finite_state, tree_hash, gradient_receipt, rng_payload
from .live_json_snapshot_282 import write_json_snapshot as dump
from .reports.full25_source_geometry_263_20261003.run_fresh_source_geometry_263 import process_identity, memory

ROOT = Path(__file__).resolve().parents[1]
VERSION = "CURRENT65_RULES_FIXED_SIGNED_PLATES_MASKED16_DEVELOPMENT_303_v1"


class CurrentDataModel(LocalWorldModel):
    def __init__(self):
        super().__init__(512,16,128,16384)
        # Unsupported fields have explicit unknown output status; these heads
        # will receive a supervised gradient only when supported labels exist.
        self.extra_process_heads = nn.Linear(128,2+6)


def current_cursor(completed, protocol):
    assert 0 <= completed <= protocol["maximum_complete_sources"]
    return dict(complete_sources=completed, actual_committed_optimizer_updates=completed,
        epoch=completed//65, slot=completed%65)


def geometry_loss(output, targets, protocol):
    parents = targets["parents"]
    q = len(output["presence_logits"])
    assert len(parents)==targets["original_parents"] and sum(p["original_child_count"] for p in parents)==targets["original_children"]
    with torch.no_grad():
        cost = torch.stack([_cost(output,p) for p in parents],dim=1)
        assert torch.isfinite(cost).all()
        queries, rows = linear_sum_assignment(cost.numpy())
    positives = torch.zeros(q)
    positives[torch.from_numpy(queries)] = 1
    logits = output["presence_logits"]
    if targets["incomplete_source_no_unmatched_query_negative_labels"]:
        presence = F.binary_cross_entropy_with_logits(logits[positives.bool()],positives[positives.bool()])
        count = logits.sum()*0
    else:
        presence = F.binary_cross_entropy_with_logits(logits,positives)
        count = F.smooth_l1_loss((logits.sigmoid().sum()-len(parents))/max(1,len(parents)),torch.tensor(0.),beta=1.)
    zero = logits.sum()*0
    terms = {k:zero.clone() for k in ("mode","kind","stop","tail","weld","gap","closed","turns","direction","endpoint","curve","coarse")}
    matched = []
    errors = []
    for query,row in zip(queries.tolist(),rows.tolist()):
        parent = parents[row]
        n = parent["original_child_count"]
        sem = semantics(parent,1,logits.device)
        weights = torch.tensor([65/(3*protocol["class_counts"][c["weld_type"]+"/"+c["original_kind"]]) for c in parent["children"]],dtype=torch.float64)
        endpoint,curve,error = geometry_terms(output,query,parent)
        ce,cc,_ = geometry_terms(output["coarse_prediction"],query,parent)
        terms["endpoint"] += sum(endpoint*weights)
        terms["curve"] += sum(curve*weights)
        terms["coarse"] += sum((ce+cc)*weights)
        terms["mode"] += F.cross_entropy(output["mode_logits"][query:query+1],torch.tensor([sem["mode"]]))
        terms["kind"] += sum(F.cross_entropy(output["sequence_kind_logits"][query,:n],sem["child_kinds"][0],reduction="none")*weights)
        terms["weld"] += sum(F.cross_entropy(output["weld_type_logits"][query,:n],torch.tensor([int(c["weld_type"]=="vertical") for c in parent["children"]]),reduction="none")*weights)
        terms["stop"] += F.cross_entropy(output["sequence_kind_logits"][query,n:n+1],torch.tensor([0]))
        if n+1 < output["sequence_kind_logits"].shape[1]:
            terms["tail"] += F.cross_entropy(output["sequence_kind_logits"][query,n+1:],torch.zeros(output["sequence_kind_logits"].shape[1]-n-1,dtype=torch.long))
        if n>1:
            terms["gap"] += F.cross_entropy(output["gap_logits"][query,:n-1],sem["positive_gaps"][0].long())
        terms["closed"] += F.cross_entropy(output["closed_logits"][query:query+1],torch.tensor([int(parent["is_closed"])]))
        terms["turns"] += F.cross_entropy(output["turns_logits"][query:query+1],torch.tensor([int(parent["is_closed"])]))
        if sem["mode"]==1:
            terms["direction"] += F.cross_entropy(output["direction_logits"][query:query+1],((sem["directions"]+1)//2).long())
        for ci,c in enumerate(parent["children"]):
            if sem["mode"]==2 and c["original_kind"]=="arc":
                terms["direction"] += weights[ci]*F.cross_entropy(output["mixed_direction_logits"][query,ci:ci+1],torch.tensor([int(c["world_representation"]["arc_parameters"]["sweep_rad"]>0)]))
        errors.extend(error.detach().max(dim=1).values.tolist())
        matched.append(dict(query=query,parent=row))
    for key in ("mode","stop","tail","gap","closed","turns","direction"):
        terms[key] = terms[key]/max(1,len(parents))
    discrete = sum(terms[k] for k in ("mode","kind","stop","weld","gap","closed","turns","direction"))
    total = 10*presence+.1*count+2*discrete+.05*terms["tail"]+terms["endpoint"]+terms["curve"]+.25*terms["coarse"]
    assert torch.isfinite(total)
    return total,dict(matched=matched,loss=float(total.detach()),conditional_geometry_errors_mm=errors,
        known_geometry_children=targets["original_children"],original_declared_children=targets["original_declared_child_count"],
        missing_or_conflicting_children_retained=targets["missing_or_process_conflict_children"],
        negative_presence_masked_for_incomplete_source=targets["incomplete_source_no_unmatched_query_negative_labels"],
        prediction_accuracy=False)


def attribute_loss(model, output, targets, attribute, matching):
    by_id = {r["original_child_id"]:r for r in attribute["children"]}
    embeddings, fractions, locations, directions, label_rows = [],[],[],[],[]
    for match in matching:
        parent = targets["parents"][match["parent"]]
        points,_ = parent_samples(output,torch.tensor([match["query"]]),**semantics(parent,1,output["presence_logits"].device))
        for ci,child in enumerate(parent["children"]):
            xyz = points[0,ci].detach()
            delta = xyz[-1]-xyz[0]
            if float(delta.norm())<=1e-8:
                continue
            for endpoint in (0,1):
                embeddings.append(output["child_embeddings"][match["query"],ci].detach())
                fractions.append(float(endpoint))
                locations.append(xyz[-1 if endpoint else 0].float()/1000)
                directions.append((delta/delta.norm()).float())
                label_rows.append((by_id[child["original_child_id"]]["fields"],endpoint))
    if not embeddings:
        return sum(p.sum()*0 for p in model.contract_attributes.parameters()),dict(known_state_cells=0)
    count = len(embeddings)
    state = torch.full((count,len(FIELDS)),STATES.index("unknown"),dtype=torch.long)
    known = torch.zeros_like(state,dtype=torch.bool)
    values = torch.zeros(count,len(CONTINUOUS_FIELDS))
    binary = torch.zeros(count,len(BINARY_FIELDS),dtype=torch.long)
    for row,(fields,endpoint) in enumerate(label_rows):
        for col,field in enumerate(FIELDS):
            label = fields[field]
            appropriate = (endpoint==0 if field.startswith("start_") or field=="front_wall" else endpoint==1 if field.startswith("end_") or field=="back_wall" else endpoint==0)
            if not label["known"] or not appropriate:
                continue
            state[row,col] = STATES.index(label["state"])
            known[row,col] = True
            if label["state"]=="value":
                if field in CONTINUOUS_FIELDS:
                    values[row,CONTINUOUS_FIELDS.index(field)] = float(label["value"])/1000
                if field in BINARY_FIELDS:
                    binary[row,BINARY_FIELDS.index(field)] = int(label["value"])
    prediction = model.contract_attributes(torch.stack(embeddings),torch.tensor(fractions),torch.stack(locations),torch.stack(directions))
    total,receipt = attribute_supervised_loss(prediction,dict(states=state,state_known=known,continuous_m=values,
        normals=torch.zeros(count,2,3),binary=binary),material_side_labels_frozen=True)
    receipt = {k:v for k,v in receipt.items() if k!="components"}
    receipt.update(development_customer_fields=True,query_positions_from_current_predictions=True,geometry_detached=True,
        hole_edge_supervised_cells=0,segment_reason_supervised_cells=0,formal_all16_accuracy=False)
    return total,receipt


def ordered_material_loss(model, source, output, targets, attribute, matching):
    by_id = {r["original_child_id"]:r for r in attribute["children"]}
    embeddings, positions, thickness, normals = [],[],[],[]
    for match in matching:
        parent = targets["parents"][match["parent"]]
        for ci,child in enumerate(parent["children"]):
            for sample in by_id[child["original_child_id"]]["ordered_material_samples"]:
                embeddings.append(output["child_embeddings"][match["query"],ci])
                positions.append([sample["xyz_world_mm"]])
                thickness.append(sample["thickness_mm"])
                normals.append(sample["normals"])
    zero = sum(p.sum()*0 for p in model.material_head.parameters())+sum(p.sum()*0 for p in model.material_encoder.parameters())
    if not embeddings:
        return zero,dict(ordered_material_candidate_points=0,side_exchange_used=False,formal_E_accuracy=False)
    result = model.predict_materials(source,torch.stack(embeddings),np.asarray(positions,float))
    true_t = torch.tensor(thickness)
    true_n = torch.tensor(normals)
    predicted_t = result["thickness_mm"][:,0]
    predicted_n = result["CAD_outward_unit_normals_world"][:,0]
    cosine = (predicted_n*true_n).sum(dim=-1).clamp(-1,1)
    dt = (predicted_t-true_t).abs()
    dn = torch.acos(cosine)*180/math.pi
    total = F.smooth_l1_loss(predicted_t,true_t,beta=1)+((1-cosine)/(1-math.cos(math.radians(3)))).mean()
    total += .1*((result["raw_normal_lengths"][:,0]-1)**2).mean()
    return total,dict(ordered_material_candidate_points=len(embeddings),fixed_plate1_plate2_order=True,
        side_exchange_used=False,absolute_normal_dot_used=False,geometry_detached=True,
        native_labels_used_only_in_training_auxiliary_branch=True,customer_reviewed=False,
        conditional_thickness_mm_median=float(dt.detach().median()),conditional_signed_normal_deg_median=float(dn.detach().median()),
        conditional_candidate_points_both_sides_pass=int(((dt<=1)&(dn<=3)).all(dim=-1).sum()),formal_E_accuracy=False)


def commit(output, model, optimizer, completed, protocol, record=None):
    current = current_cursor(completed,protocol)
    rng = rng_payload(model,{},{})
    state = finite_state(model,optimizer)
    payload = dict(version=VERSION,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),cursor=current,
        protocol=protocol,state_receipt=state,RNG=rng,RNG_sha256=tree_hash(rng),source_receipt=record)
    stream = io.BytesIO()
    torch.save(payload,stream)
    checkpoint = atomic_bytes(output/"checkpoints"/("latest_A.pt" if completed%2==0 else "latest_B.pt"),stream.getvalue())
    if completed in (0,1) or completed%65==0:
        atomic_bytes(output/"checkpoints"/("initial_zero.pt" if completed==0 else "first_real_update.pt" if completed==1 else "epoch_%03d.pt"%(completed//65)),stream.getvalue())
    receipt = dict(utc=utc(),version=VERSION,cursor=current,checkpoint=checkpoint,state_receipt=state,
        source_receipt_sha256=tree_hash(record) if record else None,full_contract_pass=False)
    history = output/"checkpoints/history"/("update_%05d.json"%completed)
    assert not history.exists()
    dump(history,receipt)
    dump(output/"checkpoints/latest_commit.json",receipt)
    return receipt


def train(protocol_path, output, resume=False):
    protocol = read(protocol_path)
    output = Path(output)
    if resume:
        assert output.is_dir()
    else:
        output.mkdir(exist_ok=False)
        for name in ("checkpoints","checkpoints/history","source_receipts","optimizer_executions","final_testing"):
            (output/name).mkdir()
    native = process_identity()
    dump(output/"actual_compute_process_identity.json",native)
    completed,stage = 0,"REGISTERED_INPUT_VALIDATION"
    def progress(**extra):
        dump(output/"progress.json",dict(utc=utc(),stage=stage,cursor=current_cursor(completed,protocol),
            actual_native_identity=native,resources=memory(),source_packages=65,full_contract_pass=False,**extra))
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        assert torch.version.cuda is None
        for relative,sha in protocol["source_bindings_sha256"].items():
            assert file_sha(ROOT/relative)==sha,relative
        manifest = read(ROOT/protocol["manifest_path"])
        assert len(manifest)==65 and protocol["maximum_complete_sources"]==65*80
        random.seed(303); np.random.seed(303); torch.manual_seed(303)
        model = CurrentDataModel()
        base = torch.load(ROOT/protocol["initial_checkpoint_path"],map_location="cpu")
        assert file_sha(ROOT/protocol["initial_checkpoint_path"])==protocol["initial_checkpoint_sha256"]
        missing,unexpected = model.load_state_dict(base["state_dict"],strict=False)
        assert set(missing)=={"extra_process_heads.weight","extra_process_heads.bias"} and not unexpected
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4,foreach=False)
        geom_parameters = [p for n,p in model.named_parameters() if not n.startswith(("material_encoder.","material_head.","contract_attributes.","extra_process_heads."))]
        aux_parameters = [p for n,p in model.named_parameters() if n.startswith(("material_encoder.","material_head.","contract_attributes.","extra_process_heads."))]
        if resume:
            descriptor = read(output/"checkpoints/latest_commit.json")
            assert file_sha(descriptor["checkpoint"]["path"])==descriptor["checkpoint"]["sha256"]
            saved = torch.load(descriptor["checkpoint"]["path"],map_location="cpu")
            assert saved["protocol"]==protocol
            model.load_state_dict(saved["state_dict"],strict=True); optimizer.load_state_dict(saved["optimizer"])
            completed = saved["cursor"]["complete_sources"]
            random.setstate(saved["RNG"]["python"]); np.random.set_state(saved["RNG"]["numpy"]); torch.set_rng_state(saved["RNG"]["torch"])
            assert finite_state(model,optimizer)==saved["state_receipt"]
            intent_path = output/"optimizer_step_intent.json"
            assert not intent_path.exists() or read(intent_path)["update_after"]<=completed
        else:
            commit(output,model,optimizer,0,protocol)
        while completed<protocol["maximum_complete_sources"]:
            epoch,slot = completed//65,completed%65
            index = protocol["global_epoch_source_orders"][epoch][slot]
            entry = manifest[index]
            stage = "REAL_ALL_ORIGINAL_XYZ_FORWARD"
            progress(source_index_metadata_only=index,source_origin=entry["origin"])
            source = RawPointSource(ROOT,ROOT/entry["group_directory"])
            assert len(source.xyz_mm)==entry["original_points"]
            targets = read(ROOT/entry["target_path"])
            attribute = read(ROOT/entry["attribute_path"])
            optimizer.zero_grad(set_to_none=True)
            output_values = model(source)
            assert output_values["input_receipt"]["STEP_reads"]==0 and output_values["input_receipt"]["truth_point_queries"]==0
            geometry,gr = geometry_loss(output_values,targets,protocol)
            attributes,ar = attribute_loss(model,output_values,targets,attribute,gr["matched"])
            material,mr = ordered_material_loss(model,source,output_values,targets,attribute,gr["matched"])
            loss = geometry+.1*attributes+.1*material
            stage = "REAL_CURRENT_DATA_BACKWARD"
            progress(source_index_metadata_only=index,loss=float(loss.detach()))
            loss.backward()
            gradient = gradient_receipt(model)
            assert gradient["nonzero"] and torch.isfinite(loss)
            geom_norm = torch.nn.utils.clip_grad_norm_(geom_parameters,100.)
            aux_norm = torch.nn.utils.clip_grad_norm_(aux_parameters,100.)
            assert torch.isfinite(geom_norm) and torch.isfinite(aux_norm)
            before = tree_hash(model.state_dict())
            dump(output/"optimizer_step_intent.json",dict(update_before=completed,update_after=completed+1,
                source_index_metadata_only=index,before_model_sha256=before,utc=utc(),actual_identity=native))
            optimizer.step()
            after = tree_hash(model.state_dict())
            assert before!=after
            invocation = dict(actual_optimizer_invocations=1,optimizer_step_returned=True,update_after=completed+1,
                before_model_sha256=before,after_model_sha256=after,parameter_change_verified=True,actual_identity=native)
            dump(output/"optimizer_executions"/("update_%05d.json"%(completed+1)),invocation)
            record = dict(utc=utc(),source_index_metadata_only=index,source_origin=entry["origin"],actual_optimizer_execution=invocation,
                input_receipt=output_values["input_receipt"],geometry_loss_receipt=gr,attribute_loss_receipt=ar,material_loss_receipt=mr,
                gradient_receipt=gradient,geom_preclip_norm=float(geom_norm),aux_preclip_norm=float(aux_norm),
                total_loss=float(loss.detach()),development_FIT=True,formal_accuracy=False,full_contract_pass=False)
            commit(output,model,optimizer,completed+1,protocol,record)
            dump(output/"source_receipts"/("update_%05d.json"%(completed+1)),record)
            completed += 1
            stage = "REAL_CURRENT65_UPDATE_COMMITTED"
            progress(last_source=index,last_origin=entry["origin"],last_total_loss=record["total_loss"])
            print("CURRENT65_REAL_UPDATE",completed,"ORIGIN",entry["origin"],"SOURCE",index,"KNOWN_FIELDS",ar["known_state_cells"],"ORDERED_POINTS",mr["ordered_material_candidate_points"],flush=True)
            del source,targets,attribute,output_values,geometry,attributes,material,loss
        model.eval()
        totals = {name:{k:0 for k in ("G","predicted","TP","FP")} for name in CATEGORIES}
        testing = []
        with torch.no_grad():
            for index,entry in enumerate(manifest):
                stage = "FULL65_FROZEN_FINAL_GEOMETRY_DEVELOPMENT_TEST"
                progress(test_source=index,tested_sources=len(testing))
                source = RawPointSource(ROOT,ROOT/entry["group_directory"])
                targets = read(ROOT/entry["target_path"])
                prediction = model(source)
                decoded = decode_current_predictions(prediction)
                score = score_current_source(decoded,targets)
                missing = targets["missing_or_process_conflict_children"]
                for name,amount in targets["missing_category_denominators"].items():
                    score["categories"][name]["G"] += amount
                score.update(original_declared_G=targets["original_declared_child_count"],missing_original_units_retained_as_failure=missing,
                    all16_attribute_full_G_accuracy="NOT_MEASURED_MISSING_COMPLETE_REVIEWED_TRUTH",final_acceptance=False)
                for name in CATEGORIES:
                    for key in totals[name]: totals[name][key] += score["categories"][name][key]
                testing.append(score)
                dump(output/"final_testing"/("source_%02d.json"%index),score)
                del source,targets,prediction,decoded
        for value in totals.values():
            value["valid_rate"] = value["TP"]/value["G"] if value["G"] else None
            value["false_rate_per_G"] = value["FP"]/value["G"] if value["G"] else None
        stage = "FULL65_DEVELOPMENT_TRAINING_AND_GEOMETRY_TEST_COMPLETE"
        dump(output/"receipt.json",dict(version=VERSION,utc=utc(),cursor=current_cursor(completed,protocol),
            tested_source_packages=len(testing),categories=totals,role="CURRENT65_DEVELOPMENT_FIT_NOT_FORMAL_ACCEPTANCE",
            original_declared_G=protocol["original_declared_children"],unknown_attribute_G_not_removed=True,
            all16_attribute_accuracy="NOT_MEASURED_COMPLETE_REVIEWED_TRUTH_MISSING",full_contract_pass=False,
            next_failed_full_round_action="REAL_GPT6_PRO_CONSULT_THEN_NEXT_REGISTERED_ROUND",real_pro_send_count=0))
        progress(tested_sources=65)
    except BaseException as error:
        dump(output/"failure.json",dict(utc=utc(),stage=stage,cursor=current_cursor(completed,protocol),
            exception=type(error).__name__,message=str(error),full_round_complete=False,full_contract_pass=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    train(args.protocol,args.output,args.resume)
