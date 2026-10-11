"""Fresh original-XYZ prediction; no supervision, trained caches or GT counts."""
import gc
from collections import defaultdict
import numpy as np
import torch
from .carrier_boundary_model_328 import SEQUENCE_HALO_ROWS
from .carrier_observation_sequences_328 import OriginalXYZObservationProvider, SequenceLayout
from .carrier_pointer_geometry_329 import legal_terminal_mask, learned_parameters, relation_features
from .analytic_carrier_slots_327 import build_carriers
from .bounded_analytic_residuals_327 import assemble_corrected_parent
from .selected_slot_process_projection_329 import project_selected_parameters
from .fresh_analytic_carrier_provider_327 import fresh_analytic_carriers
from .current_process_runtime_327 import normalize_process_paths
from .raw_point_groups_285 import RawPointSource

VERSION = "FRESH_XYZ_ALL_SLOT_NEURAL_POINTER_SHARED_AUTHORITY_RUNTIME_329_v1"
PRESENCE_THRESHOLD = .5
LINK_THRESHOLD = .5
POINTER_END_ROLE_WEIGHT = 1.
INFERENCE_BLOCK_ROWS = 8192


def invalid_start(query, carrier, reason, weld_type=None, count=1):
    return dict(query_index=int(query), original_parent_query_index=int(query),
        predicted_child_count=int(count), counted_false_children=int(count),
        predicted_weld_types=[int(carrier.weld_type == "vertical") if weld_type is None else int(weld_type)]*count,
        predicted_kinds=[1 if carrier.kind == "line" else 2]*count,
        capacity_overflow=False, reason=reason, invalid_output_ledger_version=VERSION)


def decode_carrier_heads(model, carrier, layout, heads, first_query=0):
    """Every selected start searches the full finite terminal domain.

Adjacency is learned, then checked for exact shared authority or positive
forward gap. It never joins independent parents by coordinate proximity.
"""
    active = torch.from_numpy(layout.active_slots)
    ids = layout.slot_ids[layout.active_slots]
    flat = {key:value[active] for key,value in heads.items()}
    parameters = learned_parameters(heads,layout)[active]
    probability = flat["role_logits"].sigmoid()
    selected = torch.nonzero(probability[:,1] >= PRESENCE_THRESHOLD,as_tuple=False).flatten().tolist()
    children, invalid = [], []
    next_query = int(first_query)
    for start in selected:
        query = next_query; next_query += 1
        start_slot = int(ids[start])
        predicted_type = int(flat["weld_type_logits"][start].argmax())
        if predicted_type != int(carrier.weld_type == "vertical"):
            invalid.append(invalid_start(query,carrier,"PREDICTED_TYPE_CONFLICTS_WITH_FINITE_CARRIER",predicted_type))
            continue
        allowed = torch.from_numpy(legal_terminal_mask(carrier,layout,start_slot)[layout.active_slots])
        if not bool(allowed.any()):
            invalid.append(invalid_start(query,carrier,"NO_ORDERED_FINITE_TERMINAL"))
            continue
        logits = model.pointer_logits(flat["embeddings"][start],flat["terminal_keys"]) + POINTER_END_ROLE_WEIGHT*flat["role_logits"][:,2]
        end = int(logits.masked_fill(~allowed,-torch.inf).argmax())
        q0,q1 = float(parameters[start]),float(parameters[end])
        if q0 == q1 or (carrier.weld_type == "vertical" and q1 <= q0):
            invalid.append(invalid_start(query,carrier,"MODEL_SELECTED_ZERO_OR_UPWARD_INTERVAL"))
            continue
        children.append(dict(query_index=query,start=start,end=end,start_slot=start_slot,end_slot=int(ids[end]),
            start_q=q0,end_q=q1,direction=float(np.sign(q1-q0)),
            parent_start_probability=float(probability[start,3]),
            parent_end_probability=float(probability[end,4])))
    # Evaluate only pairs of predicted children, not all cells in a global graph.
    possible_edges = []
    for ai,a in enumerate(children):
        if a["parent_end_probability"] >= PRESENCE_THRESHOLD:
            continue
        for bi,b in enumerate(children):
            if ai == bi or b["parent_start_probability"] >= PRESENCE_THRESHOLD or a["direction"] != b["direction"]:
                continue
            shared = a["end_slot"] == b["start_slot"]
            gap = a["direction"]*(b["start_q"]-a["end_q"])
            if not shared and gap <= 0:
                continue
            extra = relation_features(parameters[a["end"]],parameters[b["start"]],shared,True)
            logits = model.relation_logits(flat["embeddings"][a["start"]],flat["embeddings"][a["end"]],
                flat["embeddings"][b["start"]],flat["embeddings"][b["end"]],extra)
            scores = logits.softmax(-1)
            relation = int(scores.argmax())
            confidence = float(scores[relation])
            if confidence < LINK_THRESHOLD or relation == 0 or (relation == 1) != shared:
                continue
            possible_edges.append((confidence,ai,bi,"SHARED" if shared else "POSITIVE_GAP"))
    outgoing,incoming = {},{}
    for confidence,ai,bi,gap in sorted(possible_edges,key=lambda e:(-e[0],e[1],e[2])):
        if ai in outgoing or bi in incoming:
            continue
        # Strict forward children imply acyclicity; still check selected links.
        cursor=bi; cycle=cursor == ai
        while not cycle and cursor in outgoing:
            cursor=outgoing[cursor][0]; cycle=cursor == ai
        if not cycle:
            outgoing[ai]=(bi,gap); incoming[bi]=ai
    parents,visited,reserved = [],set(),set()
    for root in (i for i in range(len(children)) if i not in incoming):
        chain,gaps = [],[]; cursor=root
        while True:
            if cursor in visited:
                raise ValueError("Prediction adjacency repeated a child")
            visited.add(cursor); chain.append(children[cursor])
            if cursor not in outgoing:
                break
            cursor,gap=outgoing[cursor]; gaps.append(gap)
        first=chain[0]; nodes={}
        for child in chain:
            for slot,q in ((child["start_slot"],child["start_q"]),(child["end_slot"],child["end_q"])):
                if slot in nodes and nodes[slot]["q_mm"] != q:
                    raise ValueError("Shared slot acquired multiple parameter values")
                nodes[slot]=dict(node_id=slot,slot_index=slot,q_mm=q)
        try:
            if reserved.intersection(nodes):
                raise ValueError("INDEPENDENT_PARENT_SELECTED_ALREADY_RESERVED_AUTHORITY_SLOT")
            refs=[dict(start_node=c["start_slot"],end_node=c["end_slot"]) for c in chain]
            raw=flat["residual"][first["start"]].detach().numpy().astype(np.float64)
            fitted,projection=project_selected_parameters(carrier,list(nodes.values()),refs,gaps,raw)
            parent=assemble_corrected_parent(carrier,fitted,refs,gaps,raw,False)
            authority=list(parent["boundary_nodes"])
            boundary_rank={value:index for index,value in enumerate(authority)}
            for index,(segment,refs_one) in enumerate(zip(parent["segments"],refs)):
                segment.update(segment_index=index,model_sequence_index=index,segment_count=len(chain),
                    endpoints_mm=[segment["points"][0],segment["points"][-1]],
                    boundary_refs=[boundary_rank[refs_one["start_node"]],boundary_rank[refs_one["end_node"]]],
                    attributes={},formal_E_status="UNKNOWN_UNREVIEWED")
            parent.update(query_index=first["query_index"],original_parent_query_index=first["query_index"],
                direction=int(first["direction"]),predicted_mode=parent["mode"],
                boundaries_mm=[parent["boundary_nodes"][k]["point_mm"] for k in authority],
                present_probability=float(probability[first["start"],1]),
                slot_process_projection=projection,GT_or_STEP_queries=0)
            reserved.update(nodes)
            parents.append(parent)
        except ValueError as error:
            invalid.append(invalid_start(first["query_index"],carrier,str(error),count=len(chain)))
    if len(visited) != len(children):
        raise ValueError("Prediction ledger contains an unvisited cycle")
    counted=sum(p["child_count"] for p in parents)+sum(v["counted_false_children"] for v in invalid)
    if counted != len(selected):
        raise ValueError("Selected child-start output disappeared from the false ledger")
    return dict(parents=parents,invalid_predictions=invalid,total_prediction_children=counted,
        selected_start_slots=len(selected),next_query_index=next_query)


def infer_complete_carrier_heads(model,tokens,provider,layout):
    pieces=defaultdict(list); calls=0
    for first in range(0,layout.rows,INFERENCE_BLOCK_ROWS):
        last=min(layout.rows,first+INFERENCE_BLOCK_ROWS)
        lo,hi=max(0,first-SEQUENCE_HALO_ROWS),min(layout.rows,last+SEQUENCE_HALO_ROWS)
        features,groups=provider.observations(layout,lo,hi)
        sequence=model.encode_sequence(torch.from_numpy(features),tokens[torch.from_numpy(groups)],tokens.mean(0))
        heads=model.heads(sequence[first-lo:last-lo]); calls+=1
        for key,value in heads.items():
            pieces[key].append(value)
    return {key:torch.cat(value,dim=0) for key,value in pieces.items()},calls


def predict_from_original_source(model,source,progress=None):
    if type(source) is not RawPointSource:
        raise ValueError("Only verified all-original XYZ source is accepted")
    report=progress or (lambda **values:None)
    # Every input undergoes the same generator, irrespective of its filename.
    records,generation=fresh_analytic_carriers(source.xyz_mm,
        lambda stage,**values:report(stage=stage,**values))
    carriers=build_carriers(records)
    del records
    gc.collect()
    provider=OriginalXYZObservationProvider(source)
    model.eval()
    parents,invalid,queries,slots,calls=[],[],0,0,0
    with torch.no_grad():
        tokens,context=model.context(source)
        for ordinal,carrier in enumerate(carriers):
            layout=SequenceLayout.from_carrier(carrier)
            heads,number=infer_complete_carrier_heads(model,tokens,provider,layout)
            decoded=decode_carrier_heads(model,carrier,layout,heads,queries)
            parents.extend(decoded["parents"]);invalid.extend(decoded["invalid_predictions"])
            queries=decoded["next_query_index"];slots+=carrier.slot_count;calls+=number
            del heads,decoded
            if ordinal%100 == 0 or ordinal+1 == len(carriers):
                report(stage="FRESH_ALL_CELL_FREE_NEURAL_PREDICTION",complete_carriers=ordinal+1,
                    original_carriers=len(carriers),complete_slots=slots,selected_start_outputs=queries,
                    sequence_forwards=calls)
    result=normalize_process_paths(dict(parents=parents,invalid_predictions=invalid))
    result.update(version=VERSION,full_original_context=context,fresh_XYZ_generation=generation,
        all_exact_alias_capacity_domains=len(carriers),all_declared_slots_processed=slots,
        sequence_forwards=calls,source_candidate_caches_read=False,training_feature_caches_read=False,
        target_count_used=False,target_endpoint_windows_used=False,GT_or_STEP_queries=0,
        signed_board_normal_accuracy="NOT_MEASURED",all_E16_accuracy="NOT_MEASURED",
        full_contract_pass=False)
    return result
