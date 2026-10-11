"""Original consuming decoder with streamed offline decision observations.

Gate identities, outputs, invalid counts and query order remain unchanged.
No observation is a GT comparison, free extraction score or closure proof.
"""
from collections import Counter
import torch
from .carrier_boundary_runtime_329 import (PRESENCE_THRESHOLD, LINK_THRESHOLD,
    POINTER_END_ROLE_WEIGHT, invalid_start, learned_parameters, legal_terminal_mask,
    relation_features, project_selected_parameters, assemble_corrected_parent, np)
from .streamed_role_gate_347 import require_recomputed_start_identity

ORIGINAL_GATE_CHECKED_DECODER_SOURCE_SHA256 = 'fc3df9e79a2fd165e529401eff5ed89f75b58f8a7d9f1794249e8d1f52276990'

def _observed_consuming_decode(model, carrier, layout, heads, gate, first_query, trace):
    """Every selected start searches the full finite terminal domain.

Adjacency is learned, then checked for exact shared authority or positive
forward gap. It never joins independent parents by coordinate proximity.
"""
    active = torch.from_numpy(layout.active_slots)
    ids = layout.slot_ids[layout.active_slots]
    # Consume tensors progressively; no complete input/output double retention.
    flat = {"role_logits": heads.pop("role_logits")[active]}
    probability = flat["role_logits"].sigmoid()
    selected = torch.nonzero(probability[:,1] >= PRESENCE_THRESHOLD,as_tuple=False).flatten().tolist()
    require_recomputed_start_identity(gate, ids, selected, layout)
    trace("SELECTED_START_SET", count=len(selected), first_query_index=int(first_query), slots=[int(ids[s]) for s in selected])
    if not selected:
        heads.clear()
        return dict(parents=[], invalid_predictions=[], total_prediction_children=0,
            selected_start_slots=0, next_query_index=int(first_query))
    parameters = learned_parameters(heads,layout)[active]
    # Queries are unused by the frozen decoder; pointer_logits derives them from
    # the selected start embedding. Fractions have already become parameters.
    heads.pop("terminal_queries", None)
    heads.pop("within_cell_fraction")
    for key in ("weld_type_logits", "residual", "terminal_keys", "embeddings"):
        flat[key] = heads.pop(key)[active]
    heads.clear()
    children, invalid = [], []
    next_query = int(first_query)
    for start in selected:
        query = next_query; next_query += 1
        start_slot = int(ids[start])
        predicted_type = int(flat["weld_type_logits"][start].argmax())
        trace("SELECTED_START", query_index=query, start_slot=start_slot, predicted_type=predicted_type)
        if predicted_type != int(carrier.weld_type == "vertical"):
            trace("START_TYPE_VETO", query_index=query, start_slot=start_slot, predicted_type=predicted_type)
            invalid.append(invalid_start(query,carrier,"PREDICTED_TYPE_CONFLICTS_WITH_FINITE_CARRIER",predicted_type))
            continue
        allowed = torch.from_numpy(legal_terminal_mask(carrier,layout,start_slot)[layout.active_slots])
        if not bool(allowed.any()):
            trace("START_NO_LEGAL_TERMINAL", query_index=query, start_slot=start_slot)
            invalid.append(invalid_start(query,carrier,"NO_ORDERED_FINITE_TERMINAL"))
            continue
        logits = model.pointer_logits(flat["embeddings"][start],flat["terminal_keys"]) + POINTER_END_ROLE_WEIGHT*flat["role_logits"][:,2]
        end = int(logits.masked_fill(~allowed,-torch.inf).argmax())
        trace("FULL_LEGAL_TERMINAL_SELECTED", query_index=query, start_slot=start_slot, end_slot=int(ids[end]), complete_legal_terminals=int(allowed.sum()))
        q0,q1 = float(parameters[start]),float(parameters[end])
        if q0 == q1 or (carrier.weld_type == "vertical" and q1 <= q0):
            trace("START_PARAMETER_VETO", query_index=query, start_slot=start_slot, end_slot=int(ids[end]), start_q_mm=q0, end_q_mm=q1)
            invalid.append(invalid_start(query,carrier,"MODEL_SELECTED_ZERO_OR_UPWARD_INTERVAL"))
            continue
        children.append(dict(query_index=query,start=start,end=end,start_slot=start_slot,end_slot=int(ids[end]),
            start_q=q0,end_q=q1,direction=float(np.sign(q1-q0)),
            parent_start_probability=float(probability[start,3]),
            parent_end_probability=float(probability[end,4])))
    for prechain_index, child in enumerate(children):
        trace("PRECHAIN_CHILD", prechain_index=prechain_index, **child)
    # Evaluate only pairs of predicted children, not all cells in a global graph.
    possible_edges = []
    for ai,a in enumerate(children):
        if a["parent_end_probability"] >= PRESENCE_THRESHOLD:
            trace("PAIR_A_PARENT_END_VETO", a=ai, a_query=a["query_index"], covers_all_b=len(children), parent_end_probability=a["parent_end_probability"])
            continue
        for bi,b in enumerate(children):
            if ai == bi or b["parent_start_probability"] >= PRESENCE_THRESHOLD or a["direction"] != b["direction"]:
                trace("PAIR_B_VETO", a=ai, b=bi, self_pair=ai == bi, parent_start_veto=b["parent_start_probability"] >= PRESENCE_THRESHOLD, direction_veto=a["direction"] != b["direction"])
                continue
            shared = a["end_slot"] == b["start_slot"]
            gap = a["direction"]*(b["start_q"]-a["end_q"])
            if not shared and gap <= 0:
                trace("PAIR_FINITE_AUTHORITY_ORDER_VETO", a=ai, b=bi, shared=shared, directed_gap_mm=gap)
                continue
            extra = relation_features(parameters[a["end"]],parameters[b["start"]],shared,True)
            logits = model.relation_logits(flat["embeddings"][a["start"]],flat["embeddings"][a["end"]],
                flat["embeddings"][b["start"]],flat["embeddings"][b["end"]],extra)
            scores = logits.softmax(-1)
            relation = int(scores.argmax())
            confidence = float(scores[relation])
            trace("PAIR_RELATION_HEAD", a=ai, b=bi, shared=shared, directed_gap_mm=gap, relation=relation, confidence=confidence, probabilities=scores.detach().tolist())
            if confidence < LINK_THRESHOLD or relation == 0 or (relation == 1) != shared:
                continue
            possible_edges.append((confidence,ai,bi,"SHARED" if shared else "POSITIVE_GAP"))
            trace("POSSIBLE_EDGE", a=ai, b=bi, confidence=confidence, gap_mode="SHARED" if shared else "POSITIVE_GAP")
    outgoing,incoming = {},{}
    for confidence,ai,bi,gap in sorted(possible_edges,key=lambda e:(-e[0],e[1],e[2])):
        if ai in outgoing or bi in incoming:
            trace("GREEDY_EDGE_AUTHORITY_CONFLICT", a=ai, b=bi, confidence=confidence, existing_outgoing=ai in outgoing, existing_incoming=bi in incoming)
            continue
        # Strict forward children imply acyclicity; still check selected links.
        cursor=bi; cycle=cursor == ai
        while not cycle and cursor in outgoing:
            cursor=outgoing[cursor][0]; cycle=cursor == ai
        if not cycle:
            outgoing[ai]=(bi,gap); incoming[bi]=ai
            trace("SELECTED_EDGE", a=ai, b=bi, confidence=confidence, gap_mode=gap)
        else:
            trace("GREEDY_EDGE_CYCLE_VETO", a=ai, b=bi, confidence=confidence)
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
        trace("PREDICTED_CHAIN", root_query=first["query_index"], queries=[c["query_index"] for c in chain], gaps=gaps, decoder_closed_flag=False)
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
            trace("BEFORE_PARAMETER_PROJECTION", root_query=first["query_index"], nodes=list(nodes.values()), refs=refs, gaps=gaps, root_residual=raw.tolist())
            fitted,projection=project_selected_parameters(carrier,list(nodes.values()),refs,gaps,raw)
            trace("AFTER_PARAMETER_PROJECTION", root_query=first["query_index"], fitted_nodes=fitted, projection=projection)
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
            trace("ASSEMBLED_PREDICTED_PARENT", root_query=first["query_index"], child_count=parent["child_count"], selected_authority_slots=list(nodes))
        except ValueError as error:
            trace("CHAIN_FALSE_OUTPUT_LEDGER", root_query=first["query_index"], counted_false_children=len(chain), reason=str(error), selected_authority_slots=list(nodes))
            invalid.append(invalid_start(first["query_index"],carrier,str(error),count=len(chain)))
    if len(visited) != len(children):
        raise ValueError("Prediction ledger contains an unvisited cycle")
    counted=sum(p["child_count"] for p in parents)+sum(v["counted_false_children"] for v in invalid)
    if counted != len(selected):
        raise ValueError("Selected child-start output disappeared from the false ledger")
    return dict(parents=parents,invalid_predictions=invalid,total_prediction_children=counted,
        selected_start_slots=len(selected),next_query_index=next_query)


def audit_gate_checked_decoder_decisions(model, carrier, layout, heads, gate, first_query=0, sink=None):
    if torch.is_grad_enabled():
        raise ValueError("Offline consuming decoder tracing requires no-grad inference")
    counts = Counter()
    def trace(stage, **values):
        counts[stage] += 1
        if sink is not None:
            sink(dict(stage=stage, carrier_ordinal=int(carrier.ordinal), **values))
    decoded = _observed_consuming_decode(model, carrier, layout, heads, gate, int(first_query), trace)
    assert counts["SELECTED_START"] == decoded["selected_start_slots"]
    assert not heads
    return decoded, dict(version="OFFLINE_ORIGINAL_CONSUMING_DECODER_DECISION_TRACE_352_v1",
        complete_decision_event_counts=dict(counts),
        selected_start_slots=decoded["selected_start_slots"],
        total_prediction_children=decoded["total_prediction_children"],
        complete_false_output_ledger_and_ordered_gate_identity_preserved=True,
        original_head_storage_consumed=True, detailed_records_streamed_to_sink=sink is not None,
        original_decoder_always_supplies_closed_flag_false=True,
        formal_geometry_accuracy_measured=False, formal_E16_accuracy_measured=False,
        full_contract_pass=False)
