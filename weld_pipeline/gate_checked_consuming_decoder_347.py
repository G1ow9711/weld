"""Frozen consuming decoder with fail-closed ordered-start identity verification."""
import torch
from .carrier_boundary_runtime_329 import (PRESENCE_THRESHOLD, LINK_THRESHOLD,
    POINTER_END_ROLE_WEIGHT, invalid_start, learned_parameters, legal_terminal_mask,
    relation_features, project_selected_parameters, assemble_corrected_parent, np)
from .streamed_role_gate_347 import require_recomputed_start_identity

ORIGINAL_CONSUMING_DECODER_SOURCE_SHA256 = 'c8d8e68c73b2c6a4d926bbcbfbd0f492f2eb3eb21630d8e5fce338605c120ebd'

def decode_gate_checked_consuming_heads(model, carrier, layout, heads, gate, first_query=0):
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
