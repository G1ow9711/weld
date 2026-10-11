"""Rebuild published process chains after filtering and whole-parent reversal.

Pure prediction postprocessing. No labels or STEP input. Previous303 remains
immutable; outputs carry a distinct version and preserve filtered candidates.
"""
from copy import deepcopy
import numpy as np
from .world_set_runtime_295 import decode_local_predictions
from .contract_curve_geometry import CircularArc,sample_arc,verify_sampled_arc
from .customer_process_rules_302 import is_long_enough

VERSION="PREDICTED_CONTIGUOUS_PROCESS_COMPONENTS_324_v1"
ENDPOINT_PAIRS=(
    ("front_wall","back_wall"),("start_height","end_height"),
    ("start_hole_height","end_hole_height"),("start_base_edge","end_base_edge"),
    ("FrontWallDistance","BackWallDistance"),("startHeight","endHeight"),
    ("StartViaHoleHeight","EndViaHoleHeight"),
    ("StartBasePlateEdgeDistance","EndBasePlateEdgeDistance"),
    ("WorkRangeFrontSplit","WorkRangeBackSplit"),
)


def reverse_segment(child):
    old_endpoints=[list(child["points"][-1]),list(child["points"][0])]
    if child["kind"]=="arc":
        original=CircularArc.from_parameters(child["arc_parameters"])
        end=np.asarray(old_endpoints[0],np.float64)
        arc=CircularArc(original.center_mm,original.radius_mm,original.plane_normal,
            (end-original.center_mm)/original.radius_mm,-original.sweep_rad,original.is_closed)
        replacement=sample_arc(arc)
        replacement["points"][0],replacement["points"][-1]=old_endpoints
        verify_sampled_arc(replacement)
        child.update(replacement)
    else:
        child["points"].reverse()
        child["endpoints_mm"].reverse()
    child["boundary_refs"].reverse()
    attributes=child.get("attributes")
    if isinstance(attributes,dict):
        for a,b in ENDPOINT_PAIRS:
            if a in attributes or b in attributes:
                attributes[a],attributes[b]=attributes.get(b),attributes.get(a)
    # Reverse only along-path positions. Board order and signed vectors stay.
    material=child.get("unordered_material_estimate")
    if isinstance(material,dict):
        for field in ("thickness_mm","CAD_outward_unit_normals_world"):
            if field in material:material[field]=list(reversed(material[field]))
        if "fractions" in material:
            material["fractions"]=[1-f for f in reversed(material["fractions"])]
    child["path_reversed_for_process_rule"]=True


def rebuild_component(parent,segments,gaps,component_index,whole_reversed,closed):
    assert len(gaps)==len(segments)-1
    boundaries=[]
    for i,child in enumerate(segments):
        start,end=[np.asarray(child["points"][j],np.float64) for j in (0,-1)]
        if i and gaps[i-1]=="SHARED":
            start_ref=segments[i-1]["boundary_refs"][1]
            if not np.array_equal(start,np.asarray(boundaries[start_ref])):
                raise ValueError("SHARED_PREDICTED_BOUNDARY_COORDINATES_DISAGREE")
        else:
            if i and gaps[i-1]!="POSITIVE_GAP":raise ValueError("UNKNOWN_PREDICTED_GAP")
            start_ref=len(boundaries);boundaries.append(start.tolist())
        if closed and i==len(segments)-1:
            end_ref=segments[0]["boundary_refs"][0] if i else start_ref
            if not np.array_equal(end,np.asarray(boundaries[end_ref])):
                raise ValueError("PREDICTED_CLOSURE_COORDINATES_DISAGREE")
        else:
            end_ref=len(boundaries);boundaries.append(end.tolist())
        child.update(segment_index=i,segment_count=len(segments),boundary_refs=[start_ref,end_ref])
    result=dict(parent)
    result.update(segments=segments,gaps=gaps,boundaries_mm=boundaries,child_count=len(segments),
        is_closed=closed,turns=int(closed),path_reversed_for_process_rule=whole_reversed,
        prediction_component_index=component_index,original_parent_query_index=parent["query_index"],
        published_endpoint_coordinates_mm=[segments[0]["points"][0],segments[-1]["points"][-1]],
        relation_rebuild_version=VERSION,formal_E_status="UNKNOWN_UNREVIEWED")
    return result


def normalize_process_paths(decoded):
    decoded=deepcopy(decoded)
    retained=[];short=[]
    for parent in decoded["parents"]:
        segments=parent["segments"]
        assert segments and len(parent["gaps"])==len(segments)-1
        for child in segments:
            child["model_sequence_index"]=child["segment_index"]
        keep=[is_long_enough(c["kind"],c["length_mm"]) for c in segments]
        for child,valid in zip(segments,keep):
            if not valid:
                short.append(dict(query_index=parent["query_index"],model_sequence_index=child["model_sequence_index"],
                    kind=child["kind"],length_mm=child["length_mm"],published=False,
                    original_candidate=deepcopy(child),reason="BELOW_GLOBAL_PROCESS_MINIMUM"))
        if not any(keep):continue
        vertical=[c for c,valid in zip(segments,keep) if valid and c["weld_type"]=="vertical"]
        rising=[float(c["points"][-1][2]-c["points"][0][2]) for c in vertical]
        whole_reverse=all(v>1e-6 for v in rising) if vertical else tuple(segments[-1]["points"][-1])<tuple(segments[0]["points"][0])
        if whole_reverse:
            segments.reverse();keep.reverse();parent["gaps"].reverse()
            parent["direction"]*=-1
            for child in segments:reverse_segment(child)
        reason=None
        for child,valid in zip(segments,keep):
            if valid and child["weld_type"]=="vertical":
                z=np.asarray(child["points"],np.float64)[:,2]
                if z[0]-z[-1]<=1e-6 or np.max(np.diff(z))>1e-6:
                    reason="NO_WHOLE_PARENT_ORIENTATION_MAKES_EVERY_VERTICAL_CHILD_DOWNWARD"
                    break
        if reason:
            selected=[c for c,valid in zip(segments,keep) if valid]
            decoded["invalid_predictions"].append(dict(query_index=parent["query_index"],
                predicted_child_count=len(selected),counted_false_children=len(selected),
                predicted_weld_types=[int(c["weld_type"]=="vertical") for c in selected],
                predicted_kinds=[1 if c["kind"]=="line" else 2 for c in selected],
                capacity_overflow=False,reason=reason,retained_unresolved_candidates=selected))
            continue
        gaps=list(parent["gaps"])
        complete=all(keep)
        if parent["is_closed"] and not complete:
            # Start just after a removed child, retaining wraparound adjacency.
            first_removed=keep.index(False);shift=(first_removed+1)%len(segments)
            edges=gaps+["SHARED"]
            segments=segments[shift:]+segments[:shift]
            keep=keep[shift:]+keep[:shift]
            edges=edges[shift:]+edges[:shift]
            gaps=edges[:-1]
        runs=[];run=[];run_gaps=[]
        for i,(child,valid) in enumerate(zip(segments,keep)):
            if not valid:
                if run:runs.append((run,run_gaps));run=[];run_gaps=[]
                continue
            if run:run_gaps.append(gaps[i-1])
            run.append(child)
        if run:runs.append((run,run_gaps))
        for component,(run,run_gaps) in enumerate(runs):
            retained.append(rebuild_component(parent,run,run_gaps,component,whole_reverse,
                bool(parent["is_closed"] and complete and len(runs)==1)))
    decoded.update(version=VERSION,parents=retained,internal_unpublished_short_candidates=short,
        total_prediction_children=sum(p["child_count"] for p in retained)+sum(p["counted_false_children"] for p in decoded["invalid_predictions"]),
        present_parent_count=len(retained)+len(decoded["invalid_predictions"]),
        process_rules_applied_without_GT_or_STEP=True,short_filter_relations_rebuilt=True,
        board_side_exchange_used=False,signed_material_normal_flip_used=False,full_contract_pass=False)
    return decoded


def decode_current_predictions(output):
    return normalize_process_paths(decode_local_predictions(output))
