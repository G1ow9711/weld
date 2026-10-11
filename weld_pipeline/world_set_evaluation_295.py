"""Conservative child geometry FIT; frozen unknown contract conventions explicit."""
import heapq,math
import numpy as np
from .world_set_evaluation_285 import CATEGORIES,category_keys,reference_segment,distance_to_primitive
from .contract_curve_geometry import CircularArc

VERSION="MAX_VALID_CHILD_CARDINALITY_THEN_MIN_ERROR_CONSERVATIVE_FIT_295_v1"

def maximum_valid_minimum_cost(npred,ntruth,edges):
    """Unit-capacity successive shortest augmenting paths; no forbidden forced edges."""
    count=npred+ntruth+2; start=count-2; end=count-1
    graph=[[] for _ in range(count)]
    def add(a,b,cost,tag=None):
        graph[a].append([b,1,cost,len(graph[b]),tag])
        graph[b].append([a,0,-cost,len(graph[a])-1,None])
    for p in range(npred): add(start,p,0.)
    for t in range(ntruth): add(npred+t,end,0.)
    for p,t,cost in edges: add(p,npred+t,float(cost),(p,t))
    potentials=[0.]*count
    while True:
        distances=[math.inf]*count; predecessor=[None]*count; distances[start]=0.; queue=[(0.,start)]
        while queue:
            d,a=heapq.heappop(queue)
            if d>distances[a]+1e-12: continue
            for index,(b,capacity,cost,reverse,tag) in enumerate(graph[a]):
                if not capacity: continue
                new=d+cost+potentials[a]-potentials[b]
                if new<distances[b]-1e-12:
                    distances[b]=new; predecessor[b]=(a,index); heapq.heappush(queue,(new,b))
        if predecessor[end] is None: break
        for node,value in enumerate(distances):
            if math.isfinite(value): potentials[node]+=value
        node=end
        while node!=start:
            a,index=predecessor[node]; edge=graph[a][index]
            edge[1]-=1; graph[node][edge[3]][1]+=1; node=a
    return [edge[4] for p in range(npred) for edge in graph[p] if edge[4] is not None and edge[1]==0]

def geometry_error(prediction,truth,reference,allow_reverse):
    if prediction["weld_type"]!=truth["weld_type"] or prediction["kind"]!=truth["original_kind"]: return None
    a=np.asarray(prediction["points"],np.float64); b=np.asarray(reference["points"],np.float64)
    if bool(prediction.get("is_closed",False))!=bool(reference.get("is_closed",False)): return None
    closed=bool(reference.get("is_closed",False))
    direct=np.linalg.norm(a[[0,-1]]-b[[0,-1]],axis=1)
    reversed_error=np.linalg.norm(a[[0,-1]]-b[[-1,0]],axis=1)
    reverse=allow_reverse and reversed_error.max()<direct.max()
    endpoint=reversed_error if reverse else direct
    if not closed and endpoint.max()>3: return None
    if prediction["kind"]=="arc":
        pa,ta=[CircularArc.from_parameters(v["arc_parameters"]) for v in (prediction,reference)]
        if pa.is_closed!=ta.is_closed: return None
        # Analytic arc export already rejects excess winding/backtracking.
        if abs(pa.sweep_rad)>2*math.pi+1e-12 or abs(ta.sweep_rad)>2*math.pi+1e-12: return None
    whole=float(max(distance_to_primitive(a,reference).max(),distance_to_primitive(b,prediction).max()))
    if whole>3: return None
    return dict(endpoint_errors_mm=endpoint.tolist() if not closed else None,bidirectional_sampled_maximum_mm=whole,
        whole_open_parent_reversed=bool(reverse),closed_start_free=closed,sampling_step_mm=1.,geometry_cost=float((endpoint.sum() if not closed else 0)+whole))

def score_source(decoded,targets):
    truth=[]; predicted=[]
    for pi,parent in enumerate(targets["parents"]):
        for ci,child in enumerate(parent["children"]): truth.append((pi,ci,child,reference_segment(child)))
    for pi,parent in enumerate(decoded["parents"]):
        for ci,child in enumerate(parent["segments"]): predicted.append((pi,ci,child))
    assert len(truth)==targets["original_children"]
    edges=[]; accepted={}
    for p,(pi,ci,child) in enumerate(predicted):
        parent=decoded["parents"][pi]
        for t,(ti,tci,target,reference) in enumerate(truth):
            tp=targets["parents"][ti]
            # Whole reversal is unambiguous for a single process unit. For multi
            # units keep ordered orientation until a complete legal D1 mapping
            # is frozen; do not independently reverse each child.
            reverse=parent["child_count"]==1 and tp["original_child_count"]==1
            error=geometry_error(child,target,reference,reverse)
            if error is not None:
                edges.append((p,t,error["geometry_cost"])); accepted[(p,t)]=error
    matches=maximum_valid_minimum_cost(len(predicted),len(truth),edges)
    categories={key:dict(G=0,predicted=0,TP=0,FP=0) for key in CATEGORIES}
    for _,_,child,_ in truth:
        for key in category_keys(child["weld_type"],child["original_kind"]): categories[key]["G"]+=1
    for _,_,child in predicted:
        for key in category_keys(child["weld_type"],child["kind"]): categories[key]["predicted"]+=1
    for invalid in decoded["invalid_predictions"]:
        for child in range(invalid["counted_false_children"]):
            weld=invalid["predicted_weld_types"][min(child,len(invalid["predicted_weld_types"])-1)]
            kind=invalid["predicted_kinds"][min(child,len(invalid["predicted_kinds"])-1)]
            keys=category_keys("vertical" if weld else "horizontal","line" if kind==1 else "arc") if kind in (1,2) else ("overall","vertical" if weld else "horizontal")
            for key in keys: categories[key]["predicted"]+=1
    details=[]
    for p,t in matches:
        pi,ci,child,_=truth[t]
        for key in category_keys(child["weld_type"],child["original_kind"]): categories[key]["TP"]+=1
        details.append(dict(prediction_row=p,truth_row=t,original_child_id=child["original_child_id"],**accepted[(p,t)]))
    for value in categories.values(): value["FP"]=value["predicted"]-value["TP"]
    assert categories["overall"]["predicted"]==decoded["total_prediction_children"]
    return dict(version=VERSION,categories=categories,matches=details,original_children=len(truth),original_parents=len(targets["parents"]),
        invalid_parent_hypotheses=len(decoded["invalid_predictions"]),parent_child_count_equality_gate_removed=True,
        maximum_valid_matching_cardinality=len(matches),all_published_invalid_duplicate_low_confidence_predictions_retained=True,
        D1_equivalent_merge_thresholds="UNFROZEN_NO_INVENTED_MERGE",D2_weld_side="UNKNOWN_CUSTOMER_TRUTH",D3_multi_unit_reversal="CONSERVATIVE_ORDERED_PENDING_PROCESS_MAPPING",
        hard_group_truth_and_direction_dictionary_frozen=False,complete_contract_geometry_scorer=False,formal_E_reviewed_cells=0,full_contract_pass=False)

def aggregate_sources(scores):
    categories={key:{field:sum(s["categories"][key][field] for s in scores) for field in ("G","predicted","TP","FP")} for key in CATEGORIES}
    for value in categories.values():
        g=value["G"]; valid=value["TP"]/g if g else None; false=value["FP"]/g if g else None
        value.update(valid_rate=valid,false_rate_per_G=false,status="PASS" if g and valid>.95 and false<=.05 else "FAIL" if g or value["FP"] else "NOT_EVALUATED_NO_GT")
    return dict(version=VERSION,role="ALL25_FIT_NOT_INDEPENDENT_ACCEPTANCE",categories=categories,
        geometry_development_gates_pass=all(v["status"]=="PASS" for v in categories.values() if v["G"]),
        D1_and_business_direction_protocol_incomplete=True,hard_groups_and_G0_scope_missing=True,formal_E_reviewed_cells=0,
        named_board_signed_normal_accuracy_not_measured=True,independent_800_100_200_available=False,full_contract_pass=False)
