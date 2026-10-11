"""Development geometry checks with explicit downward vertical direction."""
import numpy as np
from .world_set_evaluation_295 import maximum_valid_minimum_cost,geometry_error
from .world_set_evaluation_285 import CATEGORIES,category_keys,reference_segment


def score_current_source(decoded,targets):
    truth = [(pi,ci,c,reference_segment(c)) for pi,p in enumerate(targets["parents"]) for ci,c in enumerate(p["children"])]
    predicted = [(pi,ci,c) for pi,p in enumerate(decoded["parents"]) for ci,c in enumerate(p["segments"])]
    edges,accepted = [],{}
    for i,(pi,ci,c) in enumerate(predicted):
        for j,(ti,tci,target,reference) in enumerate(truth):
            # Explicit downwards process direction removes vertical reversal.
            reverse = target["weld_type"]=="horizontal" and targets["parents"][ti]["original_child_count"]==1 and decoded["parents"][pi]["child_count"]==1
            error = geometry_error(c,target,reference,reverse)
            if error is not None:
                if c["weld_type"]=="vertical" and np.asarray(c["points"])[-1,2]>np.asarray(c["points"])[0,2]+1e-6:
                    continue
                edges.append((i,j,error["geometry_cost"]))
                accepted[(i,j)] = error
    matches = maximum_valid_minimum_cost(len(predicted),len(truth),edges)
    categories = {name:{key:0 for key in ("G","predicted","TP","FP")} for name in CATEGORIES}
    for _,_,c,_ in truth:
        for name in category_keys(c["weld_type"],c["original_kind"]): categories[name]["G"]+=1
    for _,_,c in predicted:
        for name in category_keys(c["weld_type"],c["kind"]): categories[name]["predicted"]+=1
    for invalid in decoded["invalid_predictions"]:
        for index in range(invalid["counted_false_children"]):
            weld = invalid["predicted_weld_types"][min(index,len(invalid["predicted_weld_types"])-1)]
            kind = invalid["predicted_kinds"][min(index,len(invalid["predicted_kinds"])-1)]
            names = category_keys("vertical" if weld else "horizontal","line" if kind==1 else "arc") if kind in (1,2) else ("overall","vertical" if weld else "horizontal")
            for name in names: categories[name]["predicted"]+=1
    details = []
    for i,j in matches:
        _,_,child,_ = truth[j]
        for name in category_keys(child["weld_type"],child["original_kind"]): categories[name]["TP"]+=1
        details.append(dict(prediction_row=i,truth_row=j,original_child_id=child["original_child_id"],**accepted[(i,j)]))
    for value in categories.values(): value["FP"] = value["predicted"]-value["TP"]
    assert categories["overall"]["predicted"]==decoded["total_prediction_children"]
    return dict(categories=categories,matches=details,original_known_children=len(truth),
        explicit_vertical_down_direction_checked=True,role="CURRENT_DATA_DEVELOPMENT_GEOMETRY_DIAGNOSTIC",
        whole_parent_topology_weld_side_and_all16_acceptance_not_complete=True,full_contract_pass=False)
