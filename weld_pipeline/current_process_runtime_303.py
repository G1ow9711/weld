"""Apply the same process convention to model predictions, without labels."""
import numpy as np
from .world_set_runtime_295 import decode_local_predictions
from .contract_curve_geometry import CircularArc, sample_arc
from .customer_process_rules_302 import is_long_enough


def normalize_process_paths(decoded):
    retained = []
    short = []
    for parent in decoded["parents"]:
        segments = parent["segments"]
        for c in segments:
            c["model_sequence_index"] = c["segment_index"]
        first = np.asarray(segments[0]["points"])[0]
        last = np.asarray(segments[-1]["points"])[-1]
        welds = {c["weld_type"] for c in segments}
        reverse = (last[2]>first[2]+1e-6) if welds=={"vertical"} else tuple(last)<tuple(first) if welds=={"horizontal"} else False
        if reverse:
            segments.reverse()
            parent["gaps"].reverse()
            parent["direction"] *= -1
            for c in segments:
                if c["kind"]=="arc":
                    original = CircularArc.from_parameters(c["arc_parameters"])
                    end = original.point_at(np.array([original.length_mm]))[0]
                    new_arc = CircularArc(original.center_mm,original.radius_mm,original.plane_normal,
                        (end-original.center_mm)/original.radius_mm,-original.sweep_rad,original.is_closed)
                    replacement = sample_arc(new_arc)
                    c.update(replacement)
                else:
                    c["points"].reverse()
                    c["endpoints_mm"].reverse()
                c["boundary_refs"].reverse()
        valid = []
        for c in segments:
            if not is_long_enough(c["kind"],c["length_mm"]):
                short.append(dict(query_index=parent["query_index"],model_sequence_index=c["model_sequence_index"],
                    kind=c["kind"],length_mm=c["length_mm"],published=False,reason="BELOW_GLOBAL_PROCESS_MINIMUM"))
            else:
                valid.append(c)
        for i,c in enumerate(valid): c.update(segment_index=i,segment_count=len(valid))
        if valid:
            parent.update(segments=valid,child_count=len(valid),path_reversed_for_process_rule=bool(reverse))
            retained.append(parent)
    decoded["parents"] = retained
    decoded["internal_unpublished_short_candidates"] = short
    decoded["total_prediction_children"] = sum(p["child_count"] for p in retained)+sum(p["counted_false_children"] for p in decoded["invalid_predictions"])
    decoded["present_parent_count"] = len(retained)+len(decoded["invalid_predictions"])
    decoded["process_rules_applied_without_GT_or_STEP"] = True
    return decoded


def decode_current_predictions(output):
    return normalize_process_paths(decode_local_predictions(output))
