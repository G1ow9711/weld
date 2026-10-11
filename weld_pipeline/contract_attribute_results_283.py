"""Fixed-side field export and full-denominator cell scoring, no GT prediction."""
import math
import numpy as np
from .contract_attribute_heads_283 import FIELDS, CONTINUOUS_FIELDS, NORMAL_FIELDS, BINARY_FIELDS, STATES


def prediction_rows(output):
    arrays = {key: value.detach().cpu().numpy() for key, value in output.items()}
    logits = arrays["state_logits"]
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite prediction cells cannot be omitted from the contract output")
    rows = []
    for index in range(len(logits)):
        record = {}
        for column, field in enumerate(FIELDS):
            state = STATES[int(logits[index, column].argmax())]
            value = None
            reason = None
            if state == "value":
                if field in NORMAL_FIELDS:
                    value = arrays["normals"][index, NORMAL_FIELDS.index(field)].astype(float)
                    if abs(float(np.linalg.norm(value)) - 1) > 0.001:
                        state, value, reason = "unknown", None, "UNDEFINED_OR_NONUNIT_DIRECTED_NORMAL"
                    else:
                        value = value.tolist()
                elif field in BINARY_FIELDS:
                    value = bool(arrays["binary_logits"][index, BINARY_FIELDS.index(field)].argmax())
                else:
                    value = float(arrays["continuous_m"][index, CONTINUOUS_FIELDS.index(field)]) * 1000
            record[field] = dict(status=state, value=value, reason=reason)
        rows.append(record)
    return rows


def field_pass(field, prediction, truth, *, geometry_passed):
    if not geometry_passed or not truth or not prediction or not truth.get("truth_available", False):
        return False
    state = truth.get("status")
    if state == "unknown" or prediction.get("status") != state:
        return False
    if state in ("no_object", "out_of_range", "not_applicable"):
        return True
    if state != "value":
        return False
    predicted, expected = prediction.get("value"), truth.get("value")
    if field in BINARY_FIELDS:
        return type(predicted) is bool and type(expected) is bool and predicted == expected
    if field in NORMAL_FIELDS:
        try:
            p, q = np.asarray(predicted, float), np.asarray(expected, float)
        except (TypeError, ValueError):
            return False
        if p.shape != (3,) or q.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(q).all():
            return False
        a, b = float(np.linalg.norm(p)), float(np.linalg.norm(q))
        if abs(a - 1) > 0.001 or abs(b - 1) > 0.001:
            return False
        angle = math.degrees(math.atan2(float(np.linalg.norm(np.cross(p, q))), float(p @ q)))
        return angle <= 3
    if type(predicted) not in (int, float) or type(expected) not in (int, float):
        return False
    if not math.isfinite(predicted) or not math.isfinite(expected) or predicted < 0 or expected < 0:
        return False
    if field.startswith("thickness") and (predicted <= 0 or expected <= 0):
        return False
    return abs(predicted - expected) <= (1 if field.startswith("thickness") else 3)


def complete_child_attribute_rates(original_children):
    """Each row is one original child, including geometric misses/unknowns.

    Each field's sample list must retain EVERY frozen mandatory query position.
    The caller must supply that full original inventory, not only matches.
    """
    children = list(original_children)
    identities = [row["truth_child_id"] for row in children]
    if not children or len(set(identities)) != len(identities):
        raise ValueError("Unique complete original child inventory required")
    counts = {field: 0 for field in FIELDS}
    for row in children:
        if set(row["field_samples"]) != set(FIELDS):
            raise ValueError("Missing fields cannot reduce any contract denominator")
        for field in FIELDS:
            samples = row["field_samples"][field]
            passed = bool(samples) and all(field_pass(field, item.get("prediction"), item.get("truth"),
                geometry_passed=row["geometry_passed"]) for item in samples)
            counts[field] += int(passed)
    return dict(original_children=len(children), field_results={field: dict(passed=counts[field],
        full_original_denominator=len(children), rate=counts[field] / len(children),
        strictly_over_95_percent=counts[field] / len(children) > 0.95) for field in FIELDS},
        geometry_misses_and_unknown_truth_predictions_retained=True, side_exchange_used=False,
        all_original_inventory_externally_verified=False, full_contract_pass=False)
