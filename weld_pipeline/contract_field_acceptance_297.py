"""E2/E3/E4 component checks. Reviewed truth and frozen semantics are required.

This module does not train, map unordered CAD materials to plate sides, select
geometry matches, or certify the full contract. Geometry matches and synchronized
attribute references must be supplied by the separately frozen D1-D4 evaluator.
"""
import math

FIELDS = (
    "thickness_1", "thickness_2", "normal_1", "normal_2", "front_wall",
    "back_wall", "start_height", "end_height", "start_hole_height",
    "end_hole_height", "start_base_edge", "end_base_edge", "has_eave",
    "hole_edge", "side_plate_split", "segment_reason",
)


def draft_dictionary():
    fields = {}
    for name in FIELDS:
        kind = "directed_unit_vector" if name.startswith("normal_") else (
            "boolean" if name in ("has_eave", "side_plate_split") else (
                "UNFROZEN" if name in ("hole_edge", "segment_reason") else "distance_mm"))
        fields[name] = dict(
            kind=kind, semantics_frozen=False, allowed_states=["value", "unknown"],
            tolerance_mm=(1 if name.startswith("thickness_") else 3) if kind == "distance_mm" else None,
            angle_tolerance_degrees=3 if kind == "directed_unit_vector" else None,
            unit_length_tolerance=0.001 if kind == "directed_unit_vector" else None,
            classification_values=None,
            reference_and_sample_rule="UNFROZEN",
            applicability_rule="UNFROZEN",
        )
    return dict(
        schema="CONTRACT_E16_DICTIONARY_DRAFT_297_v1", fields=fields,
        formal_customer_approval=False,
        board_height_reference="实际接触板及局部底板基准",
        board_height_reference_confirmed=True,
        plate_side_and_normal_sign_convention="UNFROZEN",
        critical_groups_and_minimum_counts=None,
        legacy_encodings=dict(
            state_mapping_frozen=False,
            value_51_when_documented_as_over_50=dict(
                status="out_of_range", censor=dict(threshold_mm=50, relation="gt", units="mm")),
            value_minus_1="UNFROZEN_FIELD_SPECIFIC_MAPPING",
            zero="NUMERIC_ZERO_NEVER_AN_UNKNOWN_PLACEHOLDER",
        ),
        geometry_matcher_external_and_separately_frozen=True,
        frozen_training_295_modified=False,
    )


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _normal(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_number(x) for x in value):
        return False
    return abs(math.hypot(*value) - 1) <= 0.001


def _censor(value):
    return (isinstance(value, dict) and set(value) == {"threshold_mm", "relation", "units"}
            and _number(value["threshold_mm"]) and value["threshold_mm"] >= 0
            and value["relation"] in ("gt", "ge", "lt", "le") and value["units"] == "mm")


def _valid_truth_sample(field, sample, spec):
    if not isinstance(sample, dict) or not isinstance(sample.get("reference_key"), str) or not sample["reference_key"]:
        return False
    state = sample.get("status")
    if state not in spec["allowed_states"] or state in ("unknown", "not_applicable"):
        return False
    if state == "out_of_range":
        return _censor(sample.get("censor")) and sample.get("value") is None
    if state == "no_object":
        return sample.get("value") is None
    if state != "value":
        return False
    value, kind = sample.get("value"), spec["kind"]
    if kind == "distance_mm":
        return _number(value) and value >= 0 and (not field.startswith("thickness_") or value > 0)
    if kind == "directed_unit_vector":
        return _normal(value)
    if kind == "boolean":
        return type(value) is bool
    values = spec.get("classification_values")
    if not isinstance(values, list) or not values or not all(type(x) is str for x in values):
        return False
    if kind == "category":
        return type(value) is str and value in values
    if kind == "multilabel":
        return isinstance(value, list) and all(type(x) is str and x in values for x in value) and len(set(value)) == len(value)
    return False


def sample_pass(field, prediction, truth, spec):
    if not spec.get("semantics_frozen") or not _valid_truth_sample(field, truth, spec) or not isinstance(prediction, dict):
        return False
    if not isinstance(prediction.get("source"), str) or not prediction["source"]:
        return False
    if not _number(prediction.get("confidence")) or not 0 <= prediction["confidence"] <= 1:
        return False
    if prediction.get("reference_key") != truth["reference_key"] or prediction.get("status") != truth["status"]:
        return False
    state = truth["status"]
    if state == "out_of_range":
        return (prediction.get("value") is None and _censor(prediction.get("censor"))
                and prediction["censor"] == truth["censor"])
    if state == "no_object":
        return prediction.get("value") is None
    if not _valid_truth_sample(field, prediction, spec):
        return False
    p, q, kind = prediction["value"], truth["value"], spec["kind"]
    if kind == "directed_unit_vector":
        cross = (p[1]*q[2]-p[2]*q[1], p[2]*q[0]-p[0]*q[2], p[0]*q[1]-p[1]*q[0])
        angle = math.degrees(math.atan2(math.hypot(*cross), sum(a*b for a, b in zip(p, q))))
        return angle <= 3
    if kind == "distance_mm":
        return abs(p-q) <= (1 if field.startswith("thickness_") else 3)
    if kind == "multilabel":
        return set(p) == set(q)
    return p == q


def strictly_over_95(passed, denominator):
    return denominator > 0 and passed*100 > denominator*95


def _indexed_samples(samples):
    if not isinstance(samples, list) or not samples:
        return None
    result = {}
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("key"), str) or not sample["key"] or sample["key"] in result:
            return None
        result[sample["key"]] = sample
    return result


def _field_result(field, truth, prediction, spec, geometry_passed):
    """Return denominator inclusion, reviewed truth readiness, pass, NA correctness."""
    truth = truth if isinstance(truth, dict) else {}
    app = truth.get("applicability", {})
    app_ready = (truth.get("reviewed") is True and isinstance(app, dict)
                 and type(app.get("value")) is bool and app.get("rule_frozen") is True
                 and isinstance(app.get("rule_id"), str) and bool(app["rule_id"])
                 and app["rule_id"] == spec.get("applicability_rule")
                 and spec.get("semantics_frozen") is True)
    if app_ready and app["value"] is False:
        correct_na = (isinstance(prediction, dict) and prediction.get("status") == "not_applicable"
                      and prediction.get("rule_id") == app["rule_id"] and prediction.get("value") is None
                      and isinstance(prediction.get("source"), str) and bool(prediction["source"])
                      and _number(prediction.get("confidence")) and 0 <= prediction["confidence"] <= 1)
        return False, True, False, correct_na
    expected = _indexed_samples(truth.get("samples"))
    ready = (app_ready and app["value"] is True and truth.get("sample_keys_frozen") is True
             and expected is not None and all(_valid_truth_sample(field, s, spec) for s in expected.values()))
    if not ready:
        return True, False, False, False
    actual = _indexed_samples(prediction.get("samples")) if isinstance(prediction, dict) else None
    correct = (geometry_passed is True and actual is not None and set(actual) == set(expected)
               and all(sample_pass(field, actual[k], expected[k], spec) for k in expected))
    return True, True, correct, False


def evaluate_fields(inventory, matched_predictions, dictionary, *, inventory_verified=False, geometry_protocol_frozen=False):
    """Inventory comes from authoritative source IDs, never only detected rows.

    A missing matched prediction remains in every applicable denominator. Unfrozen
    applicability is conservatively included and cannot produce an acceptance.
    This component consumes geometry matching; it cannot itself certify D1-D4.
    """
    specs = dictionary.get("fields", {})
    if set(specs) != set(FIELDS):
        raise ValueError("Exactly all 16 contract fields required")
    for field, spec in specs.items():
        expected_kind = ("directed_unit_vector" if field.startswith("normal_") else
                         "distance_mm" if field in FIELDS[:12] else None)
        if expected_kind and spec.get("kind") != expected_kind:
            raise ValueError("Contract numeric and signed normal field types cannot change")
    inventory = list(inventory)
    if not inventory:
        raise ValueError("Complete nonempty original inventory required")
    def identity(row):
        key = (row.get("workpiece_id"), row.get("truth_child_id"))
        if not all(isinstance(x, str) and x for x in key):
            raise ValueError("Original workpiece and child IDs required")
        return key
    originals = {identity(r): r for r in inventory}
    if len(originals) != len(inventory):
        raise ValueError("Duplicate original IDs")
    predictions = {}
    for row in matched_predictions:
        key = identity(row)
        if key not in originals or key in predictions:
            raise ValueError("Unknown or duplicate matched truth ID")
        predictions[key] = row
    groups = {"overall": list(originals)}
    for key, row in originals.items():
        for group in set(row.get("groups", [])):
            if group != "overall":
                groups.setdefault(group, []).append(key)
    ledger, tables = [], {}
    for key, row in originals.items():
        p = predictions.get(key, {})
        outcomes = {}
        for field in FIELDS:
            included, ready, passed, na_correct = _field_result(
                field, row.get("fields", {}).get(field), p.get("fields", {}).get(field), specs[field], p.get("geometry_passed"))
            outcomes[field] = dict(included_in_conservative_denominator=included,
                                   reviewed_truth_ready=ready, field_passed=bool(passed),
                                   frozen_not_applicable_correct=bool(na_correct))
        ledger.append(dict(workpiece_id=key[0], truth_child_id=key[1], fields=outcomes))
    outcome_map = {(r["workpiece_id"], r["truth_child_id"]): r["fields"] for r in ledger}
    for group, keys in groups.items():
        table = {}
        for field in FIELDS:
            values = [outcome_map[key][field] for key in keys]
            denominator = sum(v["included_in_conservative_denominator"] for v in values)
            passed = sum(v["field_passed"] for v in values)
            ready = all(v["reviewed_truth_ready"] for v in values)
            na_errors = sum(not v["included_in_conservative_denominator"] and not v["frozen_not_applicable_correct"] for v in values)
            table[field] = dict(
                original_units=len(keys), denominator=denominator, passed=passed,
                failed_or_unready=denominator-passed,
                reviewed_truth_ready_units=sum(v["reviewed_truth_ready"] for v in values),
                frozen_not_applicable_units=len(keys)-denominator,
                applicability_errors=na_errors,
                mechanical_rate=passed/denominator if denominator else None,
                rate_is_model_accuracy=ready and inventory_verified and geometry_protocol_frozen,
                strict_over_95=strictly_over_95(passed, denominator),
                field_acceptance_pass=(ready and na_errors == 0 and strictly_over_95(passed, denominator)
                                       and inventory_verified and geometry_protocol_frozen),
            )
        tables[group] = table
    minima = dictionary.get("critical_groups_and_minimum_counts")
    group_scope_ready = (isinstance(minima, dict) and bool(minima)
                         and all(type(n) is int and n > 0 and len(groups.get(g, [])) >= n for g, n in minima.items()))
    formal_ready = (dictionary.get("formal_customer_approval") is True and group_scope_ready
                    and set(groups) <= set(minima) and inventory_verified and geometry_protocol_frozen
                    and all(t["reviewed_truth_ready_units"] == t["original_units"]
                            for table in tables.values() for t in table.values()))
    return dict(
        schema="CONTRACT_E16_FULL_INVENTORY_COMPONENT_297_v1", original_units=len(originals),
        field_count=16, all_field_cells=len(originals)*16, groups=tables, unit_ledger=ledger,
        formal_E_scope_ready=formal_ready,
        formal_E_acceptance_pass=(formal_ready and all(
            t["field_acceptance_pass"] for table in tables.values() for t in table.values())),
        full_contract_pass=False, no_side_exchange_or_sign_flip=True,
        unknown_truth_and_geometry_misses_retain_denominator=True,
    )
