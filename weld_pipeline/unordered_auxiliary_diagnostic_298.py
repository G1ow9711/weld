"""Post-round development diagnostic at the model's own predicted welds.

Uses the already frozen 295 point masks and signed CAD candidates. This is
neither an ordered E score nor a change to the registered geometry scorer.
All eligible auxiliary children, including geometry misses, remain in G.
"""
import math


def _number(x):
    return type(x) in (int, float) and math.isfinite(x)


def _normal(x):
    return isinstance(x, (list, tuple)) and len(x) == 3 and all(_number(v) for v in x) and abs(math.hypot(*x)-1) <= 0.001


def _angle(p, q):
    cross = (p[1]*q[2]-p[2]*q[1], p[2]*q[0]-p[0]*q[2], p[0]*q[1]-p[1]*q[0])
    return math.degrees(math.atan2(math.hypot(*cross), sum(a*b for a, b in zip(p, q))))


def compare_child(estimate, truth, *, whole_open_reversed=False):
    expected = truth["samples"]
    valid = [s for s in expected if s["training_point_mask"]]
    if not valid:
        raise ValueError("Diagnostic requires at least one already frozen valid point")
    if not isinstance(estimate, dict) or estimate.get("fractions") != [0, .25, .5, .75, 1]:
        return dict(passed=False, reason="MISSING_OR_WRONG_FIXED_PREDICTION_SAMPLE_KEYS")
    thickness, normals = estimate.get("thickness_mm"), estimate.get("CAD_outward_unit_normals_world")
    if not isinstance(thickness, list) or not isinstance(normals, list) or len(thickness) != 5 or len(normals) != 5:
        return dict(passed=False, reason="MISSING_FIVE_POINT_UNORDERED_ESTIMATE")
    alternatives = []
    for permutation in ((0, 1), (1, 0)):
        dt, dn, cost = [], [], 0.0
        for sample in valid:
            if sample["fraction"] not in estimate["fractions"]:
                raise ValueError("Unexpected frozen reference fraction")
            position = estimate["fractions"].index(1-sample["fraction"] if whole_open_reversed else sample["fraction"])
            pt, pn = thickness[position], normals[position]
            pair = sample["unordered_material_pair"]
            if not all(isinstance(v, (list, tuple)) and len(v) == 2 for v in (pt, pn, pair)):
                return dict(passed=False, reason="INVALID_MATERIAL_PAIR_SHAPE")
            for pside, tside in enumerate(permutation):
                t = pair[tside]["thickness_mm"]; n = pair[tside]["CAD_outward_unit_normal_world"]
                if not (_number(pt[pside]) and pt[pside] > 0 and _number(t) and t > 0 and _normal(pn[pside]) and _normal(n)):
                    return dict(passed=False, reason="INVALID_THICKNESS_OR_SIGNED_UNIT_NORMAL")
                delta = abs(pt[pside]-t); angle = _angle(pn[pside], n)
                dt.append(delta); dn.append(angle)
                cost += (0.5*delta*delta if delta < 1 else delta-0.5) + (1-math.cos(math.radians(angle)))/(1-math.cos(math.radians(3)))
        alternatives.append(dict(permutation=list(permutation), fixed_pair_loss=cost/(2*len(valid)),
                                 thickness_errors_mm=dt, signed_normal_errors_deg=dn,
                                 passed=all(t <= 1 and n <= 3 for t, n in zip(dt, dn))))
    selected = min(alternatives, key=lambda r: (r["fixed_pair_loss"], r["permutation"]))
    return dict(selected, reason="PASS_ALL_FROZEN_MASKED_POINTS" if selected["passed"] else "MASKED_POINT_ERROR_EXCEEDS_1MM_OR_3DEG",
                point_count=len(valid), whole_child_material_pair_shared_between_all_fields_and_points=True,
                whole_open_reversal_fraction_mapping_used=whole_open_reversed,
                normal_abs_dot_or_sign_flip_used=False)


def score_source(decoded, geometry_score, auxiliary):
    published = [child for parent in decoded["parents"] for child in parent["segments"]]
    by_id = {c["original_child_id"]: c for c in auxiliary["children"]}
    if len(by_id) != auxiliary["original_children"]:
        raise ValueError("Full original auxiliary inventory required")
    matches, used_predictions = {}, set()
    for match in geometry_score["matches"]:
        key, p = match["original_child_id"], match["prediction_row"]
        if key not in by_id or key in matches or p in used_predictions or not 0 <= p < len(published):
            raise ValueError("Geometry matches must be valid and one-to-one")
        matches[key] = match; used_predictions.add(p)
    rows = []
    for key, truth in by_id.items():
        points = sum(bool(s["training_point_mask"]) for s in truth["samples"])
        if not points:
            continue
        match = matches.get(key)
        if match is None:
            outcome = dict(passed=False, reason="GEOMETRY_MISS_RETAINED_IN_AUXILIARY_G")
        else:
            child = published[match["prediction_row"]]
            outcome = compare_child(child.get("unordered_material_estimate"), truth,
                                    whole_open_reversed=bool(match["whole_open_parent_reversed"]))
        rows.append(dict(original_child_id=key, valid_frozen_points=points, geometry_matched=match is not None, **outcome))
    g, passed = len(rows), sum(r["passed"] for r in rows)
    return dict(schema="POST_ROUND_OWN_PREDICTION_UNORDERED_CAD_AUXILIARY_DIAGNOSTIC_298_v1",
                all_original_children=len(by_id), eligible_frozen_point_children=g,
                valid_frozen_points=sum(r["valid_frozen_points"] for r in rows),
                geometry_matched_eligible_children=sum(r["geometry_matched"] for r in rows),
                all_masked_material_points_correct_children=passed,
                failed_or_missing_eligible_children=g-passed,
                development_only_rate=passed/g if g else None, children=rows,
                all_eligible_geometry_misses_retained=True,
                board_sides_unordered_candidate_diagnostic_only=True,
                formal_E_accuracy_measured=False, changes_registered_geometry_scorer=False,
                actual_optimizer_updates=0, full_contract_pass=False)
