"""Offline full-domain role-pair census; no neural predictions or acceptance.

Counts use the complete finite domain, including every background-only carrier.
Only the two regular slots in a cell form a pair. Finite endpoints and observed
structural-event slots stay explicit and never enter regular 00/01/10/11 counts.
"""
from collections import Counter, defaultdict

import numpy as np

from .carrier_boundary_model_328 import ROLE_FIELDS


def _empty_role():
    return Counter({"00": 0, "01": 0, "10": 0, "11": 0,
                    "positive_paired_slots": 0, "positive_unpaired_slots": 0,
                    "negative_unpaired_slots": 0, "positive_all_slots": 0,
                    "negative_all_slots": 0})


def complete_role_pair_census(carriers, roles_by_carrier, expected_positive_counts=None):
    """01 means first subslot negative, second positive; never reverse it."""
    if set(roles_by_carrier) - set(range(len(carriers))):
        raise ValueError("Supervision points outside the complete carrier domain")
    by_category = {}
    for ordinal, carrier in enumerate(carriers):
        if carrier.ordinal != ordinal:
            raise ValueError("Carrier ordinal/order changed")
        key = carrier.weld_type + ":" + carrier.kind
        if key not in by_category:
            by_category[key] = dict(carriers=0, regular_cells=0, legal_slots=0,
                                    unpaired_slots=0, roles={r: _empty_role() for r in ROLE_FIELDS})
        stats = by_category[key]
        cells = carrier.cell_count
        regular = 2 * cells
        extra = carrier.slot_count - regular
        if extra != 2 + len(carrier.structural_events_mm):
            raise ValueError("The complete finite/event domain changed")
        stats["carriers"] += 1
        stats["regular_cells"] += cells
        stats["legal_slots"] += carrier.slot_count
        stats["unpaired_slots"] += extra
        sets = roles_by_carrier.get(ordinal, [set() for _ in ROLE_FIELDS])
        if len(sets) != len(ROLE_FIELDS):
            raise ValueError("Every role is required")
        for role, positives in zip(ROLE_FIELDS, sets):
            if not isinstance(positives, (set, frozenset)):
                raise ValueError("Positive supervision must contain unique declared slots")
            pair_bits = {}
            unpaired_positive = 0
            for slot in positives:
                if isinstance(slot, bool) or not isinstance(slot, (int, np.integer)) or not 0 <= slot < carrier.slot_count:
                    raise ValueError("Positive role outside a declared legal slot")
                if slot < regular:
                    cell, subslot = divmod(int(slot), 2)
                    pair_bits[cell] = pair_bits.get(cell, 0) | (1 << subslot)
                else:
                    unpaired_positive += 1
            counts = stats["roles"][role]
            counts["00"] += cells - len(pair_bits)
            for bits in pair_bits.values():
                counts[{1: "10", 2: "01", 3: "11"}[bits]] += 1
            counts["positive_paired_slots"] += len(positives) - unpaired_positive
            counts["positive_unpaired_slots"] += unpaired_positive
            counts["negative_unpaired_slots"] += extra - unpaired_positive
            counts["positive_all_slots"] += len(positives)
            counts["negative_all_slots"] += carrier.slot_count - len(positives)
    for key, stats in by_category.items():
        for role_index, role in enumerate(ROLE_FIELDS):
            counts = stats["roles"][role]
            if sum(counts[pattern] for pattern in ("00", "01", "10", "11")) != stats["regular_cells"]:
                raise ValueError("A regular cell disappeared from a pair denominator")
            if counts["01"] + counts["10"] + 2 * counts["11"] != counts["positive_paired_slots"]:
                raise ValueError("Paired positive mass changed")
            if counts["positive_unpaired_slots"] + counts["negative_unpaired_slots"] != stats["unpaired_slots"]:
                raise ValueError("An endpoint/event slot disappeared")
            if counts["positive_all_slots"] + counts["negative_all_slots"] != stats["legal_slots"]:
                raise ValueError("The complete all-slot denominator changed")
            if expected_positive_counts is not None:
                expected = expected_positive_counts.get(key, np.zeros(len(ROLE_FIELDS), np.int64))
                if counts["positive_all_slots"] != int(expected[role_index]):
                    raise ValueError("Census and canonical complete supervision disagree")
    return dict(categories=by_category,
                complete_carriers=len(carriers), complete_legal_slots=sum(v["legal_slots"] for v in by_category.values()),
                includes_all_background_only_carriers=True,
                pattern_01_means_only_second_subslot_positive=True,
                endpoint_and_structural_event_slots_are_not_paired=True,
                census_is_not_a_free_prediction_or_contract_score=True)


def classify_constant_delta_role_conflicts(census, deltas):
    """Exact real-arithmetic limitation, separate from floating-point inference.

For one common threshold and l1=l0+delta: delta<0 forbids 01; delta>0 forbids
10; zero forbids both. A matching sign only removes this particular obstacle;
it does not prove that the model can fit all rows or generalize.
"""
    if set(deltas) != set(ROLE_FIELDS):
        raise ValueError("Actual constant deltas for every role are required")
    result = {}
    for category, stats in census["categories"].items():
        rows = {}
        for role in ROLE_FIELDS:
            delta = float(deltas[role])
            if not np.isfinite(delta):
                raise ValueError("Finite actual-weight algebra required")
            forbidden = ("01",) if delta < 0 else (("10",) if delta > 0 else ("01", "10"))
            patterns = stats["roles"][role]
            rows[role] = dict(subslot1_minus_subslot0_logit=delta,
                impossible_patterns_in_real_arithmetic=list(forbidden),
                conflicting_regular_cells=sum(patterns[p] for p in forbidden),
                conflicting_pattern_counts={p: patterns[p] for p in forbidden},
                all_regular_cells=stats["regular_cells"], all_legal_slots=stats["legal_slots"])
        result[category] = rows
    return dict(categories=result, fixed_common_role_threshold_required=True,
                exact_real_arithmetic_algebra_not_floating_point_equality_claim=True,
                conflict_count_is_not_G_missed_weld_count=True,
                excludes_unpaired_endpoint_event_slots_from_pair_conflicts=True,
                not_a_claim_that_constant_delta_explains_every_error=True)
