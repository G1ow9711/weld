"""Compose full-label teacher conditions, distinct from a frozen free prediction.

The future338 capacity domain includes every background carrier in the census.
NN conditions apply only to carriers assigned original truth parents, searching
their entire legal terminal domain. No conditional success is a free-model TP.
"""
from collections import Counter

import torch

from .carrier_observation_sequences_328 import SequenceLayout
from .carrier_subslot_conflict_audit_341 import complete_role_pair_census, classify_constant_delta_role_conflicts
from .conditional_terminal_rank_audit_343 import audit_true_start_conditioned_terminals
from .conditional_parent_geometry_audit_344 import audit_true_parent_conditioned_geometry
from .preallocated_carrier_inference_340 import infer_preallocated_carrier_heads


def audit_complete_supervised_source_conditions(model, carriers, supervision, tokens, provider,
                                                actual_role_deltas, sink, progress=None):
    if torch.is_grad_enabled() or model.training:
        raise ValueError("Frozen eval-mode no-grad conditional diagnosis required")
    if not callable(sink):
        raise ValueError("Detailed original-parent records require a streaming sink")
    report = progress or (lambda **values: None)
    if set(supervision.parents_by_carrier) != set(supervision.roles_by_carrier):
        raise ValueError("A positive carrier or original parent disappeared from supervision")
    census = complete_role_pair_census(carriers, supervision.roles_by_carrier, supervision.positive_role_counts)
    if census["complete_legal_slots"] != supervision.complete_slots:
        raise ValueError("Complete future domain denominator changed")
    conflicts = classify_constant_delta_role_conflicts(census, actual_role_deltas)
    sink(dict(record="COMPLETE_FUTURE_DOMAIN_ROLE_PAIR_CENSUS", census=census, actual_real_weight_conflicts=conflicts))
    modes = ("TRUE_SLOTS_GT_PARAMETERS_PREDICTED_ROOT_RESIDUAL", "TRUE_SLOTS_PREDICTED_PARAMETERS_PREDICTED_ROOT_RESIDUAL")
    counts = Counter()
    categories = {}
    seen_parents = set()
    parent_count = 0
    sequence_calls = 0
    truth_carriers = 0
    for ordinal in sorted(supervision.parents_by_carrier):
        carrier = carriers[ordinal]
        if carrier.ordinal != ordinal:
            raise ValueError("Original complete carrier order changed")
        layout = SequenceLayout.from_carrier(carrier)
        report(stage="CONDITIONAL_FULL_TERMINAL_DOMAIN_BEGIN", carrier_ordinal=ordinal,
            all_carrier_rows=layout.rows, all_carrier_legal_slots=carrier.slot_count)
        heads, calls = infer_preallocated_carrier_heads(model, tokens, provider, layout)
        sequence_calls += calls
        truth_carriers += 1
        key = carrier.weld_type + ":" + carrier.kind
        category_counts = categories.setdefault(key, Counter())
        for parent in supervision.parents_by_carrier[ordinal]:
            pi = parent["original_parent_index"]
            if pi in seen_parents:
                raise ValueError("Original parent repeated in conditioned denominator")
            seen_parents.add(pi)
            truth_parent = supervision.targets["parents"][pi]
            slot_q = {int(node["slot_index"]): float(node["q_mm"]) for node in parent["nodes"].values()}
            enriched = [dict(child, true_start_q_mm=slot_q[int(child["start_slot"])],
                true_end_q_mm=slot_q[int(child["end_slot"])]) for child in parent["children"]]
            ranks = audit_true_start_conditioned_terminals(model, carrier, layout, heads, enriched)
            geometry = audit_true_parent_conditioned_geometry(carrier, layout, heads, parent, truth_parent)
            expected = len(truth_parent["children"])
            if ranks["original_children"] != expected or geometry["original_children"] != expected:
                raise ValueError("A truth child disappeared between rank and geometry conditions")
            sink(dict(record="ORIGINAL_PARENT_COMPLETE_CONDITIONAL_RESULTS", carrier_ordinal=ordinal,
                exact_geometry_identity=carrier.geometry_identity, original_parent_index_metadata_only=pi,
                category=key, original_children=expected, terminal_ranks=ranks, geometry_conditions=geometry,
                oracle_conditions_not_free_prediction_or_contract_TP=True))
            for tally in (counts, category_counts):
                tally["original_children"] += expected
                tally["original_parents"] += 1
                for child in ranks["child_results"]:
                    tally["true_start_selected_at_original_threshold"] += int(child["start_selected_at_original_threshold"])
                    tally["true_start_type_matches"] += int(child["predicted_type_matches_carrier"])
                    tally["true_start_selected_and_type_matches"] += int(child["start_selected_at_original_threshold"] and child["predicted_type_matches_carrier"])
                    tally["truth_terminal_illegal_retained"] += int(not child["truth_end_legal"])
                    tally["truth_terminal_exact_top1_given_true_start"] += int(child["selected_end_is_exact_truth_slot"])
                    tally["truth_terminal_exact_top5_given_true_start"] += int(child["truth_end_rank_with_original_tie_order"] is not None and child["truth_end_rank_with_original_tie_order"] <= 5)
                for mode in modes:
                    for phase in ("before_projection", "after_projection"):
                        result = geometry["modes"][mode][phase]
                        if result["original_children"] != expected or len(result["child_results"]) != expected:
                            raise ValueError("Invalid geometry was removed from its original denominator")
                        tally[mode + ":" + phase + ":conditional_certified_children"] += result["conditional_certified_children"]
            parent_count += 1
        del heads, layout
        report(stage="CONDITIONAL_COMPLETE_TRUTH_CARRIER_FINISHED", carrier_ordinal=ordinal,
            complete_truth_carriers=truth_carriers, all_original_children_processed=counts["original_children"],
            actual_sequence_forwards=sequence_calls)
    if seen_parents != set(range(len(supervision.targets["parents"]))):
        raise ValueError("Every original truth parent is required")
    if counts["original_children"] != supervision.original_children or counts["original_children"] != supervision.targets["original_children"]:
        raise ValueError("The original complete G denominator changed")
    for key, expected in supervision.children_by_category.items():
        if categories.get(key, {}).get("original_children", 0) != expected:
            raise ValueError("An original weld-category denominator changed")
    return dict(version="COMPLETE_G_FUTURE338_ROLE_TYPE_TERMINAL_GEOMETRY_CONDITIONS_351_v1",
        original_G=supervision.original_children, original_parent_count=parent_count,
        complete_future_carriers=census["complete_carriers"], complete_future_legal_slots=supervision.complete_slots,
        complete_background_census=census, actual_real_weight_pair_conflicts=conflicts,
        conditioned_true_carriers=truth_carriers, actual_sequence_forwards=sequence_calls,
        counts=dict(counts), categories={key: dict(value) for key, value in categories.items()},
        every_original_parent_child_and_category_denominator_retained=True,
        every_true_start_searches_complete_original_legal_terminal_domain=True,
        geometry_conditions_preserve_invalid_children_before_and_after_original_projection=True,
        detailed_original_parent_records_streamed_without_retaining_head_tensors=True,
        future338_conditioned_domain_distinct_from_current600_free329_domain=True,
        NN_background_carriers_not_claimed_tested_by_this_condition=True,
        free_prediction_before_chain_links_veto_and_conflict_diagnostics_still_required=True,
        no_conditional_geometry_counts_added_to_free_TP=True,
        all_E16_accuracy="NOT_MEASURED", formal_acceptance_measured=False,
        actual_optimizer_updates=0, full_contract_pass=False)
