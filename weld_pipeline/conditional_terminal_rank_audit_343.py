"""Offline true-start conditions, using the unchanged full legal terminal domain.

Every supplied child remains in the result, including illegal truth addresses.
Exact slot rank and probabilities diagnose pointer competition; they are not
free-prediction TP, millimetre geometry success, or contract acceptance.
"""
import math

import numpy as np
import torch

from .carrier_boundary_runtime_329 import POINTER_END_ROLE_WEIGHT, PRESENCE_THRESHOLD
from .carrier_pointer_geometry_329 import legal_terminal_mask, learned_parameters


def _active_index(layout, slot):
    row, subslot = layout.location(slot)
    if not layout.active_slots[row, subslot] or layout.slot_ids[row, subslot] != slot:
        raise ValueError("Truth address is not a declared active slot")
    return int(np.count_nonzero(layout.active_slots[:row])) + int(np.count_nonzero(layout.active_slots[row, :subslot]))


def audit_true_start_conditioned_terminals(model, carrier, layout, heads, children):
    if torch.is_grad_enabled():
        raise ValueError("Terminal-rank diagnosis requires no-grad inference")
    if layout.carrier.geometry_identity != carrier.geometry_identity:
        raise ValueError("The complete layout belongs to a different carrier")
    required = {"embeddings", "role_logits", "weld_type_logits", "terminal_keys", "within_cell_fraction"}
    if not required.issubset(heads):
        raise ValueError("Complete teacher-conditioned heads are required")
    for key in required:
        value = heads[key]
        if value.shape[:2] != (layout.rows, 2) or value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError("Finite complete original CPU heads are required")
    active = torch.from_numpy(layout.active_slots)
    ids = layout.slot_ids[layout.active_slots]
    flat = {key: heads[key][active] for key in required if key != "within_cell_fraction"}
    probability = flat["role_logits"].sigmoid()
    parameters = learned_parameters(heads, layout)[active]
    results = []
    for original_order, child in enumerate(children):
        start_slot, end_slot = child["start_slot"], child["end_slot"]
        start, true_end = _active_index(layout, start_slot), _active_index(layout, end_slot)
        legal = torch.from_numpy(legal_terminal_mask(carrier, layout, start_slot)[layout.active_slots])
        logits = (model.pointer_logits(flat["embeddings"][start], flat["terminal_keys"])
                  + POINTER_END_ROLE_WEIGHT * flat["role_logits"][:, 2])
        if logits.shape != legal.shape or not bool(torch.isfinite(logits[legal]).all()):
            raise ValueError("A complete finite legal-terminal score row is required")
        truth_legal = bool(legal[true_end])
        legal_count = int(legal.sum())
        selected_end = int(logits.masked_fill(~legal, -torch.inf).argmax()) if legal_count else None
        rank = log_probability = true_probability = ties = margin = None
        if truth_legal:
            score = logits[true_end]
            positions = torch.arange(len(ids))
            # The decoder's argmax resolves equal scores by the first active
            # index. Preserve that ordering for rank; do not average ties.
            rank = 1 + int((legal & ((logits > score) | ((logits == score) & (positions < true_end)))).sum())
            ties = int((legal & (logits == score)).sum())
            log_probability = float(score - torch.logsumexp(logits[legal], dim=0))
            true_probability = math.exp(log_probability)
            competitors = legal.clone(); competitors[true_end] = False
            margin = float(score - logits[competitors].max()) if bool(competitors.any()) else None
        predicted_type = int(flat["weld_type_logits"][start].argmax())
        row = dict(original_child_order_metadata_only=original_order,
            start_slot=int(start_slot), true_end_slot=int(end_slot),
            start_role_probability=float(probability[start, 1]),
            true_end_role_probability=float(probability[true_end, 2]),
            start_selected_at_original_threshold=bool(probability[start, 1] >= PRESENCE_THRESHOLD),
            predicted_weld_type_index=predicted_type,
            predicted_type_matches_carrier=predicted_type == int(carrier.weld_type == "vertical"),
            full_active_slots=len(ids), full_legal_terminal_slots=legal_count,
            truth_end_legal=truth_legal, truth_end_rank_with_original_tie_order=rank,
            truth_end_tied_score_count=ties, truth_end_log_probability=log_probability,
            truth_end_probability=true_probability, truth_end_margin_over_best_other_legal_terminal=margin,
            selected_end_slot=int(ids[selected_end]) if selected_end is not None else None,
            selected_end_is_exact_truth_slot=selected_end == true_end if selected_end is not None else False,
            learned_true_start_parameter_mm=float(parameters[start]),
            learned_true_end_parameter_mm=float(parameters[true_end]),
            learned_selected_end_parameter_mm=float(parameters[selected_end]) if selected_end is not None else None,
            parent_start_role_probability=float(probability[start, 3]),
            parent_end_at_truth_role_probability=float(probability[true_end, 4]),
            parent_end_at_selected_role_probability=float(probability[selected_end, 4]) if selected_end is not None else None)
        if "true_start_q_mm" in child:
            row["true_start_within_slot_parameter_error_mm"] = abs(float(parameters[start]) - float(child["true_start_q_mm"]))
        if "true_end_q_mm" in child:
            row["true_end_within_slot_parameter_error_mm"] = abs(float(parameters[true_end]) - float(child["true_end_q_mm"]))
            row["selected_end_source_parameter_error_mm"] = abs(float(parameters[selected_end]) - float(child["true_end_q_mm"])) if selected_end is not None else None
        results.append(row)
    if len(results) != len(children):
        raise ValueError("A child disappeared from the conditioned denominator")
    return dict(original_children=len(children), child_results=results,
        all_supplied_children_retained=True, full_original_legal_terminal_domain_used=True,
        illegal_truth_addresses_retained_with_null_rank=True,
        threshold_and_pointer_end_role_weight_unchanged=True,
        exact_slot_rank_is_not_3mm_curve_or_free_prediction_acceptance=True,
        parent_boundary_and_within_cell_values_are_causal_diagnostics=True,
        no_geometry_TP_from_this_conditioned_rank=True, full_contract_pass=False)
