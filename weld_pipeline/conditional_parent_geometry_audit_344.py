"""Offline causal geometry: true addresses/topology with predicted residuals.

Compare GT parameters and predicted within-slot parameters, each before/after
the unchanged process projection. All children stay in every denominator.
Oracle topology/address conditions cannot count as free or formal acceptance.
"""
import numpy as np
import torch

from .analytic_carrier_slots_327 import analytic_whole_curve_certificate
from .bounded_analytic_residuals_327 import assemble_corrected_parent
from .carrier_pointer_geometry_329 import learned_parameters
from .selected_slot_process_projection_329 import project_selected_parameters


def _geometry_result(carrier, nodes, refs, gaps, residual, closed, truth_children):
    try:
        assembled = assemble_corrected_parent(carrier, nodes, refs, gaps, residual, closed)
    except ValueError as error:
        return dict(parent_assembled=False, parent_error=str(error),
            original_children=len(truth_children), conditional_certified_children=0,
            child_results=[dict(original_child_order_metadata_only=i,
                conditional_3mm_geometry_certificate_passed=False, parent_error=str(error))
                for i in range(len(truth_children))])
    if len(assembled["segments"]) != len(truth_children):
        raise ValueError("Assembled child order/count changed")
    results = []
    for i, (segment, truth) in enumerate(zip(assembled["segments"], truth_children)):
        try:
            certificate = analytic_whole_curve_certificate(segment, truth["world_representation"])
            category_matches = (segment["weld_type"] == truth["weld_type"]
                and segment["kind"] == truth["original_kind"])
            passed = (category_matches and certificate["ordered_endpoint_maximum_mm"] <= 3.
                and certificate["continuous_bidirectional_upper_bound_mm"] <= 3.)
            results.append(dict(original_child_order_metadata_only=i, category_matches=category_matches,
                conditional_3mm_geometry_certificate_passed=bool(passed), continuous_certificate=certificate))
        except ValueError as error:
            results.append(dict(original_child_order_metadata_only=i,
                conditional_3mm_geometry_certificate_passed=False, certificate_unavailable_error=str(error)))
    return dict(parent_assembled=True, original_children=len(truth_children),
        conditional_certified_children=sum(r["conditional_3mm_geometry_certificate_passed"] for r in results),
        child_results=results, common_analytic_displacement_bound_mm=assembled["analytic_residual_continuous_displacement_bound_mm"])


def audit_true_parent_conditioned_geometry(carrier, layout, heads, supervised_parent, truth_parent):
    if torch.is_grad_enabled():
        raise ValueError("Conditioned geometry diagnosis requires no-grad inference")
    for key in ("within_cell_fraction", "residual"):
        if key not in heads or heads[key].shape[:2] != (layout.rows, 2) or heads[key].device.type != "cpu":
            raise ValueError("Complete original CPU parameter/residual heads are required")
        if not bool(torch.isfinite(heads[key]).all()):
            raise ValueError("Finite full parameter/residual heads are required")
    children = supervised_parent["children"]
    truths = truth_parent["children"]
    if not children or len(children) != len(truths) or supervised_parent["gaps"] != truth_parent["gaps"]:
        raise ValueError("Every ordered child and gap must remain in the conditioned diagnosis")
    if bool(supervised_parent["is_closed"]) != bool(truth_parent["is_closed"]):
        raise ValueError("The oracle closure condition changed")
    nodes = supervised_parent["nodes"]
    slot_to_identity = {int(node["slot_index"]): identity for identity, node in nodes.items()}
    if len(slot_to_identity) != len(nodes):
        raise ValueError("Independent authoritative nodes repeat a slot")
    refs = [dict(start_node=slot_to_identity[int(c["start_slot"])],
                 end_node=slot_to_identity[int(c["end_slot"])]) for c in children]
    root_location = layout.location(int(children[0]["start_slot"]))
    residual = heads["residual"][root_location].numpy().astype(np.float64)
    parameters = learned_parameters(heads, layout)
    modes = {}
    for mode in ("TRUE_SLOTS_GT_PARAMETERS_PREDICTED_ROOT_RESIDUAL", "TRUE_SLOTS_PREDICTED_PARAMETERS_PREDICTED_ROOT_RESIDUAL"):
        delivered_nodes = []
        for identity, node in nodes.items():
            slot = int(node["slot_index"])
            spec = carrier.slot_spec(slot)
            true_q = float(node["q_mm"])
            if not spec["lower_mm"] <= true_q <= spec["upper_mm"]:
                raise ValueError("A true node lies outside its fixed legal address")
            q = true_q if mode.startswith("TRUE_SLOTS_GT_") else float(parameters[layout.location(slot)])
            delivered_nodes.append(dict(node_id=identity, slot_index=slot, q_mm=q))
        before = _geometry_result(carrier, delivered_nodes, refs, supervised_parent["gaps"], residual,
            supervised_parent["is_closed"], truths)
        try:
            projected, projection = project_selected_parameters(carrier, delivered_nodes, refs,
                supervised_parent["gaps"], residual)
            after = _geometry_result(carrier, projected, refs, supervised_parent["gaps"], residual,
                supervised_parent["is_closed"], truths)
        except ValueError as error:
            projection = dict(projection_available=False, error=str(error))
            after = dict(parent_assembled=False, parent_error=str(error), original_children=len(truths),
                conditional_certified_children=0, child_results=[dict(original_child_order_metadata_only=i,
                    conditional_3mm_geometry_certificate_passed=False, parent_error=str(error)) for i in range(len(truths))])
        for result in (before, after):
            if len(result["child_results"]) != len(truths):
                raise ValueError("An invalid child disappeared from a conditioned denominator")
        modes[mode] = dict(before_projection=before, after_projection=after,
            process_projection=projection, supplied_slot_parameters=delivered_nodes)
    return dict(original_children=len(truths), root_raw_residual=residual.tolist(), modes=modes,
        all_children_and_shared_or_positive_gap_topology_retained=True,
        same_original_process_projection_and_continuous_certificate_used=True,
        oracle_addresses_parent_topology_and_closure_not_model_predictions=True,
        geometry_bounds_above3_are_unproved_not_exact_error_claims=True,
        weld_side_E16_and_free_parent_topology_not_assessed_by_this_condition=True,
        all_results_are_conditional_diagnostics_not_contract_TP=True, full_contract_pass=False)
