"""TRAIN-only full-count supervision, physical truth and normalized learning.

Evaluation keeps its frozen D3/mm rules and full eligible denominators.
"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from weld_pipeline.legacy_incremental_emission_loss_455 import resolved_views as original_resolved_views
from weld_pipeline.legacy_incremental_emission_loss_455 import emission_margin
from weld_pipeline.legacy_incremental_loss_396 import world_positions
from weld_pipeline.legacy_incremental_effective_nodes_422 import effective_nodes_torch
from weld_pipeline.legacy_scale_count_model_484 import positions_for_counts, NUMERIC_SCALES
from weld_pipeline.contract_field_acceptance_297 import FIELDS


def resolved_views(record):
    for view in original_resolved_views(record):
        if "protected_keep_child_mask" in record:
            order = view["canonical_permutation"]
            view["protected_keep_child_mask"] = np.asarray(record["protected_keep_child_mask"])[order]
            view["protected_keep_world_endpoints_mm"] = np.asarray(record["protected_keep_world_endpoints_mm"])[order]
        yield view


def candidate_output(output, index):
    return {name: output[name][index:index + 1] for name in (
        "frozen325_positions_mm", "position_increment_dimensionless", "frozen325_counts", "full_candidate_domain_mm")}


def full_count_nodes(output, candidate, count):
    raw = positions_for_counts(candidate_output(output, candidate), int(count))[0]
    return effective_nodes_torch(raw, output["gap_logits"][candidate])[:, :count]


def event_loss(output, parents_by_candidate, base, background_known, group_ids,
               hard_negative_weights, original_G, original_G_by_group):
    presence, counts = output["presence_logits"], output["count_logits"]
    b, q = presence.shape
    assert len(parents_by_candidate) == b and original_G > 0
    assert sum(original_G_by_group.values()) == original_G
    assert bool(torch.all(hard_negative_weights == 1.)), "Extra4 negatives prohibited in this stage"
    known = background_known.bool()[:, None].expand_as(presence).clone()
    known[group_ids == 2] = False  # Frozen HArc publishes its original geometry.
    yy, child_cost = torch.zeros_like(presence), torch.zeros_like(presence)
    groups = {g: {name: [] for name in ("geometry", "count", "relation", "physical", "keep", "maximum", "emission", "protection")}
              for g in (0, 1)}
    assignments = []
    failures = dict(count=0, gap=0, endpoint_visible_prefix=0, positive_out_of_domain=0)
    for si, records in enumerate(parents_by_candidate):
        group = int(group_ids[si])
        if group == 2 or not records:
            continue
        assert group in groups and len(records) <= q
        terms = groups[group]
        views_by_parent = [list(resolved_views(record)) for record in records]
        assert all(views_by_parent)
        count_log = counts[si].detach().log_softmax(-1).cpu().numpy()
        presence_cost = F.softplus(-presence[si]).detach().cpu().numpy()
        selected, cost, full_by_count = {}, np.empty((q, len(records))), {}
        for ti, views in enumerate(views_by_parent):
            alternatives = []
            for view in views:
                n = view["original_child_count"]
                if n not in full_by_count:
                    full_by_count[n] = full_count_nodes(output, si, n)
                nodes = full_by_count[n]
                got = world_positions(nodes.detach(), view["candidate_geometry"])
                wanted = got.new_tensor(view["resolved_original_world_endpoints_mm"])
                distance = torch.linalg.vector_norm(got - wanted[None], dim=-1).mean((-1, -2)) / 3.
                topology = F.binary_cross_entropy_with_logits(output["closed_logits"][si].detach(),
                    distance.new_full((q,), float(view["parent_is_closed"])), reduction="none") / 4.
                if n > 1:
                    gaps = distance.new_tensor([s == "POSITIVE_GAP" for s in view["canonical_gap_status"]]).expand(q, -1)
                    topology = topology + F.binary_cross_entropy_with_logits(output["gap_logits"][si, :, :n - 1].detach(), gaps, reduction="none").mean(-1) / 4.
                if view.get("direction_semantics_confirmed") is True:
                    topology = topology + F.binary_cross_entropy_with_logits(output["direction_logits"][si].detach(),
                        distance.new_full((q,), float(not view["negative_output_chart"])), reduction="none") / 4.
                alternatives.append((distance + topology).cpu().numpy() - count_log[:, n] / 4. + presence_cost)
            alternatives = np.stack(alternatives, axis=1)
            best = alternatives.argmin(1)
            cost[:, ti] = alternatives[np.arange(q), best]
            for qi, vi in enumerate(best.tolist()):
                selected[(qi, ti)] = views[vi]
        assert np.isfinite(cost).all()
        anchors = [(int(record["fixed_train_query"]), ti) for ti, record in enumerate(records) if record.get("fixed_train_query") is not None]
        assert len({qi for qi, ti in anchors}) == len(anchors)
        remaining_q = [qi for qi in range(q) if qi not in {a for a, t in anchors}]
        remaining_t = [ti for ti in range(len(records)) if ti not in {t for a, t in anchors}]
        pairs = list(anchors)
        if remaining_t:
            aq, at = linear_sum_assignment(cost[np.ix_(remaining_q, remaining_t)])
            pairs += [(remaining_q[qi], remaining_t[ti]) for qi, ti in zip(aq.tolist(), at.tolist())]
        assert len(pairs) == len(records) and len({ti for qi, ti in pairs}) == len(records)
        span = output["full_candidate_domain_mm"][si, 1] - output["full_candidate_domain_mm"][si, 0]
        assert float(span) > 0
        for qi, ti in pairs:
            view = selected[(qi, ti)]
            n = view["original_child_count"]
            yy[si, qi], known[si, qi], child_cost[si, qi] = 1., True, n
            nodes = full_by_count[n][qi]
            got = world_positions(nodes, view["candidate_geometry"])
            wanted = got.new_tensor(view["resolved_original_world_endpoints_mm"])
            error = torch.linalg.vector_norm(got - wanted, dim=-1)
            terms["geometry"].append(F.smooth_l1_loss(error / span, torch.zeros_like(error), beta=3. / float(span)))
            terms["maximum"].append(F.relu(error.max() - 3.) / span)
            count_loss = F.cross_entropy(counts[si, qi:qi + 1], torch.tensor([n], device=got.device)) / 4.
            terms["count"].append(count_loss)
            relation = F.binary_cross_entropy_with_logits(output["closed_logits"][si, qi], got.new_tensor(float(view["parent_is_closed"]))) / 4.
            if view.get("direction_semantics_confirmed") is True:
                relation = relation + F.binary_cross_entropy_with_logits(output["direction_logits"][si, qi], got.new_tensor(float(not view["negative_output_chart"]))) / 4.
            if n > 1:
                gaps = got.new_tensor([s == "POSITIVE_GAP" for s in view["canonical_gap_status"]])
                relation = relation + F.binary_cross_entropy_with_logits(output["gap_logits"][si, qi, :n - 1], gaps) / 4.
            terms["relation"].append(relation)
            domain = output["full_candidate_domain_mm"][si]
            physical = (F.relu(domain[0] - nodes).mean() + F.relu(nodes - domain[1]).mean()
                        + F.relu(1e-6 + nodes[:, 0] - nodes[:, 1]).mean()) / span
            if n > 1:
                physical = physical + F.relu(nodes[:-1, 1] - nodes[1:, 0]).mean() / span
            terms["physical"].append(physical)
            protected_mask = view.get("protected_keep_child_mask")
            if protected_mask is not None and bool(np.asarray(protected_mask).any()):
                mask = torch.as_tensor(protected_mask, dtype=torch.bool, device=got.device)
                reference = got.new_tensor(view["protected_keep_world_endpoints_mm"])
                assert bool(torch.isfinite(reference[mask]).all())
                keep_error = torch.linalg.vector_norm(got[mask] - reference[mask], dim=-1)
                keeping = F.smooth_l1_loss(keep_error / span, torch.zeros_like(keep_error), beta=3. / float(span))
                terms["keep"].append(keeping)
                # Partial correct children retain positive presence and the true
                # full parent count, without distilling a wrong parent partition.
                presence_keep = F.relu(presence.new_tensor(float(np.log(4.))) - presence[si, qi])
                competitors = torch.cat((counts[si, qi, :n], counts[si, qi, n + 1:]))
                count_keep = F.relu(2. - (counts[si, qi, n] - competitors.max())) / 4.
                protection = keeping + F.relu(keep_error.max() - 3.) / span + presence_keep + count_keep
                terms["protection"].append(protection)
            if view.get("protected_complete_parent"):
                margin = emission_margin(presence[si, qi], counts[si, qi], n)
                # Separate gains: presence1; count4.
                presence_part = F.relu(presence.new_tensor(float(np.log(4.))) - presence[si, qi])
                terms["emission"].append(presence_part + (margin - presence_part) / 4.)
            predicted_count = int(counts[si, qi].detach().argmax())
            failures["count"] += int(predicted_count != n)
            visible = min(predicted_count, n)
            failures["endpoint_visible_prefix"] += int(visible > 0 and float(error[:visible].detach().max()) > 3.)
            if predicted_count == n and n > 1:
                failures["gap"] += int((output["gap_logits"][si, qi, :n - 1].detach() >= 0.).cpu().tolist() != [s == "POSITIVE_GAP" for s in view["canonical_gap_status"]])
            actual = output["positions_mm"][si, qi, :predicted_count]
            failures["positive_out_of_domain"] += int(bool(((actual < domain[0]) | (actual > domain[1])).any()))
            assignments.append(dict(candidate=si, predicted_parent_query=qi,
                original_parent_index=view["original_parent_index"], original_child_count=n,
                canonical_original_child_ids=view["canonical_original_child_ids"],
                canonical_gap_status=view["canonical_gap_status"], resolved_view_index=view["resolved_view_index"],
                canonical_permutation=view["canonical_permutation"], negative_output_chart=view["negative_output_chart"],
                reserved_train_anchor=records[ti].get("fixed_train_query") is not None,
                protected_complete_parent=bool(view.get("protected_complete_parent"))))
    zero = presence.sum() * 0.
    mean = lambda xs: torch.stack(xs).mean() if xs else zero
    positive, negative = known & (yy > .5), known & (yy <= .5)
    prediction_cost = counts.detach().argmax(-1).clamp_min(1).to(presence.dtype)
    group_terms, group_losses, protection_losses = {}, [], {}
    for group in (0, 1):
        chosen = (group_ids == group)[:, None].expand_as(known)
        p, z = chosen & positive, chosen & negative
        denominator = max(1, int(original_G_by_group[group]))
        pos = (F.softplus(-presence[p]) * child_cost[p]).sum() / denominator
        neg = (F.softplus(presence[z]) * prediction_cost[z]).sum() / denominator
        terms = groups[group]
        keep = mean(terms["protection"])
        protection_losses[group] = keep
        group_terms[group] = dict(positive=pos, negative=neg, emission_keep=keep)
        group_losses.append(pos + neg + mean(terms["count"]) + 2. * mean(terms["geometry"])
            + mean(terms["relation"]) + .1 * mean(terms["physical"]) + mean(terms["keep"])
            + mean(terms["maximum"]) + mean(terms["emission"]))
    active = (group_ids != 2)[:, None, None, None].expand_as(output["position_increment_dimensionless"])
    regularization = output["position_increment_dimensionless"][active].square().mean() if bool(active.any()) else zero
    total = torch.stack(group_losses).mean() + .001 * regularization
    assert bool(torch.isfinite(total))
    return total, dict(assignments=assignments, group_presence_terms=group_terms,
        protection_losses=protection_losses, residual_regularization=regularization,
        training_correspondence_failure_counts=failures, original_G=original_G,
        original_positive_parents=sum(map(len, parents_by_candidate)), known_parent_presence_cells=int(known.sum()),
        geometry_supervised_children=sum(row["original_child_count"] for row in assignments),
        full_GT_count_geometry_used=True, HArc_geometry_training_frozen=True,
        protected_complete_parent_count=sum(row["protected_complete_parent"] for row in assignments),
        reserved_train_anchor_count=sum(row["reserved_train_anchor"] for row in assignments))


def attribute_loss(output, target, totals):
    """Exact source field macro average, even when processed in chunks."""
    state, values = output["state_logits"], output["scalar_values"]
    zero = state.sum() * 0.
    terms = []
    active_fields = [j for j in range(16) if totals["state_known"][j] or totals["scalar_known"][j]
                     or (FIELDS[j].startswith("normal_") and totals["normal_known"][int(FIELDS[j][-1]) - 1])]
    for j in active_fields:
        value = zero
        known = target["state_known"][:, j].bool()
        if bool(known.any()):
            ce = F.cross_entropy(state[known, j], target["state_class"][known, j].long(), reduction="none")
            value = value + (ce * target["grade_weight"][known, j]).sum() / (4. * totals["state_known"][j])
        known = target["scalar_known"][:, j].bool()
        if bool(known.any()):
            wanted = target["scalar_values"][known, j]
            assert bool(torch.isfinite(wanted).all())
            if FIELDS[j] in ("has_eave", "side_plate_split"):
                loss = F.binary_cross_entropy_with_logits(values[known, j], wanted, reduction="none")
            else:
                tolerance = 1. if FIELDS[j].startswith("thickness_") else 3.
                scale = NUMERIC_SCALES[j]
                loss = F.smooth_l1_loss((values[known, j] - wanted) / scale, torch.zeros_like(wanted), beta=tolerance / scale, reduction="none")
            value = value + (loss * target["grade_weight"][known, j]).sum() / totals["scalar_known"][j]
        if FIELDS[j].startswith("normal_"):
            side = int(FIELDS[j][-1]) - 1
            known = target["normal_known"][:, side].bool()
            if bool(known.any()):
                wanted = target["signed_normals"][known, side]
                assert bool(torch.isfinite(wanted).all())
                predicted = output["signed_unit_normals"][known, side]
                directed = 1. - (predicted * wanted).sum(-1)
                unit = F.relu((output["normal_lengths"][known, side] - 1.).abs() - .001)
                value = value + ((directed + .1 * unit) * target["grade_weight"][known, j]).sum() / totals["normal_known"][side]
        terms.append(value)
    loss = torch.stack(terms).sum() / max(1, len(active_fields)) if terms else zero
    assert bool(torch.isfinite(loss))
    return loss
