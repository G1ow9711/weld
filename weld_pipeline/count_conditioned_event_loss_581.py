"""TRAIN-only count-conditioned supervision with independent group losses.

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
from weld_pipeline.legacy_legal_allocation_model_518 import legal_positions_for_counts, margin_logits_for_counts
from weld_pipeline.legacy_scale_count_model_484 import NUMERIC_SCALES
from weld_pipeline.contract_field_acceptance_297 import FIELDS


def resolved_views(record):
    for view in original_resolved_views(record):
        if "protected_keep_child_mask" in record:
            order = view["canonical_permutation"]
            view["protected_keep_child_mask"] = np.asarray(record["protected_keep_child_mask"])[order]
            view["protected_keep_world_endpoints_mm"] = np.asarray(record["protected_keep_world_endpoints_mm"])[order]
        yield view


def full_count_nodes(output, candidate, count, events):
    batch = len(output['presence_logits'])
    part = {name: value[candidate:candidate + 1] for name, value in output.items()
            if isinstance(value, torch.Tensor) and value.ndim and len(value) == batch}
    return events.positions_for_requested_counts(part, int(count))[0, :, :count]


def event_loss(output, parents_by_candidate, base, background_known, group_ids,
               hard_negative_weights, original_G, original_G_by_group, *, events):
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
                    full_by_count[n] = full_count_nodes(output, si, n, events)
                nodes = full_by_count[n]
                got = world_positions(nodes.detach(), view["candidate_geometry"])
                wanted = got.new_tensor(view["resolved_original_world_endpoints_mm"])
                distance = torch.linalg.vector_norm(got - wanted[None], dim=-1).mean((-1, -2)) / 3.
                topology = F.binary_cross_entropy_with_logits(output["closed_logits"][si].detach(),
                    distance.new_full((q,), float(view["parent_is_closed"])), reduction="none")
                if n > 1:
                    gaps = distance.new_tensor([s == "POSITIVE_GAP" for s in view["canonical_gap_status"]]).expand(q, -1)
                    topology = topology + F.binary_cross_entropy_with_logits(output["gap_logits"][si, :, :n - 1].detach(), gaps, reduction="none").mean(-1)
                if view.get("direction_semantics_confirmed") is True:
                    topology = topology + F.binary_cross_entropy_with_logits(output["direction_logits"][si].detach(),
                        distance.new_full((q,), float(not view["negative_output_chart"])), reduction="none")
                alternatives.append((distance + topology).cpu().numpy() - count_log[:, n] + presence_cost)
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
            terms["geometry"].append(F.smooth_l1_loss(error / 3., torch.zeros_like(error), beta=1.))
            terms["maximum"].append(F.relu(error.max() / 3. - 1.))
            count_loss = F.cross_entropy(counts[si, qi:qi + 1], torch.tensor([n], device=got.device))
            terms["count"].append(count_loss)
            relation = F.binary_cross_entropy_with_logits(output["closed_logits"][si, qi], got.new_tensor(float(view["parent_is_closed"])))
            if view.get("direction_semantics_confirmed") is True:
                relation = relation + F.binary_cross_entropy_with_logits(output["direction_logits"][si, qi], got.new_tensor(float(not view["negative_output_chart"])))
            if n > 1:
                gaps = got.new_tensor([s == "POSITIVE_GAP" for s in view["canonical_gap_status"]])
                relation = relation + F.binary_cross_entropy_with_logits(output["gap_logits"][si, qi, :n - 1], gaps)
            requested = torch.full_like(output['frozen325_counts'], n)
            margin_logits = margin_logits_for_counts(output, requested)[si, qi]
            target_nodes = np.asarray(view['canonical_positions_mm'])
            domain = output['full_candidate_domain_mm'][si]
            margin_truth = got.new_tensor([float(target_nodes[0, 0] > float(domain[0])),
                                          float(target_nodes[-1, 1] < float(domain[1]))])
            relation = relation + F.binary_cross_entropy_with_logits(margin_logits, margin_truth)
            terms["relation"].append(relation)
            domain = output["full_candidate_domain_mm"][si]
            physical = (F.relu(domain[0] - nodes).mean() + F.relu(nodes - domain[1]).mean()
                        + F.relu(nodes[:, 0] - nodes[:, 1]).mean()) / span
            if n > 1:
                physical = physical + F.relu(nodes[:-1, 1] - nodes[1:, 0]).mean() / span
            terms["physical"].append(physical)
            protected_mask = view.get("protected_keep_child_mask")
            if protected_mask is not None and bool(np.asarray(protected_mask).any()):
                mask = torch.as_tensor(protected_mask, dtype=torch.bool, device=got.device)
                reference = got.new_tensor(view["protected_keep_world_endpoints_mm"])
                assert bool(torch.isfinite(reference[mask]).all())
                keep_error = torch.linalg.vector_norm(got[mask] - reference[mask], dim=-1)
                keeping = F.smooth_l1_loss(keep_error / 3., torch.zeros_like(keep_error), beta=1.)
                terms["keep"].append(keeping)
                # Partial correct children retain positive presence and the true
                # full parent count, without distilling a wrong parent partition.
                presence_keep = F.relu(presence.new_tensor(float(np.log(4.))) - presence[si, qi])
                competitors = torch.cat((counts[si, qi, :n], counts[si, qi, n + 1:]))
                count_keep = F.relu(2. - (counts[si, qi, n] - competitors.max()))
                protection = keeping + F.relu(keep_error.max() / 3. - 1.) + presence_keep + count_keep
                terms["protection"].append(protection)
            if view.get("protected_complete_parent"):
                margin = emission_margin(presence[si, qi], counts[si, qi], n)
                # Frozen prior plus gain1 independent increments.
                presence_part = F.relu(presence.new_tensor(float(np.log(4.))) - presence[si, qi])
                terms["emission"].append(presence_part + (margin - presence_part))
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
    regularization_by_group = {}
    for group in (0, 1):
        chosen = group_ids == group
        regularization_by_group[group] = (output["allocation_increment_logits"][chosen].square().mean()
            if bool(chosen.any()) else zero)
    loss_by_group = {group:group_losses[group] + .001 * regularization_by_group[group] for group in (0, 1)}
    regularization = torch.stack(list(regularization_by_group.values())).mean()
    total = torch.stack(list(loss_by_group.values())).mean()
    assert bool(torch.isfinite(total))
    return total, dict(loss_by_group=loss_by_group, assignments=assignments, group_presence_terms=group_terms,
        protection_losses=protection_losses, residual_regularization=regularization,
        training_correspondence_failure_counts=failures, original_G=original_G,
        original_positive_parents=sum(map(len, parents_by_candidate)), known_parent_presence_cells=int(known.sum()),
        geometry_supervised_children=sum(row["original_child_count"] for row in assignments),
        full_GT_count_geometry_used=True, HArc_geometry_training_frozen=True, geometry_tolerance_normalizer_mm=3.,
        whole_line_bidirectional_error_bounded_by_same_chart_endpoint_maximum=True,
        protected_complete_parent_count=sum(row["protected_complete_parent"] for row in assignments),
        reserved_train_anchor_count=sum(row["reserved_train_anchor"] for row in assignments))

