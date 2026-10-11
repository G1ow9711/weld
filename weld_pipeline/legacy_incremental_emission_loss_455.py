"""TRAIN-only stable parent matching, resolved D3 views and emission protection."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from weld_pipeline.legacy_incremental_correspondence_414 import target_variants
from weld_pipeline.legacy_incremental_loss_396 import canonical_target, world_positions
from weld_pipeline.legacy_incremental_effective_nodes_422 import effective_nodes_torch


def resolved_views(record):
    original_ids = list(record['original_child_ids'])
    original_negative = record['parent_chart_direction'] == -1
    canonical_ids = original_ids[::-1] if original_negative else original_ids
    assert len(canonical_ids) == len(set(canonical_ids)) == record['original_child_count']
    for vi, view in enumerate(target_variants(record)):
        positions, negative = canonical_target(view)
        ids = view['original_child_ids'][::-1] if negative else view['original_child_ids']
        permutation = [canonical_ids.index(cid) for cid in ids]
        assert sorted(permutation) == list(range(len(canonical_ids)))
        resolved = dict(view, resolved_view_index=vi, canonical_positions_mm=positions,
                        canonical_original_child_ids=list(ids), canonical_permutation=permutation,
                        negative_output_chart=negative,
                        canonical_gap_status=view['original_gap_status'][::-1] if negative else view['original_gap_status'])
        resolved['resolved_original_world_endpoints_mm'] = np.asarray(record['canonical_original_world_endpoints_mm'])[permutation]
        # These two current-reference arrays already use the positive chart.
        for key in ('corrective_keep_child_mask', 'corrective_keep_world_endpoints_mm'):
            if key in record:
                resolved[key] = np.asarray(record[key])[permutation]
        if 'teacher_correct_child_mask' in record:
            mask = list(record['teacher_correct_child_mask'])
            if original_negative:
                mask = mask[::-1]
            resolved['teacher_correct_canonical_mask'] = np.asarray(mask, bool)[permutation]
        yield resolved


def emission_margin(presence, counts, correct_count):
    competitors = torch.cat((counts[:correct_count], counts[correct_count + 1:]))
    assert len(competitors) == len(counts) - 1
    return F.relu(presence.new_tensor(math.log(4.)) - presence) + F.relu(2. - (counts[correct_count] - competitors.max()))


def event_loss(output, parents_by_candidate, base, background_known, group_ids, hard_negative_weights, original_G):
    presence, counts = output['presence_logits'], output['count_logits']
    positions = effective_nodes_torch(output['positions_mm'], output['gap_logits'])
    b, q = presence.shape
    assert len(parents_by_candidate) == b and original_G > 0
    known = background_known.to(device=presence.device, dtype=torch.bool)[:, None].expand_as(presence).clone()
    yy, child_cost = torch.zeros_like(presence), torch.zeros_like(presence)
    geometry, count_terms, relation, physics, teacher_keep, current_keep, maximum, emit = ([] for _ in range(8))
    assignments = []
    emitted_by_group = {int(g): [] for g in torch.unique(group_ids).tolist()}
    failures = dict(count=0, gap=0, endpoint_visible_prefix=0, positive_out_of_domain=0)
    for si, records in enumerate(parents_by_candidate):
        if not records:
            continue
        assert len(records) <= q
        options = [list(resolved_views(record)) for record in records]
        assert all(options)
        count_log = counts[si].detach().log_softmax(-1).cpu().numpy()
        presence_cost = F.softplus(-presence[si]).detach().cpu().numpy()
        cost, selected = np.empty((q, len(records))), {}
        for ti, views in enumerate(options):
            view_costs = []
            for view in views:
                n = view['original_child_count']
                assert 0 < n <= positions.shape[2]
                got = world_positions(positions[si, :, :n].detach(), view['candidate_geometry'])
                wanted = torch.as_tensor(view['resolved_original_world_endpoints_mm'], dtype=got.dtype, device=got.device)
                distance = torch.linalg.vector_norm(got - wanted[None], dim=-1).mean((-1, -2)) / 3.
                topology = F.binary_cross_entropy_with_logits(output['closed_logits'][si].detach(), distance.new_full((q,), float(view['parent_is_closed'])), reduction='none')
                if n > 1:
                    gaps = distance.new_tensor([s == 'POSITIVE_GAP' for s in view['canonical_gap_status']]).expand(q, -1)
                    topology = topology + F.binary_cross_entropy_with_logits(output['gap_logits'][si, :, :n - 1].detach(), gaps, reduction='none').mean(-1)
                if view.get('direction_semantics_confirmed') is True:
                    topology = topology + F.binary_cross_entropy_with_logits(output['direction_logits'][si].detach(), distance.new_full((q,), float(not view['negative_output_chart'])), reduction='none')
                view_costs.append((distance + topology).cpu().numpy() - count_log[:, n] + presence_cost)
            alternatives = np.stack(view_costs, axis=1)
            best = alternatives.argmin(1)
            cost[:, ti] = alternatives[np.arange(q), best]
            for qi, vi in enumerate(best.tolist()):
                selected[(qi, ti)] = views[vi]
        assert np.isfinite(cost).all()
        anchored = [(int(record['fixed_train_query']), ti) for ti, record in enumerate(records) if record.get('fixed_train_query') is not None]
        assert len({qi for qi, ti in anchored}) == len(anchored)
        assert all(0 <= qi < q and records[ti]['protected_complete_parent'] for qi, ti in anchored)
        remaining_q = [qi for qi in range(q) if qi not in {a for a, t in anchored}]
        remaining_t = [ti for ti in range(len(records)) if ti not in {t for a, t in anchored}]
        pairs = list(anchored)
        if remaining_t:
            qq, tt = linear_sum_assignment(cost[np.ix_(remaining_q, remaining_t)])
            pairs.extend((remaining_q[a], remaining_t[t]) for a, t in zip(qq.tolist(), tt.tolist()))
        assert len(pairs) == len(records) and len({t for qi, t in pairs}) == len(records)
        for qi, ti in pairs:
            view = selected[(qi, ti)]
            n = view['original_child_count']
            yy[si, qi], known[si, qi], child_cost[si, qi] = 1., True, n
            got_parameters = positions[si, qi, :n]
            got = world_positions(got_parameters, view['candidate_geometry'])
            wanted = torch.as_tensor(view['resolved_original_world_endpoints_mm'], dtype=got.dtype, device=got.device)
            error = torch.linalg.vector_norm(got - wanted, dim=-1)
            geometry.append(F.smooth_l1_loss(error, torch.zeros_like(error), beta=1.))
            maximum.append(F.relu(error.max() - 3.))
            count_terms.append(F.cross_entropy(counts[si, qi:qi + 1], torch.tensor([n], device=got.device)))
            if view.get('direction_semantics_confirmed') is True:
                relation.append(F.binary_cross_entropy_with_logits(output['direction_logits'][si, qi], got.new_tensor(float(not view['negative_output_chart']))))
            relation.append(F.binary_cross_entropy_with_logits(output['closed_logits'][si, qi], got.new_tensor(float(view['parent_is_closed']))))
            if n > 1:
                gaps = got.new_tensor([s == 'POSITIVE_GAP' for s in view['canonical_gap_status']])
                relation.append(F.binary_cross_entropy_with_logits(output['gap_logits'][si, qi, :n - 1], gaps))
            domain = output['full_candidate_domain_mm'][si]
            physical = F.relu(domain[0] - got_parameters).mean() + F.relu(got_parameters - domain[1]).mean() + F.relu(1e-6 + got_parameters[:, 0] - got_parameters[:, 1]).mean()
            if n > 1:
                physical = physical + F.relu(got_parameters[:-1, 1] - got_parameters[1:, 0]).mean()
            physics.append(physical)
            if 'teacher_correct_canonical_mask' in view:
                mask = torch.as_tensor(view['teacher_correct_canonical_mask'], device=got.device)
                if bool(mask.any()):
                    teacher_query = view['teacher_correct_query_index']
                    teacher = effective_nodes_torch(base['positions_mm'][si, teacher_query], base['gap_logits'][si, teacher_query])[:n]
                    teacher = teacher[view['canonical_permutation']]
                    teacher_keep.append(F.smooth_l1_loss(got_parameters[mask], teacher[mask], beta=1.))
            if 'corrective_keep_child_mask' in view:
                mask = torch.as_tensor(view['corrective_keep_child_mask'], dtype=torch.bool, device=got.device)
                if bool(mask.any()):
                    reference = torch.as_tensor(view['corrective_keep_world_endpoints_mm'], dtype=got.dtype, device=got.device)
                    assert torch.isfinite(reference[mask]).all()
                    current_keep.append(F.smooth_l1_loss(got[mask], reference[mask], beta=1.))
            if view.get('protected_complete_parent'):
                value = emission_margin(presence[si, qi], counts[si, qi], n)
                emit.append(value)
                emitted_by_group[int(group_ids[si])].append(value)
            predicted_count = int(counts[si, qi].detach().argmax())
            failures['count'] += int(predicted_count != n)
            visible = min(predicted_count, n)
            failures['endpoint_visible_prefix'] += int(visible > 0 and float(error[:visible].detach().max()) > 3.)
            if predicted_count == n and n > 1:
                failures['gap'] += int((output['gap_logits'][si, qi, :n - 1].detach() >= 0.).cpu().tolist() != [s == 'POSITIVE_GAP' for s in view['canonical_gap_status']])
            actual = positions[si, qi, :predicted_count]
            failures['positive_out_of_domain'] += int(bool(((actual < domain[0]) | (actual > domain[1])).any()))
            assignments.append(dict(candidate=si, predicted_parent_query=qi, original_parent_index=view['original_parent_index'],
                original_child_count=n, resolved_view_index=view['resolved_view_index'],
                canonical_original_child_ids=view['canonical_original_child_ids'], canonical_permutation=view['canonical_permutation'],
                canonical_gap_status=view['canonical_gap_status'], negative_output_chart=view['negative_output_chart'],
                reserved_train_anchor=records[ti].get('fixed_train_query') is not None,
                protected_complete_parent=bool(view.get('protected_complete_parent'))))
    zero = presence.sum() * 0.
    mean = lambda values: torch.stack(values).mean() if values else zero
    hard = hard_negative_weights.to(device=presence.device, dtype=presence.dtype)
    assert hard.shape == presence.shape and bool(torch.all((hard == 1.) | (hard == 4.)))
    positive, negative = known & (yy > .5), known & (yy <= .5)
    predicted_cost = counts.detach().argmax(-1).clamp_min(1).to(presence.dtype)
    positive_loss = (F.softplus(-presence[positive]) * child_cost[positive]).sum() / float(original_G)
    negative_loss = (F.softplus(presence[negative]) * hard[negative] * predicted_cost[negative]).sum() / float(original_G)
    group_terms = {}
    for group in torch.unique(group_ids).tolist():
        chosen = (group_ids == group)[:, None].expand_as(known)
        p, z = chosen & positive, chosen & negative
        group_terms[int(group)] = dict(
            positive=(F.softplus(-presence[p]) * child_cost[p]).sum() / float(original_G),
            negative=(F.softplus(presence[z]) * hard[z] * predicted_cost[z]).sum() / float(original_G),
            emission_keep=sum(emitted_by_group[int(group)], zero) / max(1, len(emit)))
    event = positive_loss + negative_loss + mean(count_terms)
    raw_regularization = (output['positions_mm'] - base['positions_mm']).square().mean()
    total = event + 2. * mean(geometry) + mean(relation) + mean(teacher_keep) + .1 * mean(physics) + .001 * raw_regularization + mean(current_keep) + mean(maximum) + mean(emit)
    assert torch.isfinite(total)
    for assigned in assignments:
        if assigned['reserved_train_anchor']:
            assert bool(positive[assigned['candidate'], assigned['predicted_parent_query']])
            assert not bool(negative[assigned['candidate'], assigned['predicted_parent_query']])
    return total, dict(event=event, geometry_mm=mean(geometry), relation=mean(relation), physical=mean(physics),
        residual_regularization=raw_regularization, known_correct_teacher_keep=mean(teacher_keep),
        correct_current_V_world_keep=mean(current_keep), full_GT_endpoint_maximum_hinge_mm=mean(maximum),
        protected_emission_margin_keep=mean(emit), assignments=assignments, group_presence_terms=group_terms,
        training_correspondence_failure_counts=failures, original_positive_parents=sum(map(len, parents_by_candidate)),
        known_parent_presence_cells=int(known.sum()), protected_complete_parent_count=len(emit),
        reserved_train_anchor_count=sum(row['reserved_train_anchor'] for row in assignments),
        all_GT_children_in_maximum_geometry=True, D3_correspondence_resolved_once=True)
