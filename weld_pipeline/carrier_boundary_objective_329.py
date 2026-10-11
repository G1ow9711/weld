"""Complete-source objective with bounded memory and exact streamed gradients."""
import math
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from .carrier_boundary_model_328 import ROLE_FIELDS, SEQUENCE_HALO_ROWS
from .carrier_observation_sequences_328 import SequenceLayout
from .bounded_analytic_residuals_327 import differentiable_point_at
from .carrier_pointer_geometry_329 import legal_terminal_mask, learned_parameters, relation_features

VERSION = "COMPLETE_SOURCE_FULL_TERMINAL_DOMAIN_MACRO_OBJECTIVE_329_v1"
BLOCK_ROWS = 8192
COUNT_WEIGHT = .1
POINTER_END_ROLE_WEIGHT = 1.
RELATION_FIELDS = ("NO_ADJACENT_LINK", "SHARED", "POSITIVE_GAP")


def encode_block(model, tokens, reader, ordinal, layout, first=0, last=None):
    last = layout.rows if last is None else last
    lo, hi = max(0, first - SEQUENCE_HALO_ROWS), min(layout.rows, last + SEQUENCE_HALO_ROWS)
    features, groups = reader.observations(ordinal, lo, hi)
    sequence = model.encode_sequence(torch.from_numpy(features), tokens[torch.from_numpy(groups)], tokens.mean(0))
    return sequence[first - lo:last - lo]


def role_loss(logits, layout, first, last, carrier, supervision):
    active = torch.from_numpy(layout.active_slots[first:last])
    values = logits[active]
    negative_weight, positive_weight = supervision.role_weights(carrier)
    nw = values.new_tensor(negative_weight)
    loss = (F.softplus(values) * nw).sum()
    positives = supervision.roles_by_carrier.get(carrier.ordinal, ())
    for role, slots in enumerate(positives):
        for slot in slots:
            row, sub = layout.location(slot)
            if first <= row < last:
                value = logits[row - first, sub, role]
                loss = loss - F.softplus(value) * nw[role] + F.softplus(-value) * float(positive_weight[role])
    return loss


def positive_losses(model, heads, layout, carrier, supervision):
    """Labels select loss terms only; complete terminal logits always remain."""
    terms = {key: heads["role_logits"].sum() * 0. for key in ("pointer", "type", "offset", "geometry", "relation")}
    q_values = learned_parameters(heads, layout)
    active = torch.from_numpy(layout.active_slots)
    keys = heads["terminal_keys"][active]
    ids = layout.slot_ids[layout.active_slots]
    terminal_rank = np.empty(carrier.slot_count, np.int64)
    terminal_rank[ids] = np.arange(len(ids))
    end_roles = heads["role_logits"][active][:, 2]
    embedding = lambda slot: heads["embeddings"][layout.location(slot)]
    parameter = lambda slot: q_values[layout.location(slot)]
    all_children = []
    child_weight = supervision.child_weight(carrier)
    boundary_weight = supervision.boundary_weight(carrier)
    for parent in supervision.parents_by_carrier[carrier.ordinal]:
        first_slot = parent["children"][0]["start_slot"]
        raw = heads["residual"][layout.location(first_slot)]
        for node in parent["nodes"].values():
            slot, target_q = int(node["slot_index"]), float(node["q_mm"])
            row, sub = layout.location(slot)
            width = layout.upper_mm[row] - layout.lower_mm[row]
            if width:
                target_fraction = (target_q - layout.lower_mm[row]) / width
                if not -1e-12 <= target_fraction <= 1. + 1e-12:
                    raise ValueError("Supervision outside registered finite cell")
                fraction = heads["within_cell_fraction"][row, sub]
                terms["offset"] = terms["offset"] + boundary_weight * F.smooth_l1_loss(
                    fraction, fraction.new_tensor(np.clip(target_fraction, 0., 1.)), beta=.1)
        for index, child in enumerate(parent["children"]):
            start, end = child["start_slot"], child["end_slot"]
            allowed = torch.from_numpy(legal_terminal_mask(carrier, layout, start)[layout.active_slots])
            if not bool(allowed[terminal_rank[end]]):
                raise ValueError("Registered true terminal was excluded from complete legal domain")
            logits = model.pointer_logits(embedding(start), keys) + POINTER_END_ROLE_WEIGHT * end_roles
            logits = logits.masked_fill(~allowed, -torch.inf)
            terms["pointer"] = terms["pointer"] + child_weight * F.cross_entropy(
                logits[None], torch.tensor([int(terminal_rank[end])]))
            target_type = int(carrier.weld_type == "vertical")
            terms["type"] = terms["type"] + child_weight * F.cross_entropy(
                heads["weld_type_logits"][layout.location(start)][None], torch.tensor([target_type]))
            samples = torch.as_tensor(child["loss_samples_mm"], dtype=torch.float64)
            t = torch.linspace(0., 1., len(samples), dtype=torch.float64)
            q = parameter(start) * (1. - t) + parameter(end) * t
            predicted = differentiable_point_at(carrier, q, raw)
            distances = torch.linalg.vector_norm(predicted - samples, dim=1) / 3.
            scalar = lambda x: F.smooth_l1_loss(x, torch.zeros_like(x), beta=1. / 3., reduction="none")
            geometry = scalar(distances[[0, -1]]).mean() + scalar(distances).mean() + scalar(distances.max())
            terms["geometry"] = terms["geometry"] + child_weight * geometry
            all_children.append(dict(parent=parent["original_parent_index"], position=index,
                start=start, end=end, parent_record=parent))
    relation_terms = []
    for a in all_children:
        for b in all_children:
            if a is b:
                continue
            adjacent = a["parent"] == b["parent"] and b["position"] == a["position"] + 1
            label = (1 if a["parent_record"]["gaps"][a["position"]] == "SHARED" else 2) if adjacent else 0
            sa, ea, sb, eb = a["start"], a["end"], b["start"], b["end"]
            da = parameter(ea) - parameter(sa)
            db = parameter(eb) - parameter(sb)
            extra = relation_features(parameter(ea), parameter(sb), ea == sb, bool((da * db).detach() > 0))
            logits = model.relation_logits(embedding(sa), embedding(ea), embedding(sb), embedding(eb), extra)
            relation_terms.append((label, F.cross_entropy(logits[None], torch.tensor([label]))))
    # Macro means across observed link classes; absent positive-gap truth is
    # explicitly unlearned, not fabricated. Average positive carriers equally.
    if relation_terms:
        by_label = defaultdict(list)
        for label, term in relation_terms:
            by_label[label].append(term)
        terms["relation"] = torch.stack([torch.stack(v).mean() for v in by_label.values()]).mean()
        terms["relation"] = terms["relation"] / len(supervision.parents_by_carrier)
    return terms


def count_loss_and_derivative(expected_count, reference_count):
    """Derivative of 0.1*SmoothL1(log1p(S)-log1p(G)), beta=1."""
    difference = math.log1p(expected_count) - math.log1p(reference_count)
    value = .5 * difference * difference if abs(difference) < 1. else abs(difference) - .5
    slope = difference if abs(difference) < 1. else math.copysign(1., difference)
    return COUNT_WEIGHT * value, COUNT_WEIGHT * slope / (1. + expected_count)


def complete_source_backward(model, source, carriers, reader, supervision, progress=None):
    """Two full cell passes, one all-original context, then exact chain rule.

The first cell pass measures S with fixed current weights. The second contributes
dL/dS times each local count gradient, plus complete supervised losses. Proxy
token gradients are returned once to the full original-point encoder.
"""
    report = progress or (lambda **values: None)
    model.train()
    model.zero_grad(set_to_none=True)
    tokens, context_receipt = model.context(source)
    proxy = tokens.detach().requires_grad_(True)
    expected_count, first_slots, forwards = 0., 0, 0
    with torch.no_grad():
        for ordinal, carrier in enumerate(carriers):
            layout = SequenceLayout.from_carrier(carrier)
            for first in range(0, layout.rows, BLOCK_ROWS):
                last = min(layout.rows, first + BLOCK_ROWS)
                sequence = encode_block(model, proxy, reader, ordinal, layout, first, last)
                logits = model.roles(model.slot_embeddings(sequence))
                active = torch.from_numpy(layout.active_slots[first:last])
                expected_count += float(logits[..., 1][active].sigmoid().to(torch.float64).sum())
                first_slots += int(active.sum())
                forwards += 1
            if ordinal % 100 == 0 or ordinal + 1 == len(carriers):
                report(stage="EXACT_COUNT_ALL_FINITE_CELLS", complete_carriers=ordinal + 1,
                    original_carriers=len(carriers), complete_slots=first_slots, sequence_forwards=forwards)
    if first_slots != supervision.complete_slots:
        raise ValueError("Count pass omitted declared slots")
    count_value, count_slope = count_loss_and_derivative(expected_count, supervision.original_children)
    totals = dict(role=0., pointer=0., type=0., offset=0., geometry=0., relation=0., count=count_value)
    second_slots, second_count, pointer_children = 0, 0., 0
    for ordinal, carrier in enumerate(carriers):
        layout = SequenceLayout.from_carrier(carrier)
        if ordinal in supervision.parents_by_carrier:
            sequence = encode_block(model, proxy, reader, ordinal, layout)
            heads = model.heads(sequence)
            active = torch.from_numpy(layout.active_slots)
            role = role_loss(heads["role_logits"], layout, 0, layout.rows, carrier, supervision)
            positive = positive_losses(model, heads, layout, carrier, supervision)
            local_count = heads["role_logits"][..., 1][active].sigmoid().to(torch.float64).sum()
            loss = role + sum(positive.values()) + count_slope * local_count
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite full positive-carrier objective")
            loss.backward()
            totals["role"] += float(role.detach())
            for key, value in positive.items():
                totals[key] += float(value.detach())
            second_count += float(local_count.detach())
            second_slots += int(active.sum())
            pointer_children += sum(len(p["children"]) for p in supervision.parents_by_carrier[ordinal])
            forwards += 1
            del heads, sequence, positive, role, loss, local_count
        else:
            for first in range(0, layout.rows, BLOCK_ROWS):
                last = min(layout.rows, first + BLOCK_ROWS)
                sequence = encode_block(model, proxy, reader, ordinal, layout, first, last)
                logits = model.roles(model.slot_embeddings(sequence))
                active = torch.from_numpy(layout.active_slots[first:last])
                role = role_loss(logits, layout, first, last, carrier, supervision)
                local_count = logits[..., 1][active].sigmoid().to(torch.float64).sum()
                loss = role + count_slope * local_count
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite complete background objective")
                loss.backward()
                totals["role"] += float(role.detach())
                second_count += float(local_count.detach())
                second_slots += int(active.sum())
                forwards += 1
                del sequence, logits, role, local_count, loss
        if ordinal % 100 == 0 or ordinal + 1 == len(carriers):
            report(stage="FULL_CELL_ROLE_POINTER_GEOMETRY_BACKWARD", complete_carriers=ordinal + 1,
                original_carriers=len(carriers), complete_slots=second_slots, sequence_forwards=forwards,
                complete_original_child_pointer_losses=pointer_children)
    if second_slots != first_slots or pointer_children != supervision.original_children:
        raise ValueError("Incomplete full source gradients")
    relative = abs(second_count - expected_count) / max(1., expected_count)
    if relative > 1e-5 or proxy.grad is None or not bool(torch.isfinite(proxy.grad).all()):
        raise ValueError("Fixed two-pass count or streamed context gradients disagree")
    tokens.backward(proxy.grad)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError("Nonfinite true model gradient")
    point_participation = any(p.grad is not None and float(p.grad.abs().max()) > 0
        for p in model.context.point_encoder.parameters())
    if not point_participation:
        raise ValueError("Original XYZ point encoder did not participate in full source loss")
    return dict(version=VERSION, context_receipt=context_receipt, loss_terms=totals,
        loss_value=sum(totals.values()), expected_child_count=expected_count,
        count_two_pass_relative_difference=relative, complete_background_and_positive_slots=second_slots,
        full_legal_terminal_domain_positive_children=pointer_children, sequence_forwards=forwards,
        original_point_encoder_gradient_participated=point_participation,
        target_count_used_only_in_training_count_loss=True, target_windows_used=False,
        actual_optimizer_updates=0, full_contract_pass=False)
