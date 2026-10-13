"""Learned directed parent candidates that preserve their finite input curves.

Each child retains both orientations. Shared boundaries require the same
authenticated input event and exact coordinates. No endpoint averaging, arc
refitting, reference reader or business-frame invention occurs here.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn

from .contract_curve_geometry import CircularArc
from .server_predicted_parent_pool import proposal_edges


SCHEMA = 'SUPPORTED_FINITE_DIRECTED_PARENT_POOL_V1'
EVENT_CONSTRUCTIONS = {'OBSERVED_BOUNDARY', 'INPUT_CURVE_INTERSECTION',
                       'OBSERVED_PROJECTION', 'FINITE_CARRIER_BOUNDARY'}
ENDPOINT_ATTRIBUTES = [('start_height', 'end_height'),
    ('start_hole_height', 'end_hole_height'), ('start_base_edge', 'end_base_edge')]


def _validate_curve(child):
    ends = np.asarray(child['endpoints_mm'], np.float64)
    length = float(child['length_mm'])
    if ends.shape != (2, 3) or not np.isfinite(ends).all() or not math.isfinite(length) or length <= 0:
        raise ValueError('A finite, positive, complete input curve is required')
    if child['kind'] == 'line':
        if child['is_closed'] or not math.isclose(np.linalg.norm(ends[1] - ends[0]), length, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError('Input line geometry and length disagree')
    elif child['kind'] == 'arc':
        arc = CircularArc.from_parameters(child['arc_parameters'])
        if (arc.is_closed != bool(child['is_closed'])
                or not math.isclose(arc.length_mm, length, rel_tol=1e-10, abs_tol=1e-6)
                or np.max(np.linalg.norm(arc.point_at([0., length]) - ends, axis=1)) > 1e-6):
            raise ValueError('Input finite arc parameters and boundaries disagree')
    else:
        raise ValueError('Unsupported finite curve type')


def geometry_fingerprint(child):
    fields = ['kind', 'endpoints_mm', 'length_mm', 'is_closed']
    if child['kind'] == 'arc':
        fields.append('arc_parameters')
    value = {key: child[key] for key in fields}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def directed_children(children):
    """Reverse complete curves and endpoint-owned metadata without sign flips."""
    result = []
    for index, original in enumerate(children):
        _validate_curve(original)
        for orientation in (0, 1):
            child = deepcopy(original)
            child.update(input_candidate_index=index, input_orientation=orientation)
            if orientation:
                child['endpoints_mm'] = original['endpoints_mm'][::-1]
                if 'points' in original:
                    child['points'] = original['points'][::-1]
                if 'sample_positions_mm' in original:
                    child['sample_positions_mm'] = [original['length_mm'] - q
                                                    for q in original['sample_positions_mm'][::-1]]
                if child['kind'] == 'arc':
                    arc = CircularArc.from_parameters(original['arc_parameters'])
                    end = arc.point_at(arc.length_mm)
                    child['arc_parameters'] = CircularArc(arc.center_mm, arc.radius_mm,
                        arc.plane_normal, (end - arc.center_mm) / arc.radius_mm,
                        -arc.sweep_rad, arc.is_closed).to_parameters()
                for name, shape in [('raw_export_frame_axes', (2, 3, 3)),
                                    ('frame_origins_mm', (2, 3))]:
                    if name in original:
                        value = np.asarray(original[name], np.float64)
                        if value.shape != shape or not np.isfinite(value).all():
                            raise ValueError('Malformed endpoint frame cannot be silently reversed')
                        child[name] = value[::-1].tolist()
                for name in ('boundary_event_ids', 'boundary_refs', 'node_ids'):
                    if name in original:
                        if len(original[name]) != 2:
                            raise ValueError('An endpoint-owned identity must have two entries')
                        child[name] = original[name][::-1]
                if 'endpoint_attributes' in original:
                    for left, right in ENDPOINT_ATTRIBUTES:
                        attrs = child['endpoint_attributes']
                        a, b = attrs.pop(left, None), attrs.pop(right, None)
                        if b is not None:
                            attrs[left] = b
                        if a is not None:
                            attrs[right] = a
            _validate_curve(child)
            child['input_geometry_sha256'] = geometry_fingerprint(child)
            result.append(child)
    return result


def _shared_event(left, right, events):
    a = left.get('boundary_event_ids', [None, None])[1]
    b = right.get('boundary_event_ids', [None, None])[0]
    if a is None or a != b or a not in events:
        return None
    event = events[a]
    sha = event.get('input_receipt_sha256', '')
    point = np.asarray(event.get('xyz_mm'), np.float64)
    if (event.get('construction') not in EVENT_CONSTRUCTIONS or len(sha) != 64
            or any(c not in '0123456789abcdef' for c in sha)
            or point.shape != (3,) or not np.isfinite(point).all()):
        raise ValueError('A shared boundary lacks an authenticated finite input event')
    if (not np.array_equal(left['endpoints_mm'][1], point)
            or not np.array_equal(right['endpoints_mm'][0], point)):
        return None  # A close pair does not justify moving either input curve.
    return a


def legal_relations(children, edges, events):
    legal = np.ones((len(edges), 3), bool)
    for row, (left, right) in enumerate(edges):
        a, b = children[int(left)], children[int(right)]
        legal[row, 1] = _shared_event(a, b, events) is not None
        legal[row, 2] = np.linalg.norm(np.asarray(a['endpoints_mm'][1]) - b['endpoints_mm'][0]) > 1e-6
    return legal


class DirectedLinkModel(nn.Module):
    """Actual trainable runtime link logits plus the original 24-column prior."""
    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(nn.Linear(282, 128), nn.SiLU(), nn.Linear(128, 3))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features, left_hidden, right_hidden, legal, prior_probabilities):
        count = len(features)
        if (features.shape != (count, 24) or left_hidden.shape != (count, 128)
                or right_hidden.shape != (count, 128) or legal.shape != (count, 3)
                or legal.dtype != torch.bool or prior_probabilities.shape != (count, 3)
                or not legal[:, 0].all()):
            raise ValueError('Directed learned relation input contract changed')
        if (not torch.isfinite(features).all() or not torch.isfinite(left_hidden).all()
                or not torch.isfinite(right_hidden).all() or not torch.isfinite(prior_probabilities).all()
                or torch.any(prior_probabilities < 0)
                or not torch.allclose(prior_probabilities.sum(-1), torch.ones(count, device=features.device), atol=1e-5)):
            raise ValueError('Relation evidence and original prior must be finite probabilities')
        evidence = torch.cat((torch.asinh(features), left_hidden, right_hidden,
                              legal[:, 1:].to(features.dtype)), -1)
        logits = torch.log(prior_probabilities.clamp_min(1e-12)) + self.residual(evidence)
        return logits.masked_fill(~legal, -torch.inf)


def assemble_fixed_parent(states, chain, gap_kinds, events, closed=False):
    selected = [states[int(i)] for i in chain]
    if not selected or len(gap_kinds) != len(chain) - 1:
        raise ValueError('A complete nonempty ordered chain and all gaps are required')
    originals = [int(s['input_candidate_index']) for s in selected]
    if len(set(originals)) != len(originals):
        raise ValueError('One physical input candidate cannot appear in both orientations')
    if len(chain) > 1 and any(c['is_closed'] for c in selected):
        raise ValueError('A closed-circle child cannot belong to a longer parent')
    boundary_ids = []
    for a, b, kind in zip(selected[:-1], selected[1:], gap_kinds):
        if kind == 1:
            event = _shared_event(a, b, events)
            if event is None:
                raise ValueError('Shared link would require unsupported geometry movement')
            boundary_ids.append(event)
        elif kind == 2:
            if np.linalg.norm(np.asarray(a['endpoints_mm'][1]) - b['endpoints_mm'][0]) <= 1e-6:
                raise ValueError('A positive gap requires distinct actual boundaries')
            boundary_ids.append(None)
        else:
            raise ValueError('Unknown directed relation class')
    closing = None
    if closed:
        if len(chain) == 1:
            if not selected[0]['is_closed']:
                raise ValueError('A singleton cannot invent a closed curve')
        else:
            if any(kind != 1 for kind in gap_kinds):
                raise ValueError('A continuous closed parent cannot contain a positive gap')
            closing = _shared_event(selected[-1], selected[0], events)
            if closing is None:
                raise ValueError('A closed chain requires its authenticated common event')
    for child in selected:
        _validate_curve(child)
        if geometry_fingerprint(child) != child['input_geometry_sha256']:
            raise ValueError('Directed input geometry changed before assembly')
    return dict(child_count=len(chain), is_closed=bool(closed or selected[0]['is_closed']),
        gaps=['SHARED' if kind == 1 else 'POSITIVE_GAP' for kind in gap_kinds],
        segments=deepcopy(selected), input_children=originals,
        directed_input_states=list(map(int, chain)), shared_boundary_event_ids=boundary_ids,
        closing_boundary_event_id=closing, joint_boundary_max_displacement_mm=0.,
        arc_refits=0, frame_origin_translations=0, geometry_preserved_exactly=True)


def build_supported_parent_pool(children, child_hidden, priors, model, events,
                                neighbors=16, beam_width=2, max_chain_children=64,
                                max_parents=500_000, max_link_edges=4_000_000):
    if min(neighbors, beam_width, max_chain_children, max_parents, max_link_edges) < 1:
        raise ValueError('Positive declared directed search resources required')
    if model.training:
        raise ValueError('Runtime learned relation scoring requires evaluation mode')
    hidden = np.asarray(child_hidden, np.float32)
    if hidden.shape != (len(children), 128) or not np.isfinite(hidden).all():
        raise ValueError('Every complete input candidate requires its frozen embedding')
    states = directed_children(children)
    edges = proposal_edges(states, neighbors, max_link_edges)
    if len(edges):
        edges = edges[edges[:, 0] // 2 != edges[:, 1] // 2]
    features, probabilities = priors.link_probabilities(states, edges)
    legal = legal_relations(states, edges, events)
    device = next(model.parameters()).device
    values = []
    with torch.inference_mode():
        for start in range(0, len(edges), 8192):
            pair = edges[start:start + 8192]
            tensor = lambda a: torch.as_tensor(a, device=device)
            values.append(model(tensor(features[start:start + 8192]),
                tensor(hidden[pair[:, 0] // 2]), tensor(hidden[pair[:, 1] // 2]),
                tensor(legal[start:start + 8192]),
                tensor(probabilities[start:start + 8192].astype(np.float32))).cpu().numpy())
    logits = np.concatenate(values) if values else np.empty((0, 3), np.float32)
    if not np.isfinite(logits[legal]).all():
        raise ValueError('Actual runtime learned relation logits are nonfinite')
    successors = [[] for _ in states]
    for (left, right), scores, allowed in zip(edges, logits, legal):
        for kind in (1, 2):
            if allowed[kind] and scores[kind] > scores[0]:
                successors[int(left)].append((int(right), kind, float(scores[kind] - scores[0])))
    parents = [assemble_fixed_parent(states, [i], [], events) for i in range(len(states))]
    if len(parents) > max_parents:
        raise ValueError('Both complete orientations exceed declared parent resources')
    seen = {(tuple([i]), tuple(), bool(s['is_closed'])) for i, s in enumerate(states)}
    audit = Counter()
    for first in range(len(states)):
        beam = [([first], [], 0.)]
        for _ in range(1, max_chain_children):
            extended = []
            for chain, gaps, score in beam:
                used = {i // 2 for i in chain}
                for right, kind, gain in successors[chain[-1]]:
                    if right == first and len(chain) > 1 and kind == 1:
                        if any(value != 1 for value in gaps):
                            audit['closed_paths_with_positive_gaps_rejected'] += 1
                            continue
                        signature = (tuple(chain), tuple(gaps), True)
                        if signature not in seen:
                            parents.append(assemble_fixed_parent(states, chain, gaps, events, True))
                            seen.add(signature)
                    elif right // 2 not in used:
                        extended.append((chain + [right], gaps + [kind], score + gain))
            if len(parents) > max_parents:
                raise ValueError('Complete directed parent pool exceeds declared resources')
            if not extended:
                break
            extended.sort(key=lambda row: (-row[2], row[0], row[1]))
            audit['beam_extensions_not_retained'] += max(0, len(extended) - beam_width)
            beam = extended[:beam_width]
            for chain, gaps, _ in beam:
                signature = (tuple(chain), tuple(gaps), False)
                if signature not in seen:
                    parents.append(assemble_fixed_parent(states, chain, gaps, events))
                    seen.add(signature)
            if len(parents) > max_parents:
                raise ValueError('Complete directed parent pool exceeds declared resources')
        else:
            if any(successors[chain[-1]] for chain, _, _ in beam):
                audit['chain_limit_prefixes_with_successors'] += len(beam)
    return dict(parents=parents, directed_children=states, link_pairs=edges,
        link_features=features, link_prior_probabilities=probabilities,
        learned_link_logits=logits, link_legal=legal,
        audit=dict(schema=SCHEMA, parents=len(parents), physical_candidates=len(children),
            directed_states=len(states), edges=len(edges), learned_link_scoring_used=True,
            original_run446_priors_sha256=priors.checkpoint_sha256,
            original_prior_is_soft=True, endpoint_displacement_mm=0., arc_refits=0,
            frame_origin_translations=0, input_event_count=len(events),
            search_limits=dict(neighbors=neighbors, beam_width=beam_width,
                               max_chain_children=max_chain_children, max_parents=max_parents,
                               max_link_edges=max_link_edges), search_capacity_diagnostics=dict(audit),
            input_only=True, GT_or_STEP_reads=0, full_contract_pass=False))
