"""Input-only query proposals with carrier and learned free-geometry arms.

No label reader, source name or supervision owner enters this runtime. All
finite profiles are visited. Bad hypotheses are diagnostics, not mandatory
false predictions in a subsequently empty selected output.
"""
from collections import Counter
import hashlib
import json
import math

import numpy as np
import torch

from .contract_curve_geometry import CircularArc, sample_arc
from .legacy_incremental_event_runtime_393 import reverse_segment
from .server_full_training_samples import query_frame, geometry_descriptor
from .world_set_runtime_285 import line_segment


def compact_segment(segment):
    result = {name: value for name, value in segment.items()
              if name not in ('points', 'sample_positions_mm', 'sampling_audit', 'sampling_verification')}
    result['endpoints_mm'] = np.asarray(segment['points'])[[0, -1]].tolist()
    return result


def sampled_segment(compact):
    if compact['kind'] == 'line':
        segment = line_segment(*compact['endpoints_mm'])
    else:
        segment = sample_arc(CircularArc.from_parameters(compact['arc_parameters']))
    segment.update({k: v for k, v in compact.items() if k not in ('points', 'sample_positions_mm')})
    return segment


def segment_key(segment):
    fields = ('kind', 'endpoints_mm', 'arc_parameters', 'weld_type', 'profile', 'branch')
    value = {k: segment[k] for k in fields if k in segment}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()


def carrier_segment(curve, bounds, reverse):
    lo, hi = np.asarray(bounds, float) * curve.length
    if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi <= curve.length:
        raise ValueError('Predicted interval is outside its finite input carrier')
    represented = curve.segment(float(lo), float(hi))
    segment = (line_segment(*represented['endpoints_mm']) if curve.kind == 'line'
               else sample_arc(CircularArc.from_parameters(represented['arc_parameters'])))
    if reverse:
        segment.update(reverse_segment(segment))
    return segment


def free_segment(output, point, basis):
    if int(np.argmax(output['kind'])) == 0:
        ends = point + np.asarray(output['line_points'], float) @ basis.T * 1000
        return line_segment(*ends)
    shape = np.asarray(output['arc_shape'], float)
    if not np.isfinite(shape).all():
        raise ValueError('Nonfinite learned circle parameters')
    center = point + shape[:3] @ basis.T * 1000
    # Resource violations fail explicitly; no arbitrary radius clipping.
    radius = math.exp(float(shape[3])) * 1000
    normal = shape[4:7] @ basis.T
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm == 0:
        raise ValueError('Predicted circle has no oriented plane')
    normal /= normal_norm
    start = shape[7:10] @ basis.T
    start -= normal * (start @ normal)
    norm = float(np.linalg.norm(start))
    if norm == 0:
        raise ValueError('Predicted circle has no in-plane start direction')
    start /= norm
    closed = bool(int(np.argmax(output['closed'])))
    sweep = float(shape[10]) * math.pi
    if closed:
        if sweep == 0:
            raise ValueError('Closed circle has no predicted traversal sign')
        sweep = math.copysign(2 * math.pi, sweep)
    return sample_arc(CircularArc(center, radius, normal, start, sweep, closed))


def predicted_fields(output, segment, basis):
    # Frame axes retain the exported row-axis convention and learned signs.
    ends = np.asarray(segment['points'])[[0, -1]]
    frames = np.asarray(output['frame_axes'], float) @ basis.T
    origins = ends + np.asarray(output['frame_offsets'], float) @ basis.T * 1000
    scores = np.asarray(output['attribute_states'], float)
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)
    codes = scores.argmax(-1)
    names = ['front_wall', 'back_wall', 'start_height', 'end_height',
             'start_hole_height', 'end_hole_height', 'start_base_edge', 'end_base_edge']
    encodings = ['EXACT_MM', 'DISTANCE_GT_50', 'NO_HOLE', 'NO_BLOCKER_OR_DISTANCE_GT_50']
    attributes = {}
    for i, name in enumerate(names):
        applicable = segment['weld_type'] == 'horizontal' or name == 'back_wall'
        attributes[name] = dict(state=encodings[int(codes[i])] if applicable else 'NOT_REQUIRED',
                                value_mm=float(output['attribute_mm'][i]) if applicable and codes[i] == 0 else None,
                                confidence=float(scores[i, codes[i]]) if applicable else None,
                                source='FULL_TRAIN_PROCESS_HEAD', calibrated=False)
    return dict(endpoint_attributes=attributes, raw_export_frame_axes=frames.tolist(),
                frame_origins_mm=origins.tolist(), frame_business_semantics_certified=False,
                missing_plate_normals_not_replaced_by_circle_plane_normal=True)


def proposal_token(segment, hidden, query_point, basis, reference_score):
    compact = compact_segment(segment)
    points = np.asarray(compact['endpoints_mm'], float)
    if segment['kind'] == 'line':
        tangent = (points[1] - points[0]) / segment['length_mm']
        tangents = np.stack((tangent, tangent))
        curvature = 0.
    else:
        arc = CircularArc.from_parameters(segment['arc_parameters'])
        tangents = math.copysign(1., arc.sweep_rad) * np.cross(arc.plane_normal,
                                             (points - arc.center_mm) / arc.radius_mm)
        curvature = 1 / arc.radius_mm
    extra = np.r_[float(segment['kind'] == 'arc'), float(segment['weld_type'] == 'vertical'),
                  float(segment['is_closed']), math.log1p(segment['length_mm']),
                  ((points - query_point) @ basis / 1000).ravel(), (tangents @ basis).ravel(),
                  curvature * 100, float(segment['branch'] == 'carrier'),
                  float(segment['branch'] == 'free'), reference_score]
    if extra.shape != (20,) or not np.isfinite(extra).all():
        raise ValueError('Predicted child token geometry changed')
    return np.r_[hidden, extra].astype(np.float32)


class CompleteQueryProposals:
    def __init__(self, model, mean, std, device='cuda', mode='fixed_and_grid',
                 grid_candidates_per_fragment=2, batch_size=2048, max_children=500_000,
                 fixed_query_fractions=(.125, .375, .625, .875)):
        if mode not in ('fragment4', 'grid', 'fixed_and_grid'):
            raise ValueError('Unknown complete input query mode')
        if min(grid_candidates_per_fragment, batch_size, max_children) < 1:
            raise ValueError('Positive explicit proposal resources required')
        self.model = model.to(device).eval()
        self.device = device
        self.mean = torch.as_tensor(mean, device=device, dtype=torch.float32)
        self.std = torch.as_tensor(std, device=device, dtype=torch.float32)
        self.mode, self.grid_candidates = mode, grid_candidates_per_fragment
        self.batch_size, self.max_children = batch_size, max_children
        fractions = np.asarray(fixed_query_fractions, float)
        if (fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all()
                or np.any(fractions < 0) or np.any(fractions > 1)
                or len(np.unique(fractions)) != len(fractions)):
            raise ValueError('Declared fixed input query fractions must be unique within [0,1]')
        self.fixed_query_fractions = fractions

    def normal_tokens(self, raw):
        return torch.cat((torch.clamp((raw[:, :456] - self.mean[:456]) / self.std[:456], -20, 20),
                          raw[:, 456:458], raw[:, 458:459] / 10), 1)

    @torch.inference_mode()
    def score_queries(self, inp, pid, positions, tokens):
        positions = np.asarray(positions, float)
        frames = [query_frame(inp, pid, float(q)) for q in positions]
        curve = inp.curve(pid)
        geometry = np.array([geometry_descriptor(curve, p, b, known) for p, b, known in frames])
        raw = inp.features_at(pid, positions)
        tensor = lambda x: torch.tensor(x, device=self.device, dtype=torch.float32)
        hidden = self.model.encode_queries(torch.clamp((tensor(raw) - self.mean) / self.std, -20, 20),
                                          torch.full((len(raw),), pid, device=self.device, dtype=torch.long),
                                          {pid: tokens}, self.normal_tokens, tensor(geometry))
        output = self.model.predict(hidden, tensor(raw[:, 456]))
        result = {name: value.detach().cpu().numpy() for name, value in output.items()}
        result['reference_probability'] = torch.sigmoid(output['reference_score']).cpu().numpy()
        return result, frames

    def build(self, inp, progress=None):
        children, vectors, origins, bases = [], [], [], []
        diagnostics, signatures = Counter(), set()
        visited, queries_scored, branches_scored = 0, 0, Counter()
        for pid, profile in enumerate(inp.profiles):
            begin, end = map(int, inp.offsets[pid:pid + 2])
            if begin == end:
                diagnostics['EMPTY_INPUT_PROFILE_RETAINED_IN_AUDIT'] += 1
                continue
            curve = inp.curve(pid)
            tokens = torch.tensor(inp.tokens(pid)[0], device=self.device, dtype=torch.float32)
            candidates = []
            if self.mode != 'grid':
                spans = inp.spans[begin:end, :2]
                candidates.extend((spans[:, 0, None] + self.fixed_query_fractions
                                  * (spans[:, 1] - spans[:, 0])[:, None]).ravel().tolist())
            if self.mode != 'fragment4':
                selected = {i: [] for i in range(end - begin)}
                for chunk in inp.grid_chunks(pid, chunk_size=self.batch_size):
                    result, _ = self.score_queries(inp, pid, chunk['positions_mm'], tokens)
                    queries_scored += len(chunk['positions_mm'])
                    local = np.minimum(np.searchsorted(inp.spans[begin:end, 1], chunk['positions_mm'], side='right'), end - begin - 1)
                    for i in np.unique(local):
                        indices = np.flatnonzero(local == i)
                        old = selected[int(i)]
                        old.extend((float(result['reference_probability'][j]), float(chunk['positions_mm'][j])) for j in indices)
                        # Stable score ties use the original increasing input coordinate.
                        old.sort(key=lambda pair: (-pair[0], pair[1]))
                        selected[int(i)] = old[:self.grid_candidates]
                candidates.extend(q for values in selected.values() for _, q in values)
            candidates = np.unique(candidates)
            for start in range(0, len(candidates), self.batch_size):
                positions = candidates[start:start + self.batch_size]
                result, frames = self.score_queries(inp, pid, positions, tokens)
                queries_scored += len(positions)
                for row, (q, (point, basis, _)) in enumerate(zip(positions, frames)):
                    output = {name: values[row] for name, values in result.items()}
                    for branch in ('carrier', 'free'):
                        try:
                            segment = (carrier_segment(curve, output['interval_ratios'], bool(output['direction'].argmax()))
                                       if branch == 'carrier' else free_segment(output, point, basis))
                            segment.update(profile=pid, branch=branch, query_position_mm=float(q),
                                           weld_type='horizontal' if output['weld_type'].argmax() == 0 else 'vertical',
                                           predicted_side='UNKNOWN', reference_score=float(output['reference_probability']))
                            signature = segment_key(compact_segment(segment))
                            if signature in signatures:
                                diagnostics['EXACT_REPEATED_INPUT_HYPOTHESIS'] += 1
                                continue
                            if len(children) >= self.max_children:
                                raise RuntimeError('Complete proposal pool exceeds explicit resource limit; refuse partial output')
                            segment.update(predicted_fields(output, segment, basis))
                            compact = compact_segment(segment)
                            compact.update(face_pairs=[profile.get('face_indices', [])[:2]],
                                           side_known=False, physical_sides=['UNKNOWN'])
                            children.append(compact)
                            vectors.append(proposal_token(segment, output['hidden'], point, basis,
                                                          float(output['reference_probability'])))
                            origins.append(point)
                            bases.append(basis)
                            signatures.add(signature)
                            branches_scored[branch] += 1
                        except (ValueError, OverflowError, FloatingPointError) as exc:
                            diagnostics[f'{branch}:{type(exc).__name__}:{str(exc)}'] += 1
            visited += 1
            if progress and pid % 128 == 0:
                progress(pid, len(inp.profiles), queries_scored, len(children))
        return dict(children=children, child_tokens=np.asarray(vectors, np.float32).reshape(-1, 148),
                    query_xyz=np.asarray(origins, np.float64).reshape(-1, 3),
                    query_bases=np.asarray(bases, np.float64).reshape(-1, 3, 3),
                    audit=dict(input_profiles=len(inp.profiles), nonempty_profiles_visited=visited,
                               queries_scored=queries_scored, query_mode=self.mode,
                               fixed_query_fractions=self.fixed_query_fractions.tolist(),
                               candidates=len(children), branches=dict(branches_scored),
                               rejected_hypotheses=dict(diagnostics), source_identity_feature_columns=0,
                               GT_or_STEP_reads=0, no_model_admission_gate=True,
                               rejected_hypotheses_are_not_selected_predictions=True,
                               full_contract_pass=False))
