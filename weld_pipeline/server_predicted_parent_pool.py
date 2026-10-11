"""Predicted ordered parents with genuine frozen run446 link/duplicate priors.

No reference graph or teacher order is accepted. Shared cuts are joint runtime
parameters computed from predictions, followed by analytic reconstruction.
Rejected hypotheses remain diagnostics; an empty selection has zero outputs.
"""
from collections import Counter
import math
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial import cKDTree

from .contract_curve_geometry import CircularArc, sample_arc
from .server_full_prediction_runtime import compact_segment, sampled_segment
from .sjg_additive_parent_precision_1254 import pair_features
from .sjg_child_parent_supervision_1255 import singleton, link_features
from .sjg_local_runtime_904 import read, digest
from .sjg_parent_set_features_1160 import ParentRelations
from .world_set_runtime_285 import line_segment


def child_metadata(child):
    return dict(face_pairs=child['face_pairs'], side_known=child['side_known'],
                physical_sides=child['physical_sides'], probability_mean=child['reference_score'],
                length_mm=child['length_mm'], line_fraction=float(child['kind'] == 'line'))


def directed_state(children):
    ends = np.array([c['endpoints_mm'] for c in children], dtype=float).reshape(-1, 2, 3)
    tangents = []
    for child, points in zip(children, ends):
        if child['kind'] == 'line':
            direction = (points[1] - points[0]) / child['length_mm']
            tangents.append([direction, direction])
        else:
            arc = CircularArc.from_parameters(child['arc_parameters'])
            tangents.append(math.copysign(1., arc.sweep_rad) * np.cross(arc.plane_normal,
                                                               (points - arc.center_mm) / arc.radius_mm))
    return dict(ends=ends, tangent=np.asarray(tangents).reshape(-1, 2, 3),
                length=np.array([c['length_mm'] for c in children]),
                line=np.array([c['kind'] == 'line' for c in children]),
                horizontal=np.array([c['weld_type'] == 'horizontal' for c in children]),
                closed=np.array([c['is_closed'] for c in children]),
                faces=[{f for pair in c['face_pairs'] for f in pair if f >= 0} for c in children],
                face_pairs=[{tuple(p) for p in c['face_pairs'] if len(p) == 2 and min(p) >= 0} for c in children],
                known=np.array([c['side_known'] for c in children]))


class FrozenRun446Priors:
    def __init__(self, root):
        root = Path(root)
        record = read(root / 'experiment_records/run446_frozen_baseline_reference_20261011.json')
        item = next(row for row in record['checkpoints'] if row['path'].endswith('/run446/models.pkl'))
        path = root / item['path']
        if digest(path) != item['sha256']:
            raise ValueError('Original best checkpoint fingerprint changed')
        with path.open('rb') as stream:
            models = pickle.load(stream)['models']
        self.link, self.duplicate = models['link'], models['duplicate']
        if self.link.n_features_in_ != 24 or self.duplicate.n_features_in_ != 32:
            raise ValueError('Original best relation feature contracts changed')
        self.checkpoint_sha256 = item['sha256']
        # The 526-column validity head is intentionally not called with an
        # invented/padded feature vector. New parent utility has its own head.

    def link_probabilities(self, children, pairs):
        features = link_features(directed_state(children), pairs)
        probabilities = np.zeros((len(features), 3))
        if len(features):
            probabilities[:, self.link.classes_.astype(int)] = self.link.predict_proba(features)
        return features, probabilities

    def duplicate_probabilities(self, parents, pairs):
        features = []
        for a, b in pairs:
            selected = [parents[int(a)], parents[int(b)]]
            decoded = [dict(p, segments=[sampled_segment(s) for s in p['segments']]) for p in selected]
            metadata = []
            for parent in selected:
                children = parent['segments']
                length = sum(c['length_mm'] for c in children)
                metadata.append(dict(face_pairs=[pair for c in children for pair in c['face_pairs']],
                                     side_known=all(c['side_known'] for c in children),
                                     physical_sides=[side for c in children for side in c['physical_sides']],
                                     probability_mean=sum(c['length_mm'] * c['reference_score'] for c in children) / length,
                                     length_mm=length,
                                     line_fraction=sum(c['kind'] == 'line' for c in children) / len(children)))
            context = ParentRelations(decoded, metadata)
            features.append(pair_features(context, 0, 1))
        features = np.asarray(features, np.float32).reshape(-1, 32)
        if not len(features):
            return features, np.empty(0)
        probability = self.duplicate.predict_proba(features)
        column = list(self.duplicate.classes_).index(1)
        return features, probability[:, column]


def through_three_points(start, middle, end, original):
    a, b, c = np.asarray([start, middle, end], float)
    u, v = b - a, c - a
    cross = np.cross(u, v)
    norm2 = float(cross @ cross)
    if norm2 == 0 or not math.isfinite(norm2):
        raise ValueError('Joint arc boundaries cannot define a finite three-point circle')
    center = a + ((u @ u) * np.cross(v, cross) + (v @ v) * np.cross(cross, u)) / (2 * norm2)
    normal = cross / math.sqrt(norm2)
    if normal @ original.plane_normal < 0:
        normal = -normal
    radius = float(np.linalg.norm(a - center))
    first = (a - center) / radius
    second = np.cross(normal, first)
    angle = lambda p: math.atan2(float((p - center) @ second), float((p - center) @ first)) % (2 * math.pi)
    finish, midpoint = angle(c), angle(b)
    sweep = finish if midpoint <= finish else finish - 2 * math.pi
    return CircularArc(center, radius, normal, first, sweep, False)


def assemble_parent(children, chain, gap_kinds, closed=False):
    selected = [children[int(i)] for i in chain]
    if len(gap_kinds) != len(chain) - 1:
        raise ValueError('Ordered parent gap count differs from its predicted children')
    if len(chain) == 1:
        return dict(child_count=1, is_closed=bool(selected[0]['is_closed']), gaps=[],
                    segments=selected, input_children=list(map(int, chain)),
                    joint_boundary_max_displacement_mm=0.)
    if any(c['is_closed'] for c in selected):
        raise ValueError('A full-circle child cannot occur inside a multi-child parent')
    original = np.array([c['endpoints_mm'] for c in selected], float)
    ends = original.copy()
    for i, kind in enumerate(gap_kinds):
        if kind == 1:
            cut = (original[i, 1] + original[i + 1, 0]) / 2
            ends[i, 1] = ends[i + 1, 0] = cut
        elif kind == 2:
            if np.array_equal(ends[i, 1], ends[i + 1, 0]):
                raise ValueError('Predicted positive gap has coincident boundaries')
        else:
            raise ValueError('Unknown predicted gap class')
    if closed:
        cut = (original[-1, 1] + original[0, 0]) / 2
        ends[-1, 1] = ends[0, 0] = cut
    segments = []
    for child, endpoint in zip(selected, ends):
        if child['kind'] == 'line':
            represented = line_segment(*endpoint)
        else:
            arc = CircularArc.from_parameters(child['arc_parameters'])
            middle = arc.point_at(np.array([arc.length_mm / 2]))[0]
            represented = sample_arc(through_three_points(*[endpoint[0], middle, endpoint[1]], arc))
            # One shared prediction parameter, independent of roundoff in the
            # analytic definition. This never copies a reference cut point.
            represented['points'][0], represented['points'][-1] = endpoint.tolist()
        compact = dict(child, **compact_segment(represented))
        compact['frame_origins_mm'] = (np.asarray(child['frame_origins_mm']) + endpoint - child['endpoints_mm']).tolist()
        compact['joint_geometry_frame_consistency_certified'] = False
        segments.append(compact)
    return dict(child_count=len(chain), is_closed=bool(closed),
                gaps=['SHARED' if k == 1 else 'POSITIVE_GAP' for k in gap_kinds],
                segments=segments, input_children=list(map(int, chain)),
                joint_boundary_max_displacement_mm=float(np.linalg.norm(ends - original, axis=2).max()))


def proposal_edges(children, neighbors=16, max_edges=4_000_000):
    if len(children) < 2:
        return np.empty((0, 2), np.int64)
    state = directed_state(children)
    tree = cKDTree(state['ends'][:, 0])
    distances, _ = tree.query(state['ends'][:, 1], k=min(neighbors + 1, len(children)))
    distances = np.asarray(distances).reshape(len(children), -1)
    edges = []
    for left, distance in enumerate(distances):
        for right in sorted(tree.query_ball_point(state['ends'][left, 1], float(distance.max()) + 1e-7)):
            if left != right and not children[left]['is_closed'] and not children[right]['is_closed']:
                edges.append((left, right))
        if len(edges) > max_edges:
            raise ValueError('Directed parent proposal graph exceeds explicit resources')
    return np.asarray(edges, np.int64).reshape(-1, 2)


def build_parent_pool(children, priors, neighbors=16, beam_width=2,
                      max_chain_children=64, max_parents=500_000):
    if min(neighbors, beam_width, max_chain_children, max_parents) < 1:
        raise ValueError('Positive explicit parent search resources required')
    edges = proposal_edges(children, neighbors)
    link_x, probabilities = priors.link_probabilities(children, edges)
    successors = [[] for _ in children]
    for (left, right), scores in zip(edges, probabilities):
        kind = 1 + int(np.argmax(scores[1:]))
        if scores[kind] > scores[0]:
            gain = math.log(max(float(scores[kind]), 1e-12)) - math.log(max(float(scores[0]), 1e-12))
            successors[int(left)].append((int(right), kind, gain))
    parents = [assemble_parent(children, [i], []) for i in range(len(children))]
    seen = {(tuple([i]), tuple(), bool(c['is_closed'])) for i, c in enumerate(children)}
    diagnostics = Counter()
    for first in range(len(children)):
        beam = [([first], [], 0.)]
        for _ in range(1, max_chain_children):
            extended = []
            for chain, gaps, score in beam:
                for right, kind, gain in successors[chain[-1]]:
                    if right == first and len(chain) > 1 and kind == 1:
                        signature = (tuple(chain), tuple(gaps), True)
                        if signature not in seen:
                            try:
                                parents.append(assemble_parent(children, chain, gaps, True))
                                seen.add(signature)
                            except ValueError as exc:
                                diagnostics[str(exc)] += 1
                        continue
                    if right not in chain:
                        extended.append((chain + [right], gaps + [kind], score + gain))
            if not extended:
                break
            extended.sort(key=lambda row: (-row[2], row[0], row[1]))
            beam = extended[:beam_width]
            for chain, gaps, _ in beam:
                signature = (tuple(chain), tuple(gaps), False)
                if signature in seen:
                    continue
                try:
                    parents.append(assemble_parent(children, chain, gaps))
                    seen.add(signature)
                except ValueError as exc:
                    diagnostics[str(exc)] += 1
            if len(parents) > max_parents:
                raise ValueError('Complete ordered-parent pool exceeds explicit resource limit')
        else:
            diagnostics['DECLARED_CHAIN_CAPACITY_REACHED'] += len(beam)
    return dict(parents=parents, link_pairs=edges, link_features=link_x, link_probabilities=probabilities,
                audit=dict(parents=len(parents), child_hypotheses=len(children), edges=len(edges),
                           original_run446_priors_sha256=priors.checkpoint_sha256,
                           prior_existence_score_distribution_changed=True,
                           unknown_physical_sides_retained=True,
                           rejected_hypotheses=dict(diagnostics), max_chain_children=max_chain_children,
                           GT_or_STEP_reads=0, teacher_parent_order_used=False))


def decoded_subset(parents, selected):
    actual = [dict(parents[int(i)], segments=[sampled_segment(s) for s in parents[int(i)]['segments']]) for i in selected]
    return dict(parents=actual, invalid_predictions=[],
                total_prediction_children=sum(p['child_count'] for p in actual),
                rejected_candidates_are_not_output_predictions=True, GT_or_STEP_reads=0,
                runtime_attributes_trained=True, full_contract_pass=False)


def parent_center(parent):
    centers, lengths = [], []
    for child in parent['segments']:
        if child['kind'] == 'line':
            center = np.mean(child['endpoints_mm'], axis=0)
        else:
            arc = CircularArc.from_parameters(child['arc_parameters'])
            theta = arc.sweep_rad
            center = (arc.center_mm if arc.is_closed else arc.center_mm + arc.radius_mm *
                       (np.sinc(theta / np.pi) * arc.start_direction +
                        (1 - np.cos(theta)) / theta * np.cross(arc.plane_normal, arc.start_direction)))
        centers.append(center)
        lengths.append(child['length_mm'])
    return np.average(centers, axis=0, weights=lengths)


def parent_competition_pairs(parents, neighbors=16, max_pairs=4_000_000):
    if len(parents) < 2:
        return np.empty((0, 2), np.int64)
    centers = np.array([parent_center(p) for p in parents])
    tree = cKDTree(centers)
    distances, _ = tree.query(centers, k=min(neighbors + 1, len(parents)))
    pairs = set()
    for a, distance in enumerate(np.asarray(distances).reshape(len(parents), -1)):
        for b in tree.query_ball_point(centers[a], float(distance.max()) + 1e-7):
            if a != b:
                pairs.add(tuple(sorted((a, b))))
        if len(pairs) > max_pairs:
            raise ValueError('Parent competition graph exceeds explicit resources')
    return np.asarray(sorted(pairs), np.int64).reshape(-1, 2)
