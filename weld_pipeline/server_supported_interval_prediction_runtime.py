"""Input-only complete finite interval predictions with actual weld fields.

Visits every declared input profile and its entire one-mm query grid. Scores
all input endpoint events, retains all valid top-eight Cartesian alternatives,
and constructs output geometry from double-precision input event coordinates.
Reference files, names, assignments, and labels are not accepted by this API.
"""
from collections import Counter
import math

import numpy as np
import torch

from tools.train_full_weld_reference_recall import TrainingTensors
from .server_finite_carrier_grid import finite_carrier_grid
from .server_full_prediction_runtime import predicted_fields, proposal_token
from .server_supported_directed_parent_pool import directed_children, build_supported_parent_pool
from .server_supported_interval_events import FiniteInputIntervalEvents, interval_child, shared_event_records
from .server_visual_curve_context import VisualCurveContext

SCHEMA = 'COMPLETE_SUPPORTED_INTERVAL_INPUT_ONLY_OUTPUT_V1'


class SupportedIntervalPredictionRuntime:
    def __init__(self, model, mean, std, *, query_batch=16, maximum_event_query_product=65_536,
                 maximum_children=500_000, reference_probability=.5, whole_pair_beam=64):
        if model.training:
            raise ValueError('Runtime requires a frozen evaluation-mode model')
        if min(query_batch, maximum_event_query_product, maximum_children) < 1:
            raise ValueError('Positive explicit runtime resources required')
        if not 0 <= reference_probability <= 1:
            raise ValueError('A declared reference admission probability is required')
        if not 1 <= whole_pair_beam <= 64:
            raise ValueError('Declared whole-interval beam must be between one and 64')
        self.whole_pair_beam = int(whole_pair_beam)
        self.model = model
        self.device = next(model.parameters()).device
        self.converter = TrainingTensors(mean, std, self.device)
        self.query_batch = query_batch
        self.maximum_event_query_product = maximum_event_query_product
        self.maximum_children = maximum_children
        self.reference_probability = float(reference_probability)

    def encode_query(self, inp, context, pid, positions, tokens=None):
        raw, coordinate = context.features(inp, pid, positions)
        t = self.converter.tensor
        if pid < inp.original_profile_count:
            batch = dict(features=self.converter.features(inp.features_at(pid, positions)),
                profile_ids=t(np.full(len(positions), pid, np.int64)),
                token_sequences={pid: t(inp.tokens(pid)[0]) if tokens is None else tokens},
                normalize_tokens=self.converter.normalize_tokens, geometry=t(coordinate['geometry']),
                query_local_features=t(raw[:, :72]))
            hidden = self.model.encode_native(batch, t(raw[:, 72:132]))
        else:
            hidden = self.model.encode_visual(t(raw))
        return hidden, coordinate

    @torch.inference_mode()
    def collect(self, inp, xyz, input_receipt_sha256, progress=None):
        if self.model.training: raise ValueError('Do not collect predictions while fitting model weights')
        context = VisualCurveContext(xyz)
        producer = FiniteInputIntervalEvents(xyz, input_receipt_sha256)
        children, vectors, origins, bases, priorities = [], [], [], [], []
        index_by_interval = {}; events = {}; audit = Counter()
        t = self.converter.tensor
        double = lambda x: torch.as_tensor(x, dtype=torch.float64, device=self.device)
        for pid, profile in enumerate(inp.profiles):
            length = float(inp.lengths[pid]); native = pid < inp.original_profile_count
            if native:
                a, b = inp.original.offsets[pid:pid + 2]
                spans = inp.original.spans[a:b, :2]
            else:
                spans = np.array([[0., length]], np.float64)
            bank = producer.build(profile, spans)
            positions = bank['positions_mm']; queries = finite_carrier_grid(length, [])['positions_mm']
            if native:
                encoded_events = self.model.encode_events(native_features=self.converter.features(inp.features_at(pid, positions)))
                tokens = t(inp.tokens(pid)[0])
            else:
                event_context, _ = context.features(inp, pid, positions)
                encoded_events = self.model.encode_events(visual_context=t(event_context), provenance=t(bank['provenance']))
                tokens = None
            provenance = t(bank['provenance'])[None]
            event_positions = double(positions)[None]
            batch_size = max(1, min(self.query_batch, self.maximum_event_query_product // len(positions)))
            for start in range(0, len(queries), batch_size):
                query = queries[start:start + batch_size]; count = len(query)
                hidden, coordinate = self.encode_query(inp, context, pid, query, tokens)
                query_mm = double(query); lengths = double(np.full(count, length))
                ep = event_positions.expand(count, -1)
                ev = encoded_events[None].expand(count, -1, -1)
                prov = provenance.expand(count, -1, -1)
                legal = torch.stack((ep <= query_mm[:, None], ep >= query_mm[:, None]), -1)
                event_chunk = max(1, self.maximum_event_query_product // count)
                logits = torch.cat([self.model.score_events(hidden[:, :128], ev[:, i:i + event_chunk],
                    prov[:, i:i + event_chunk], ep[:, i:i + event_chunk], query_mm, lengths)
                    for i in range(0, len(positions), event_chunk)], dim=1)
                scores, pairs = self.model.score_pairs(hidden[:, :128], ev, prov, ep, query_mm, lengths, logits, legal)
                retained = pairs['valid'].clone()
                if self.whole_pair_beam < scores.shape[1]:
                    order = scores.argsort(dim=1, descending=True, stable=True)[:, :self.whole_pair_beam]
                    retained.zero_(); retained.scatter_(1, order, pairs['valid'].gather(1, order))
                audit['whole_pairs_outside_declared_search_beam'] += int((pairs['valid'] & ~retained).sum())
                rows, alternatives = torch.where(retained)
                audit['queries_scored'] += count; audit['endpoint_events_scored'] += count * len(positions)
                audit['whole_pairs_scored'] += int(pairs['valid'].sum())
                audit['queries_without_legal_whole_pair'] += int((~pairs['valid'].any(1)).sum())
                if not len(rows): continue
                left = pairs['left_ids'][rows, alternatives]; right = pairs['right_ids'][rows, alternatives]
                bounds = torch.stack((ep[rows, left], ep[rows, right]), -1) / length
                geometry = t(coordinate['geometry'])[rows]
                prediction = self.model.predict_interval(hidden[rows], encoded_events[left], encoded_events[right],
                    geometry, bounds, query_mm[rows] / length, native=native)
                probabilities = prediction['reference_score'].sigmoid()
                # This is learned reference-policy admission, not material occupancy.
                admitted = torch.where(probabilities >= self.reference_probability)[0]
                audit['pairs_below_reference_admission'] += len(rows) - len(admitted)
                # No-query rows have no legal pair; avoid an all-infinity softmax.
                pair_logp = scores[pairs['valid'].any(1)].log_softmax(1)
                all_logp = torch.zeros_like(scores)
                all_logp[pairs['valid'].any(1)] = pair_logp
                priority = probabilities.clamp_min(1e-12).log() + all_logp[rows, alternatives]
                outputs = {k: v[admitted].cpu().numpy() for k, v in prediction.items()}
                row_ids = rows[admitted].cpu().numpy(); li = left[admitted].cpu().numpy(); ri = right[admitted].cpu().numpy()
                reference = probabilities[admitted].cpu().numpy(); rank = priority[admitted].cpu().numpy()
                for j, (row, l, r) in enumerate(zip(row_ids, li, ri)):
                    output = {k: v[j] for k, v in outputs.items()}
                    weld_type = 'horizontal' if output['weld_type'].argmax() == 0 else 'vertical'
                    signature = (pid, int(l), int(r), weld_type)
                    existing = index_by_interval.get(signature)
                    if existing is not None and rank[j] <= priorities[existing]:
                        audit['exact_repeated_interval_weaker_query'] += 1
                        continue
                    try:
                        child = interval_child(bank, l, r)
                        child = directed_children([child])[int(output['selected_direction_reversed'])]
                    except (ValueError, OverflowError) as exc:
                        # Distinct arclengths can round to the same world point.
                        # Such a hypothesis is invalid geometry, not a reason to
                        # drop the rest of the complete input or fabricate ends.
                        audit['invalid_finite_pair:' + str(exc)] += 1
                        continue
                    child.update(profile=pid, branch='carrier', query_position_mm=float(query[row]),
                        weld_type=weld_type, reference_score=float(reference[j]), predicted_side='UNKNOWN',
                        face_pairs=[profile.get('face_indices', [])[:2]], side_known=False, physical_sides=['UNKNOWN'],
                        endpoint_pair_log_probability=float(all_logp[rows[admitted[j]], alternatives[admitted[j]]]),
                        exact_input_geometry=True)
                    segment = dict(child, points=child['endpoints_mm'])
                    child.update(predicted_fields(output, segment, coordinate['bases'][row]))
                    vector = proposal_token(segment, output['hidden'], coordinate['points_mm'][row],
                                            coordinate['bases'][row], float(reference[j]))
                    events.update(shared_event_records(bank, [l, r]))
                    if existing is None:
                        if len(children) >= self.maximum_children:
                            raise RuntimeError('Complete input interval proposals exceed declared capacity; no partial result accepted')
                        index_by_interval[signature] = len(children)
                        children.append(child); vectors.append(vector); priorities.append(float(rank[j]))
                        origins.append(coordinate['points_mm'][row]); bases.append(coordinate['bases'][row])
                    else:
                        children[existing] = child; vectors[existing] = vector; priorities[existing] = float(rank[j])
                        origins[existing] = coordinate['points_mm'][row]; bases[existing] = coordinate['bases'][row]
                        audit['exact_repeated_interval_replaced_by_stronger_query'] += 1
            audit['input_profiles_visited'] += 1; audit['input_events'] += len(positions)
            if progress: progress(pid + 1, len(inp.profiles), dict(audit), len(children))
        if audit['input_profiles_visited'] != len(inp.profiles):
            raise ValueError('A complete input profile disappeared')
        return dict(children=children, child_tokens=np.asarray(vectors, np.float32).reshape(-1, 148),
            query_xyz=np.asarray(origins, np.float64).reshape(-1, 3),
            query_bases=np.asarray(bases, np.float64).reshape(-1, 3, 3), shared_events=events,
            audit=dict(schema=SCHEMA, counts=dict(audit), input_profiles=len(inp.profiles),
                original_profiles=inp.original_profile_count, raw_points=len(xyz),
                reference_probability=self.reference_probability, endpoint_topk=8, whole_alternatives_scored=64,
                retained_whole_pair_search_beam=self.whole_pair_beam,
                geometry_from_original_double_precision_events=True, all_input_profiles_and_queries_visited=True,
                exact_duplicate_policy='Same input profile/event pair/type keeps highest reference-plus-pair score; stable first tie',
                independent_orientation_states_pending_parent_pool=True, maximum_children=self.maximum_children,
                GT_or_STEP_reads=0, full_extraction_accuracy_claimed=False))


def supported_parent_data(proposals, model, priors, **search):
    """Same runtime learned relation head as training, with nullable set tokens."""
    from .server_parent_prediction_runtime import pack_tokens
    from .server_lazy_parent_competition import LazyParentCompetition
    pool = build_supported_parent_pool(proposals['children'], proposals['child_tokens'][:, :128],
        priors, model.directed_link, proposals['shared_events'], **search)
    pc = [p['input_children'] for p in pool['parents']]
    return dict(pool=pool, pc=pc, **pack_tokens(pool, proposals),
                lazy_competition=LazyParentCompetition(pool['parents'], priors),
                pool_receipt_sha256=None)
