"""Complete TRAIN-view updates on shared finite event intervals.

Labels assign existing input events and supervise complete alternatives. They do
not create an event, feature, or runtime candidate. Endpoint losses see every
event; the pair loss uses the same bounded Cartesian proposals as inference.
"""
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F

from .contract_geometry_d3_409 import child_error
from .legacy_incremental_event_runtime_393 import reverse_segment
from .server_boundary_event_selector import masked_endpoint_event_loss
from .server_full_prediction_runtime import sampled_segment
from .server_reference_recall_weld_model import reference_responsibility_loss
from .server_stable_endpoint_pairs import pair_set_loss
from .server_supported_directed_parent_pool import directed_children, legal_relations
from .server_supported_interval_events import interval_child, shared_event_records
from .server_supported_interval_weld_model import known_field_losses
from .world_set_evaluation_285 import reference_segment


class CompleteIntervalView:
    """Bounded per-view input cache; model embeddings are recomputed each update."""
    def __init__(self, inp, context, sample, view, load_bank):
        self.inp, self.sample, self.view = inp, sample, view
        self.count = len(sample['profile_ids'])
        self.context = np.empty((self.count, 147), np.float32)
        self.tokens = {}
        self.banks, self.event_features = {}, {}
        self.native_features = {int(row): sample['native_features'][i]
                                for i, row in enumerate(sample['native_row_indices'])}
        self.children = [c for p in view.parents for c in p['children']]
        self.references = [reference_segment(dict(c, world_representation=c['representation']))
                           for c in self.children]
        self.closed = np.array([p['is_closed'] and len(p['children']) == 1
                               for p in view.parents for c in p['children']], bool)
        self.groups = []
        self.teacher_ids = np.empty((self.count, 2), np.int64)
        self.teacher_good = np.zeros(self.count, bool)
        self.endpoint_positive = {}
        self.match_cache = {}
        for pid in np.unique(sample['profile_ids']):
            pid = int(pid); rows = np.flatnonzero(sample['profile_ids'] == pid)
            q = sample['input_query_positions_mm'][rows]
            self.context[rows], _ = context.features(inp, pid, q)
            bank = load_bank(pid); self.banks[pid] = bank
            positions = bank['positions_mm']
            if pid < inp.original_profile_count:
                self.tokens[pid] = inp.tokens(pid)[0]
                self.event_features[pid] = inp.features_at(pid, positions)
            else:
                self.event_features[pid] = context.features(inp, pid, positions)[0]
            for query in np.unique(q):
                self.groups.append((pid, float(query), rows[q == query]))
            for row in rows:
                query = sample['input_query_positions_mm'][row]
                legal = np.column_stack((positions <= query, positions >= query))
                ends = np.asarray(self.references[row]['points'])[[0, -1]]
                if sample['targets']['direction'][row]: ends = ends[::-1]
                positive = np.linalg.norm(bank['xyz_mm'][:, None] - ends, axis=2) <= 3.
                if self.closed[row]:
                    positive[:] = False; positive[0, 0] = True; positive[-1, 1] = True
                positive &= legal
                if not sample['targets']['interval_known'][row]: positive[:] = False
                self.endpoint_positive[int(row)] = positive
                bounds = sample['targets']['interval_ratios'][row] * bank['length_mm']
                # Label-nearest existing events are teacher assignments only.
                left = np.flatnonzero(legal[:, 0]); right = np.flatnonzero(legal[:, 1])
                li = int(left[np.argmin(abs(positions[left] - bounds[0]))])
                ri = int(right[np.argmin(abs(positions[right] - bounds[1]))])
                if positions[li] >= positions[ri]: li, ri = 0, len(positions) - 1
                self.teacher_ids[row] = [li, ri]
                self.teacher_good[row] = self.matches(int(row), li, ri)
        if not np.allclose(self.context[:, 132:], sample['geometry'], atol=1e-6, rtol=1e-6):
            raise ValueError('Input query context and complete target coordinate basis disagree')

    def matches(self, row, left, right):
        key = (row, int(left), int(right))
        if key not in self.match_cache:
            good = False
            if self.sample['targets']['interval_known'][row] and left != right:
                pid = int(self.sample['profile_ids'][row]); bank = self.banks[pid]
                if bank['positions_mm'][left] < bank['positions_mm'][right]:
                    c = interval_child(bank, left, right); c['weld_type'] = self.children[row]['weld_type']
                    segment = sampled_segment(c)
                    if self.sample['targets']['direction'][row]:
                        segment = dict(segment, **reverse_segment(segment))
                    good = child_error(segment, self.children[row], self.references[row],
                                       bool(self.closed[row]), False) is not None
            self.match_cache[key] = good
        return self.match_cache[key]

    def encode(self, model, converter, rows):
        rows = np.asarray(rows, np.int64)
        native = self.sample['profile_ids'][rows] < self.inp.original_profile_count
        if not native.all() and native.any(): raise ValueError('Encode each genuine feature route separately')
        if native.all():
            pids = self.sample['profile_ids'][rows]
            batch = dict(features=converter.features(np.stack([self.native_features[int(r)] for r in rows])),
                profile_ids=converter.tensor(pids),
                token_sequences={int(p): converter.tensor(self.tokens[int(p)]) for p in np.unique(pids)},
                normalize_tokens=converter.normalize_tokens,
                geometry=converter.tensor(self.sample['geometry'][rows]),
                query_local_features=converter.tensor(self.context[rows, :72]))
            return model.encode_native(batch, converter.tensor(self.context[rows, 72:132]))
        return model.encode_visual(converter.tensor(self.context[rows]))

    def encode_events(self, model, converter, pid, indices=None):
        values = self.event_features[pid]
        provenance = self.banks[pid]['provenance']
        if indices is not None: values, provenance = values[indices], provenance[indices]
        if pid < self.inp.original_profile_count:
            return model.encode_events(native_features=converter.features(values))
        return model.encode_events(visual_context=converter.tensor(values), provenance=converter.tensor(provenance))

    def directed_targets(self, priors, prediction):
        children, events = [], {}
        types = prediction['weld_type'].detach().argmax(1).cpu().numpy()
        scores = prediction['reference_score'].detach().sigmoid().cpu().numpy()
        for row, pid in enumerate(self.sample['profile_ids']):
            bank = self.banks[int(pid)]; li, ri = self.teacher_ids[row]
            child = interval_child(bank, li, ri)
            profile = self.inp.profiles[int(pid)]
            child.update(weld_type='vertical' if types[row] else 'horizontal', reference_score=float(scores[row]),
                         face_pairs=[profile.get('face_indices', [])[:2]], side_known=False,
                         physical_sides=['UNKNOWN'])
            events.update(shared_event_records(bank, [li, ri]))
            children.append(directed_children([child])[int(self.sample['targets']['direction'][row])])
        pairs = np.column_stack((self.sample['link_left'], self.sample['link_right']))
        labels = np.zeros(len(pairs), np.int64)
        positive = {}
        for ids, parent in zip(self.sample['parent_children'], self.view.parents):
            for a, b, gap in zip(ids[:-1], ids[1:], parent['gaps']):
                positive[a, b] = {'SHARED': 1, 'POSITIVE_GAP': 2}[gap]
            if parent['is_closed'] and len(ids) > 1: positive[ids[-1], ids[0]] = 1
        for i, pair in enumerate(pairs): labels[i] = positive.get(tuple(pair), 0)
        features, prior = priors.link_probabilities(children, pairs)
        legal = legal_relations(children, pairs, events)
        known = (self.teacher_good[pairs[:, 0]] & self.teacher_good[pairs[:, 1]]
                 & legal[np.arange(len(pairs)), labels]) if len(pairs) else np.empty(0, bool)
        self.links = dict(pairs=pairs, labels=labels, known=known, features=features.astype(np.float32),
                          prior=prior.astype(np.float32), legal=legal)
        return dict(known=int(known.sum()), unavailable=int((~known).sum()),
            shared_unavailable=int(((labels == 1) & ~known).sum()),
            classes={str(i): int(((labels == i) & known).sum()) for i in range(3)})


def complete_view_step(model, converter, data, optimizer, loss_mass=1.):
    """One complete view, every child/known field retained; bounded group gradients."""
    model.train(); optimizer.zero_grad(set_to_none=True)
    sample = data.sample; count = data.count; stats = Counter(); totals = Counter()
    for pid, query, rows in data.groups:
        bank = data.banks[pid]; positions = bank['positions_mm']; length = bank['length_mm']
        hidden = data.encode(model, converter, rows[:1])[:, :128]
        events = data.encode_events(model, converter, pid)[None]
        provenance = converter.tensor(bank['provenance'])[None]
        pos = torch.as_tensor(positions, device=converter.device, dtype=torch.float64)[None]
        q = torch.tensor([query], device=converter.device, dtype=torch.float64)
        lengths = torch.tensor([length], device=converter.device, dtype=torch.float64)
        legal_np = np.column_stack((positions <= query, positions >= query))
        legal = converter.tensor(legal_np)[None]
        positives = np.any([data.endpoint_positive[int(r)] for r in rows], axis=0)
        logits = model.score_events(hidden, events, provenance, pos, q, lengths)
        endpoint_loss = masked_endpoint_event_loss(logits, legal, converter.tensor(positives)[None])
        scores, pairs = model.score_pairs(hidden, events, provenance, pos, q, lengths, logits, legal)
        li, ri = pairs['left_ids'][0].detach().cpu().numpy(), pairs['right_ids'][0].detach().cpu().numpy()
        valid = pairs['valid'][0].detach().cpu().numpy(); good = np.zeros(len(li), bool)
        for j in np.flatnonzero(valid):
            # Keep a whole owner alternative; never cross two owners' endpoints.
            good[j] = any(data.endpoint_positive[int(r)][li[j], 0]
                          and data.endpoint_positive[int(r)][ri[j], 1]
                          and data.matches(int(r), li[j], ri[j]) for r in rows)
        pair_loss = pair_set_loss(scores, pairs['valid'], converter.tensor(good)[None])
        weight = len(rows) / count
        objective = (endpoint_loss + pair_loss) * weight * loss_mass
        if not torch.isfinite(objective): raise ValueError('Nonfinite complete event objective')
        objective.backward()
        totals['endpoint_events'] += float(endpoint_loss.detach()) * weight
        totals['whole_pairs'] += float(pair_loss.detach()) * weight
        stats['query_groups'] += 1; stats['all_events_scored'] += len(positions)
        stats['groups_with_positive_endpoint'] += int(positives.any())
        stats['groups_with_positive_whole_pair'] += int(good.any())
        stats['rows_retained'] += len(rows)
    # Full parent/field pass recomputes queries with current weights. Teacher
    # endpoint assignments remain explicit and are not an extraction result.
    predictions, route_rows = [], []
    for native in (True, False):
        rows = np.flatnonzero((sample['profile_ids'] < data.inp.original_profile_count) == native)
        if not len(rows): continue
        hidden = data.encode(model, converter, rows)
        left, right, bounds = [], [], []
        for row in rows:
            pid = int(sample['profile_ids'][row]); ids = data.teacher_ids[row]
            ev = data.encode_events(model, converter, pid, ids)
            left.append(ev[0]); right.append(ev[1])
            bounds.append(data.banks[pid]['positions_mm'][ids] / data.banks[pid]['length_mm'])
        prediction = model.predict_interval(hidden, torch.stack(left), torch.stack(right),
            converter.tensor(sample['geometry'][rows]),
            torch.tensor(np.asarray(bounds), device=converter.device, dtype=torch.float64),
            torch.tensor(sample['query_ratios'][rows], device=converter.device, dtype=torch.float64), native=native,
            reverse=converter.tensor(sample['targets']['direction'][rows]).bool())
        predictions.append(prediction); route_rows.extend(rows.tolist())
    order = converter.tensor(np.argsort(route_rows))
    prediction = {key: torch.cat([p[key] for p in predictions])[order] for key in predictions[0]}
    targets = {key: converter.tensor(value) for key, value in sample['targets'].items()}
    losses = known_field_losses(prediction, targets)
    losses.update(reference_responsibility_loss(prediction))
    losses['parent_closure'] = F.cross_entropy(model.parent_logits(prediction['hidden'], sample['parent_children']),
                                               converter.tensor(sample['parent_closed']))
    if len(sample['link_left']):
        logits = model.reference_link_logits(prediction['hidden'], converter.tensor(sample['link_left']),
            converter.tensor(sample['link_right']), converter.tensor(sample['link_relative']))
        losses['reference_links'] = F.cross_entropy(logits, converter.tensor(sample['link_labels']))
    data.link_audit = data.directed_targets(data.priors, prediction)
    links = data.links; known = links['known']
    if known.any():
        pairs = links['pairs'][known]
        logits = model.directed_link(converter.tensor(links['features'][known]),
            prediction['hidden'][converter.tensor(pairs[:, 0])], prediction['hidden'][converter.tensor(pairs[:, 1])],
            converter.tensor(links['legal'][known]), converter.tensor(links['prior'][known]))
        labels = converter.tensor(links['labels'][known]); error = F.cross_entropy(logits, labels, reduction='none')
        losses['runtime_directed_links'] = torch.stack([error[labels == i].mean() for i in range(3)
                                                       if (labels == i).any()]).mean()
    weights = dict(attribute_state=.5, attribute_mm=.5, frame_axes=.5, frame_offsets=2.,
                   parent_closure=.5, reference_links=.5, reference_prior_correction=.001)
    objective = sum(v * weights.get(k, 1.) for k, v in losses.items())
    if not torch.isfinite(objective): raise ValueError('Nonfinite complete fields/parents objective')
    (objective * loss_mass).backward()
    stats['directed_link_gradient_nonzero'] = int(any(p.grad is not None and bool(torch.any(p.grad != 0))
                                                   for p in model.directed_link.parameters()))
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
    optimizer.step()
    totals.update({k: float(v.detach()) for k, v in losses.items()})
    if stats['rows_retained'] != count: raise ValueError('A complete reference row disappeared')
    return dict(loss=float(objective.detach()) + totals['endpoint_events'] + totals['whole_pairs'],
                components=dict(totals), gradient_norm=float(grad), directed_links=data.link_audit,
                coverage=dict(stats), teacher_intervals_D3=int(data.teacher_good.sum()),
                children=count, actual_complete_output_accuracy_claimed=False)
