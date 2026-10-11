"""Full-object parent-set objective with latent complete TRAIN witnesses.

Hungarian and backprop use the same positive costs. Unmatched queries have a
separately normalized NO_OBJECT loss; matching subtracts the displaced loss.
No candidate pruning, duplicate positive targets, or GT-count inference.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from weld_pipeline.sjg_parent_set_runtime_1081 import child_geometry


class _AllLogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries, keys, chunk):
        scale = queries.shape[-1] ** -.5
        normalizer = queries.new_full(queries.shape[:-1], -float('inf'))
        for first in range(0, len(keys), chunk):
            logits = queries @ keys[first:first+chunk].T * scale
            normalizer = torch.logaddexp(normalizer, logits.logsumexp(-1))
        ctx.save_for_backward(queries, keys, normalizer)
        ctx.chunk = chunk
        return normalizer

    @staticmethod
    def backward(ctx, gradient):
        queries, keys, normalizer = ctx.saved_tensors
        scale = queries.shape[-1] ** -.5
        dq = torch.zeros_like(queries); dk = torch.empty_like(keys)
        for first in range(0, len(keys), ctx.chunk):
            kb = keys[first:first+ctx.chunk]
            weighted = torch.exp(queries @ kb.T * scale-normalizer[..., None])*gradient[..., None]
            dq.add_(weighted @ kb * scale)
            dk[first:first+ctx.chunk] = weighted.T @ queries * scale
        return dq, dk, None


def all_logsumexp(queries, keys, chunk=2048):
    if queries.ndim != 2 or keys.ndim != 2 or not len(keys):
        raise ValueError('Nonempty full categorical denominator required')
    return _AllLogSumExp.apply(queries, keys, int(chunk))


def assignment(positive, negative, common_denominator=False):
    """Minimize mean positive + mean unmatched negative, without changing scope."""
    q, p = positive.shape
    if p==0 and common_denominator:
        return np.empty(0,np.int64),np.empty(0,np.int64)
    if not 0 < p <= q:
        raise ValueError('Parent capacity exceeded or empty positive assignment')
    cost = positive/p
    if q > p:
        cost = cost-negative[:, None]/(p if common_denominator else q-p)
    rows, columns = linear_sum_assignment(cost.detach().double().cpu().numpy())
    return rows, columns


def _binary(logits, truth):
    return F.softplus(logits)-logits*truth


class ParentTargets:
    """Precompute supervision constants after the input freeze; never inference."""
    def __init__(self, compiled, bank, device):
        self.compiled = compiled; self.device = device
        self.parents = compiled['parents']; self.children = []
        self.profile_features = {}; self.transitions = []
        for parent in self.parents:
            rows = []; transitions = []; previous_ends = None
            for ci, child in enumerate(parent['children']):
                alternatives = child['alternatives']; geometry = []; bounds = []; features = []
                oracle = []
                for a in alternatives:
                    pid = a['profile']; item = bank.get(pid); curve = item['curve']
                    if pid not in self.profile_features:
                        self.profile_features[pid] = torch.as_tensor(item['interval_features'], device=device)
                    edges = item['anchors_mm']; indices = np.asarray(a['intervals'])
                    bounds.append(np.stack((edges[indices], edges[indices+1]), -1))
                    features.append(item['interval_features'][indices])
                    if curve.kind == 'line':
                        geometry.append([0., curve.length, *curve.ends[0], *curve.direction, 0., 0., 0., 1., 1.])
                    else:
                        arc = curve.arc
                        # origin/axis1/axis2, radius, signed angular direction.
                        geometry.append([1., curve.length, *arc.center_mm, *arc.start_direction,
                                         *np.cross(arc.plane_normal, arc.start_direction),
                                         arc.radius_mm, math.copysign(1., arc.sweep_rad)])
                    points, _ = child_geometry(curve, a['lo'], a['hi'], a['forward'], a['closed'], a['wrap'])
                    oracle.append(points[[0, -1]])
                oracle = np.asarray(oracle)
                if ci:
                    distance = np.linalg.norm(previous_ends[:, 1, None]-oracle[None, :, 0], axis=-1)
                    gap = parent['gaps'][ci-1]
                    # A complete witness has one connection for each ordered child pair.
                    transition = (distance/3 if gap == 'SHARED' else
                                  np.where(distance > 0, 0., np.inf) if gap == 'POSITIVE_GAP' else np.zeros_like(distance))
                    transitions.append(torch.as_tensor(transition, dtype=torch.float32, device=device))
                previous_ends = oracle
                samples = np.asarray(child['loss_samples_mm'], np.float64)
                # Ordered endpoints and central control point. Shared/closure nodes
                # are owned once; runtime reconstructs lines/arcs from these nodes.
                target = samples[[0, len(samples)//2, -1]]
                geometry = np.asarray(geometry)
                point_mask = np.ones((len(alternatives), 3), dtype=np.float64)
                point_mask[:, 1] = geometry[:, 0]  # a line is fixed by its endpoints
                if ci and parent['gaps'][ci-1] == 'SHARED':point_mask[:, 0] = 0
                if parent['closed'] and ci == len(parent['children'])-1:point_mask[:, 2] = 0
                endpoint_mask = np.ones((len(alternatives), 2), np.float32)
                for ai, a in enumerate(alternatives):
                    if geometry[ai, 0] == 0:
                        start_head = 0 if a['forward'] else 1
                        if not point_mask[ai, 0]:endpoint_mask[ai, start_head] = 0
                        if not point_mask[ai, 2]:endpoint_mask[ai, 1-start_head] = 0
                tensor = lambda value, dtype=torch.float32: torch.as_tensor(np.asarray(value), dtype=dtype, device=device)
                rows.append(dict(alternatives=alternatives, profiles=tensor([a['profile'] for a in alternatives], torch.long),
                    bounds=tensor(bounds, torch.float64), features=tensor(features), geometry=tensor(geometry, torch.float64),
                    target=tensor(target, torch.float64), point_mask=tensor(point_mask, torch.float64),
                    endpoint_mask=tensor(endpoint_mask), fractions=tensor([a['fractions'] for a in alternatives]),
                    forward=tensor([a['forward'] for a in alternatives]), closed=tensor([a['closed'] for a in alternatives]),
                    wrap=tensor([a['wrap'] for a in alternatives]), weld_type=child['weld_type'],
                    connection=None if ci == 0 or parent['gaps'][ci-1] == 'UNKNOWN' else
                    {'SHARED':1, 'POSITIVE_GAP':2}[parent['gaps'][ci-1]]))
            self.children.append(rows); self.transitions.append(transitions)

    @staticmethod
    def control_points(row, fraction):
        # Float64 retains local-mm detail even for very short anchor intervals.
        q = row['bounds'][None, ..., 0]+fraction[:, None].double()*(row['bounds'][..., 1]-row['bounds'][..., 0])[None]
        lo, hi = q.unbind(-1); g = row['geometry']; length = g[:, 1]
        closed = row['closed'].bool(); wrap = row['wrap'].bool(); forward = row['forward'].bool()
        start = torch.where(wrap, hi, lo)
        extent = torch.where(closed, length, torch.where(wrap, length-hi+lo, hi-lo))
        start = torch.where(forward, start, start+extent)
        signed = torch.where(forward, extent, -extent)
        positions = start[..., None]+signed[..., None]*q.new_tensor([0., .5, 1.])
        origin = g[:, 2:5]; axis1 = g[:, 5:8]; axis2 = g[:, 8:11]; radius = g[:, 11]
        line = origin[None, :, None]+positions[..., None]*axis1[None, :, None]
        angle = positions*g[None, :, None, 12]/radius[None, :, None]
        arc = origin[None, :, None]+radius[None, :, None, None]*(angle.cos()[..., None]*axis1[None, :, None]+angle.sin()[..., None]*axis2[None, :, None])
        return torch.where(g[None, :, None, 0:1].bool(), arc, line)


class ParentSetObjective:
    def __init__(self, model, targets, common_denominator=False):
        self.model = model; self.targets = targets
        self.common_denominator=bool(common_denominator)

    def _child(self, result, pi, ci, query_ids, alternative_ids, normalizers, endpoint_cache):
        row = self.targets.children[pi][ci]
        # A smaller view is used only for differentiating the exact already chosen
        # latent witness, not to reduce any input/categorical denominator.
        view = {k:(v[alternative_ids] if torch.is_tensor(v) and k not in ('target',) else v)
                for k,v in row.items()}
        profiles = view['profiles']; query = result['profile_query'][query_ids, ci]
        cost = normalizers[query_ids, ci, None]-query @ result['profile_key'][profiles].T/math.sqrt(self.model.width)
        eq = result['endpoint_query'][query_ids, ci]
        target_keys = self.model.endpoint_key(view['features'])
        logits = torch.einsum('qed,aed->qae', eq, target_keys)/math.sqrt(self.model.width)
        denoms = []; query_tuple = tuple(query_ids.tolist())
        for pid in profiles.tolist():
            cache_key = (ci, pid, query_tuple)
            if cache_key not in endpoint_cache:
                keys = self.model.endpoint_key(self.targets.profile_features[pid])
                endpoint_cache[cache_key] = all_logsumexp(eq.reshape(-1, self.model.width), keys).reshape(len(eq), 2)
            denoms.append(endpoint_cache[cache_key])
        endpoint_nll = torch.stack(denoms, 1)-logits
        mask = view['endpoint_mask']; divisor = mask.sum(-1).clamp_min(1)
        cost = cost+(endpoint_nll*mask).sum(-1)/divisor
        fraction = result['fraction'][query_ids, ci]
        cost = cost+((fraction[:, None]-view['fractions']).abs()*mask).sum(-1)/divisor
        points = self.targets.control_points(view, fraction)
        error = torch.linalg.vector_norm(points-view['target'], dim=-1)/3.
        pm = view['point_mask']; geometry = (error*pm).sum(-1)/pm.sum(-1).clamp_min(1)
        cost = cost+geometry.float()
        for head in ('direction','closed','wrap'):
            truth = view['forward' if head == 'direction' else head]
            cost = cost+_binary(result[head][query_ids, ci, None], truth)
        wt = row['weld_type']
        if wt >= 0:cost = cost-result['type'][query_ids, ci].log_softmax(-1)[:, wt, None]
        if row['connection'] is not None:
            cost = cost-result['connection'][query_ids, ci].log_softmax(-1)[:, row['connection'], None]
        return cost

    def _parent_base(self, result, pi, queries):
        p = self.targets.parents[pi]
        return (F.softplus(-result['existence'][queries])-
                result['count'][queries].log_softmax(-1)[:, p['child_count']-1]+
                _binary(result['parent_closed'][queries], float(p['closed'])))

    @torch.no_grad()
    def match(self, result):
        count = len(result['existence']); queries = torch.arange(count, device=result['existence'].device)
        normalizers = all_logsumexp(result['profile_query'].flatten(0, 1), result['profile_key']).reshape(count, -1)
        costs = []; witnesses = []; cache = {}
        for pi, parent in enumerate(self.targets.parents):
            back = []; running = None
            for ci, row in enumerate(self.targets.children[pi]):
                own = self._child(result, pi, ci, queries, slice(None), normalizers, cache)
                if running is None:running = own
                else:
                    combined = running[:, :, None]+self.targets.transitions[pi][ci-1][None]
                    previous, index = combined.min(1)
                    running = previous+own; back.append(index)
            value, index = running.min(1); selected = [index]
            for previous in back[::-1]:
                index = previous.gather(1, index[:, None]).squeeze(1); selected.append(index)
            selected.reverse()
            witnesses.append(torch.stack(selected, -1))
            costs.append(self._parent_base(result, pi, queries)+value/parent['child_count'])
        positive = torch.stack(costs, 1); negative = F.softplus(result['existence'])
        if not torch.isfinite(positive).all():raise RuntimeError('No finite complete witness for a parent')
        rows, columns = assignment(positive, negative, self.common_denominator)
        return dict(rows=rows, columns=columns, witnesses=witnesses, positive=positive, negative=negative)

    def __call__(self, result):
        if not self.targets.parents and self.common_denominator:
            loss=F.softplus(result['existence']).sum()
            return loss,dict(positive=0.,negative=float(loss.detach()),parents=0,
                unmatched_queries=len(result['existence']),matching_backprop_difference=0.,
                candidate_profiles=len(result['profile_key']),original_G=self.targets.compiled['G'])
        match = self.match(result)
        total_queries = len(result['existence']); parent_count = len(self.targets.parents)
        # Compute gradients only for matched positive queries, with all input
        # profile keys still in every categorical normalization.
        rows = torch.as_tensor(match['rows'], dtype=torch.long, device=result['existence'].device)
        normalizers = result['existence'].new_zeros((total_queries, self.model.child_slots))
        normalizers = normalizers.index_copy(0, rows, all_logsumexp(
            result['profile_query'][rows].flatten(0, 1), result['profile_key']).reshape(len(rows), -1))
        positives = []; cache = {}
        for qi, pi in zip(match['rows'], match['columns']):
            queries = rows.new_tensor([int(qi)]); parent = self.targets.parents[int(pi)]
            selected = match['witnesses'][int(pi)][int(qi)].tolist(); child_cost = result['existence'].new_zeros(())
            for ci, ai in enumerate(selected):
                child_cost = child_cost+self._child(result, int(pi), ci, queries, [ai], normalizers, cache).squeeze()
                if ci:child_cost = child_cost+self.targets.transitions[int(pi)][ci-1][selected[ci-1], ai]
            positives.append(self._parent_base(result, int(pi), queries).squeeze()+child_cost/parent['child_count'])
        positive = torch.stack(positives).mean()
        mask = torch.ones(total_queries, dtype=torch.bool, device=rows.device); mask[rows] = False
        negative = F.softplus(result['existence'][mask]).mean() if mask.any() else positive*0
        if self.common_denominator and mask.any():
            negative=F.softplus(result['existence'][mask]).sum()/max(parent_count,1)
        reference = match['positive'][rows, torch.as_tensor(match['columns'],device=rows.device)].mean()
        mismatch = float((positive.detach()-reference).abs())
        if mismatch > 2e-4*max(1., float(reference.abs())):
            raise RuntimeError(f'Matching/backprop objective mismatch: {mismatch}')
        return positive+negative, dict(positive=float(positive.detach()), negative=float(negative.detach()),
            parents=parent_count, unmatched_queries=int(mask.sum()), matching_backprop_difference=mismatch,
            candidate_profiles=len(result['profile_key']), original_G=self.targets.compiled['G'])


def normalization_controls():
    torch.manual_seed(1031); records=[]
    for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 2e-5)):
        q=torch.randn(7,8,dtype=dtype); k=torch.randn(37,8,dtype=dtype)
        rq=q.clone().requires_grad_();rk=k.clone().requires_grad_()
        dense=(rq@rk.T/math.sqrt(8)).logsumexp(-1)
        expected=torch.autograd.grad(dense.sum(),(rq,rk))
        for chunk in (1,11,64):
            sq=q.clone().requires_grad_();sk=k.clone().requires_grad_()
            value=all_logsumexp(sq,sk,chunk);actual=torch.autograd.grad(value.sum(),(sq,sk))
            error=max(float((a-b).abs().max()) for a,b in zip(actual,expected))
            assert torch.allclose(value,dense,atol=tolerance,rtol=tolerance) and error<tolerance
            records.append(dict(dtype=str(dtype),chunk=chunk,gradient_error=error))
    # The displaced background cost changes which query should be assigned.
    positive=torch.tensor([[1.],[2.],[3.]],dtype=torch.float64)
    negative=torch.tensor([0.,10.,0.],dtype=torch.float64)
    rows,cols=assignment(positive,negative)
    brute=[float(positive[i,0]+negative[[j for j in range(3) if j!=i]].mean()) for i in range(3)]
    assert rows.tolist()==[int(np.argmin(brute))]==[1]
    return dict(normalizers=records,displaced_background_bruteforce=True,optimizer_updates=0)
