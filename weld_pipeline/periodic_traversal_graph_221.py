"""Explicit periodic traversal preparation; no learned scoring or MILP.

Query records and local regression lifts remain literal. Physical phase is a
periodic coordinate, while direction/wrap/parent progress are separate. This
new schema cannot be consumed by the frozen 214/216 pair-only interfaces.
"""
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .primitive_incidence_209 import (
    ArcPrimitive, PolylinePrimitive, RuntimePrimitiveInput, PrimitiveError,
    ROLES, CHILD_ROLE_PAIRS, parse_primitive, canonical_json, array_sha,
)

VERSION = 'periodic_explicit_anchor_traversal_221_v2'
FG_VERSION = 'periodic_piecewise_linear_exact_physical_moments_221_v1'
PHASE_RULE = 'literal_float64_remainder_exact_period_alias_canonical_zero_no_raw_edit_v2'
EXPECTED_209_SHA = '666eceba1f6762d35ae6d872c456e43b7842dd40931e666a84d1406f3a50ee98'


def _seal(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _guard():
    if sha256(Path(__file__).with_name('primitive_incidence_209.py').read_bytes()).hexdigest() != EXPECTED_209_SHA:
        raise PrimitiveError('Frozen geometry foundation changed')


def _phase(value, period):
    value = np.asarray(value, np.float64)
    if not np.isfinite(value).all() or not math.isfinite(period) or period <= 0.:
        raise PrimitiveError('Finite lift and positive period required')
    result = np.remainder(value, period)
    # A sub-ULP negative lift can round its remainder to L. L and 0 are the
    # same periodic point; only this exact alias is represented canonically.
    # The literal local lift and actual raw output remain separate and intact.
    return np.where(result == period, 0., result)


@dataclass(frozen=True)
class TraversalScope:
    max_turns: int
    declaration: str
    business_scope_reviewed: bool = False

    def __post_init__(self):
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int) or self.max_turns < 1:
            raise PrimitiveError('Explicit finite positive turn budget required')
        if not isinstance(self.declaration, str) or not self.declaration:
            raise PrimitiveError('Explicit engineering/business scope declaration required')
        if not isinstance(self.business_scope_reviewed, bool):
            raise PrimitiveError('Explicit business scope review flag required')


class PeriodicFG:
    """Exact primitives of a periodic piecewise-linear evidence function.

All q/FG records are kept, including the distinct logical endpoint record.
An inconsistent endpoint alias refuses numerical FG features; no averaging.
"""
    def __setattr__(self, name, value):
        if getattr(self, '_locked', False):
            raise PrimitiveError('Periodic FG evidence is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, q, fg, period):
        q, fg = np.asarray(q, np.float64), np.asarray(fg, np.float64)
        if (q.ndim != 1 or len(q) < 2 or fg.shape != q.shape or not np.isfinite(q).all()
                or not np.isfinite(fg).all() or np.any(np.diff(q) <= 0.)
                or q[0] != 0. or q[-1] != period or np.any(fg < 0.) or np.any(fg > 1.)):
            raise PrimitiveError('Complete actual q and finite FG probabilities required')
        self.q, self.fg, self.period = _seal(q), _seal(fg), float(period)
        self.status = 'VERIFIED_CONTINUOUS_PERIODIC_FIELD' if fg[0] == fg[-1] else 'UNVERIFIED_ENDPOINT_ALIAS_FG'
        widths = np.diff(q)
        self.slopes = _seal(np.diff(fg) / widths)
        self.integrals = _seal(np.r_[0., np.cumsum(widths * (fg[:-1] + fg[1:]) / 2.)])
        self.square_integrals = _seal(np.r_[0., np.cumsum(widths *
            (fg[:-1]**2 + fg[:-1]*fg[1:] + fg[1:]**2) / 3.)])
        self._locked = True

    def primitive(self, x, square=False):
        if self.status != 'VERIFIED_CONTINUOUS_PERIODIC_FIELD':
            raise PrimitiveError(self.status + ': all raw endpoint records retained; no zero-fill')
        if not math.isfinite(x):
            raise PrimitiveError('Finite periodic FG argument required')
        turn = math.floor(x / self.period)
        phase = float(_phase(x, self.period))
        i = min(int(np.searchsorted(self.q, phase, side='right')) - 1, len(self.q)-2)
        offset, value, slope = phase-self.q[i], self.fg[i], self.slopes[i]
        prefix = self.square_integrals if square else self.integrals
        local = (value*value*offset + value*slope*offset**2 + slope*slope*offset**3/3.) if square else value*offset + slope*offset**2/2.
        return float(turn*prefix[-1] + prefix[i] + local)

    def moments(self, phase, direction, length):
        if direction not in (-1, 1) or not math.isfinite(length) or length <= 0.:
            raise PrimitiveError('Positive physical traversal and explicit direction required')
        end = phase + direction*length
        integral = direction * (self.primitive(end) - self.primitive(phase))
        square = direction * (self.primitive(end, True) - self.primitive(phase, True))
        mean = integral/length
        variance = square/length-mean*mean
        if variance < -1e-10:
            raise PrimitiveError('Unrepresentable periodic FG variance')
        return dict(mean=mean, std=math.sqrt(max(variance, 0.)), integral_mm=integral,
            square_integral_mm=square, version=FG_VERSION, actual_physical_interval=True)

    def fingerprint(self):
        return sha256(canonical_json(dict(version=FG_VERSION, period=self.period,
            actual_q=array_sha(self.q), all_FG_records=array_sha(self.fg), status=self.status,
            slopes=array_sha(self.slopes), integrals=array_sha(self.integrals),
            square_integrals=array_sha(self.square_integrals))).encode()).hexdigest()


def segment_traversal(primitive, phase, direction, length, kind, max_samples=1_000_001):
    """A seam is internal sampling arithmetic, never a new child or cut."""
    if not isinstance(primitive, (ArcPrimitive, PolylinePrimitive)) or not primitive.is_closed:
        raise PrimitiveError('Explicit closed arc/polyline required')
    period = primitive.length_mm
    if (direction not in (-1, 1) or not math.isfinite(phase) or not 0. <= phase < period
            or not math.isfinite(length) or length <= 0.):
        raise PrimitiveError('Finite phase, explicit direction and positive traversal required')
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 2:
        raise PrimitiveError('Explicit sample resource bound required')
    if kind not in ('partial', 'partial_with_turns', 'full_turn', 'multiple_turns'):
        raise PrimitiveError('Invalid traversal kind; zero span cannot become full turn')
    if kind == 'full_turn' and length != period:
        raise PrimitiveError('Full-turn semantics differ from physical length')
    if kind == 'multiple_turns' and (length <= period or length/period != round(length/period)):
        raise PrimitiveError('Multiple-turn semantics differ from physical length')
    if isinstance(primitive, ArcPrimitive):
        radius = primitive.arc.radius_mm
        angle_limit = min(1./radius, 2.*math.acos(1.-min(.1/radius, 2.)), math.pi/2.)
        count = max(1, int(math.ceil(length/(radius*angle_limit))))
        if count+1 > max_samples:
            raise PrimitiveError('EXPORT_SAMPLE_LIMIT: selected child retained as invalid')
        travel = np.linspace(0., length, count+1)
        points = primitive.point_at(_phase(phase+direction*travel, period))
        step = np.diff(travel)
        chord_errors = radius*(1.-np.cos(step/(2.*radius)))
        expected_chords = 2.*radius*np.sin(step/(2.*radius))
        extra = dict(signed_sweep_rad=math.copysign(1., primitive.arc.sweep_rad)*direction*length/radius,
            primitive_arc_parameters=primitive.arc.to_parameters())
        corners = []
    else:
        end = phase+direction*length
        lo, hi = min(phase, end), max(phase, end)
        corner_refs = {}
        for turn in range(math.floor(lo/period)-1, math.ceil(hi/period)+1):
            for j, q in enumerate(primitive.knot_positions_mm[:-1]):
                lifted = float(q+turn*period)
                if lo < lifted < hi:
                    corner_refs[lifted] = primitive.knots_mm[j]
        corners = sorted(corner_refs, reverse=direction == -1)
        breaks = [phase, *corners, end]
        lifted = []
        for left, right in zip(breaks[:-1], breaks[1:]):
            count = max(1, int(math.ceil(abs(right-left))))
            if len(lifted)+count+1 > max_samples:
                raise PrimitiveError('EXPORT_SAMPLE_LIMIT: selected child retained as invalid')
            lifted.extend(np.linspace(left, right, count+1)[:-1].tolist())
        lifted.append(end)
        lifted = np.asarray(lifted)
        travel = direction*(lifted-phase)
        points = primitive.point_at(_phase(lifted, period))
        # Use explicit original corner references, without smoothing/refitting.
        for q, xyz in corner_refs.items():
            points[lifted == q] = xyz
        step = np.diff(travel)
        expected_chords, chord_errors = step, np.zeros(len(step))
        extra = dict(ordered_original_corner_lifts_mm=corners,
            every_traversed_corner_retained=True)
    if kind in ('full_turn', 'multiple_turns'):
        points[-1] = points[0]  # Explicit common anchor, not independent-end snapping.
    chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if (not np.isfinite(points).all() or np.any(chords <= 0.) or np.any(step <= 0.)
            or step.max() > 1.+1e-12 or chord_errors.max() > .1+1e-12
            or not np.allclose(chords, expected_chords, rtol=1e-9, atol=1e-9)):
        raise PrimitiveError('UNREPRESENTABLE_PERIODIC_SAMPLES: original selected child retained')
    return dict(kind='arc' if isinstance(primitive, ArcPrimitive) else 'polyline',
        geometry_version=VERSION, traversal_kind=kind, direction=direction,
        start_phase_mm=phase, travel_length_mm=length, length_mm=length,
        is_closed=kind in ('full_turn', 'multiple_turns'),
        sample_positions_mm=travel.tolist(), points=points.tolist(),
        sampling_audit=dict(max_arc_step_mm=float(step.max()), max_chord_error_mm=float(chord_errors.max())),
        **extra)


@dataclass(frozen=True)
class Boundary:
    runtime_id: tuple
    path_id: int
    query_record_id: int
    role: str
    query_q_mm: float
    local_lift_mm: float
    phase_mm: float
    xyz_mm: tuple
    raw7: object
    observed_features: object
    anchor_ref: object
    alias_family: tuple
    closure_turns: int
    evidence_kind: str


@dataclass(frozen=True)
class TraversalEdge:
    runtime_id: int
    source: int
    target: int
    direction: int
    wrap_index: int
    length_mm: float
    traversal_kind: str
    eligible: bool
    reason: str


@dataclass(frozen=True)
class PeriodicContext:
    path_id: int
    primitive: object
    input: RuntimePrimitiveInput
    fg: PeriodicFG


class PeriodicTraversalGraph:
    """Full role-pair traversal multigraph, with explicit invalid candidates.

All original supported local candidates and independent ends are retained.
Closure aliases add logical ends without duplicating the point observations.
"""
    def __setattr__(self, name, value):
        if getattr(self, '_locked', False):
            raise PrimitiveError('Periodic graph is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, inputs, scope):
        _guard()
        if not isinstance(scope, TraversalScope):
            raise PrimitiveError('Explicit reviewed/preparation traversal scope required')
        self.scope = scope
        nodes, contexts, receipts = [], {}, []
        for item in inputs:
            if not isinstance(item, RuntimePrimitiveInput) or item.path_id in contexts:
                raise PrimitiveError('Unique bound runtime primitive inputs required')
            primitive = parse_primitive(item.geometry_parameters)
            if not isinstance(primitive, (ArcPrimitive, PolylinePrimitive)) or not primitive.is_closed:
                raise PrimitiveError('Whole closed request refused; open primitive requires its separate adapter')
            period, q, raw = primitive.length_mm, item.query_positions_mm, item.raw7
            if len(q) < 2 or q[0] != 0. or q[-1] != period:
                raise PrimitiveError('Complete one-period physical query coverage required')
            local = np.floor(q[:, None]/64.)*64. + raw[:, 4:7].astype(np.float64)*16.
            support = 1./(1.+np.exp(-np.clip(raw[:, 1:4].astype(np.float64), -60., 60.)))
            supported = support >= .5
            radius = np.abs((local-q[:, None])/16.) <= 1.+1e-6
            admitted = supported & radius  # A closed curve has no open-span gate.
            ids = np.flatnonzero(admitted.any(axis=1))
            features = np.asarray(item.feature_provider(ids.copy()))
            if features.dtype != np.float32 or features.shape != (len(ids),425) or not np.isfinite(features).all():
                raise PrimitiveError('Complete finite observed feature callback required; no zero fallback')
            original = []
            for j, query in enumerate(ids):
                alias = (item.path_id, 'seam_query', 0) if query in (0,len(q)-1) else (item.path_id,'query',int(query))
                for role in np.flatnonzero(admitted[query]):
                    lift = float(local[query,role])
                    phase = float(_phase(lift,period))
                    identity = (item.path_id,int(query),int(role),0)
                    node = Boundary(identity,item.path_id,int(query),ROLES[role],float(q[query]),
                        lift,phase,tuple(primitive.point_at(phase)),_seal(raw[query]),_seal(features[j]),
                        None,alias,0,'actual_query_role_output' if item.provenance.raw_output_kind=='actual_head_output' else 'synthetic_query_role_fields')
                    nodes.append(node); original.append(node)
            closures = 0
            for start in original:
                if start.role != 'start':
                    continue
                for turns in range(1,scope.max_turns+1):
                    nodes.append(Boundary((item.path_id,start.query_record_id,1,turns),item.path_id,
                        start.query_record_id,'end',start.query_q_mm,start.local_lift_mm,start.phase_mm,
                        start.xyz_mm,start.raw7,start.observed_features,start.runtime_id,start.alias_family,
                        turns,'derived_closure_anchor_REQUIRES_NEW_LEARNED_SCORE'))
                    closures += 1
            fg = 1./(1.+np.exp(-np.clip(raw[:,0].astype(np.float64),-60.,60.)))
            contexts[item.path_id]=PeriodicContext(item.path_id,primitive,item,PeriodicFG(q,fg,period))
            receipts.append(dict(path_id=item.path_id,all_query_count=len(q),original_admitted=len(original),
                generated_closure_ends=closures,all_independent_ends_retained=True,
                physical_phase_rule=PHASE_RULE,original_local_lifts_unmodified=True,
                support_gate_sha256=array_sha(supported),radius_gate_sha256=array_sha(radius),
                admission_sha256=array_sha(admitted),open_span_gate_applied=False,
                original_local_lifts_sha256=array_sha(local),observed_admitted_features_sha256=array_sha(features)))
        self.nodes = tuple(sorted(nodes,key=lambda n:n.runtime_id))
        self.index = MappingProxyType({n.runtime_id:i for i,n in enumerate(self.nodes)})
        if len(self.index) != len(self.nodes):
            raise PrimitiveError('Repeated explicit runtime boundary identity')
        self.contexts = MappingProxyType(contexts)
        self.admission_receipts_json = canonical_json(receipts)
        paths={p:tuple(i for i,n in enumerate(self.nodes) if n.path_id==p) for p in sorted(contexts)}
        self.paths = MappingProxyType(paths)
        self.roles = MappingProxyType({p:tuple(tuple(i for i in ids if self.nodes[i].role==role)
            for role in ROLES) for p,ids in paths.items()})
        buckets, offset = [], 0
        for p,roles in self.roles.items():
            for a,b in CHILD_ROLE_PAIRS:
                count=len(roles[a])*len(roles[b])-(len(roles[a]) if a==b else 0)
                if count:
                    buckets.append((offset,offset+count,roles[a],roles[b],a==b))
                offset += count
        self._buckets, self.base_pair_count = tuple(buckets), offset
        self.variants_per_pair = 2*(scope.max_turns+1)
        self.edge_count = offset*self.variants_per_pair
        self._locked = True

    def pair(self, identity):
        if isinstance(identity,bool) or not isinstance(identity,(int,np.integer)) or identity < 0:
            raise PrimitiveError('Nonnegative integer pair ID required')
        for begin,end,sources,targets,same in self._buckets:
            if begin <= identity < end:
                i,j=divmod(identity-begin,len(targets)-1 if same else len(targets))
                if same and j>=i:j+=1
                return sources[i],targets[j]
        raise PrimitiveError('Pair ordinal outside complete graph')

    def pair_id(self, source, target):
        for begin,end,sources,targets,same in self._buckets:
            if source in sources and target in targets and source!=target:
                i,j=sources.index(source),targets.index(target)
                return begin+i*(len(targets)-1)+(j if j<i else j-1) if same else begin+i*len(targets)+j
        raise PrimitiveError('Pair absent from complete role graph')

    def edge(self, identity):
        if isinstance(identity,bool) or not isinstance(identity,(int,np.integer)) or not 0<=identity<self.edge_count:
            raise PrimitiveError('Explicit traversal edge ID outside complete graph')
        pair,variant=divmod(int(identity),self.variants_per_pair)
        dindex,k=divmod(variant,self.scope.max_turns+1)
        direction=1 if dindex==0 else -1
        source,target=self.pair(pair)
        a,b=self.nodes[source],self.nodes[target]
        period=self.contexts[a.path_id].primitive.length_mm
        delta=b.phase_mm-a.phase_mm
        length=direction*delta+k*period
        eligible=0.<length<=self.scope.max_turns*period
        kind=('zero_span' if length==0. else 'invalid_span') if not eligible else (
            ('full_turn' if k==1 else 'multiple_turns') if delta==0. else ('partial' if length<period else 'partial_with_turns'))
        reason='' if eligible else 'ZERO_OR_OUTSIDE_DECLARED_TRAVERSAL_BUDGET'
        return TraversalEdge(int(identity),source,target,direction,k,length,kind,eligible,reason)

    def edge_id(self, source, target, direction, wrap_index):
        if direction not in (-1,1) or isinstance(wrap_index,bool) or not isinstance(wrap_index,(int,np.integer)) or not 0<=wrap_index<=self.scope.max_turns:
            raise PrimitiveError('Explicit direction and declared wrap index required')
        return self.pair_id(source,target)*self.variants_per_pair+(0 if direction==1 else self.scope.max_turns+1)+int(wrap_index)

    def iter_edges(self):
        for i in range(self.edge_count):yield self.edge(i)

    def relation_counts(self):
        return dict(original_role_pairs=self.base_pair_count,traversal_candidates=self.edge_count,
            message=sum(len(ids)*(len(ids)-1) for ids in self.paths.values()),
            duplicate=sum(len(ids)*(len(ids)-1)//2 for ids in self.paths.values()))

    def edge_features(self, identity):
        edge=self.edge(identity)
        if not edge.eligible:raise PrimitiveError(edge.reason + ': candidate remains in full ledger')
        start=self.nodes[edge.source]
        context=self.contexts[start.path_id]
        return dict(direction=edge.direction,wrap_index=edge.wrap_index,
            traversal_kind=edge.traversal_kind,length_mm=edge.length_mm,
            fg=context.fg.moments(start.phase_mm,edge.direction,edge.length_mm),
            geometry_version=VERSION,learned_edge_score='UNTRAINED')

    def fingerprint(self):
        _guard()
        nodes=[dict(identity=n.runtime_id,role=n.role,query_q=n.query_q_mm,local_lift=n.local_lift_mm,
            phase=n.phase_mm,xyz=n.xyz_mm,raw7=array_sha(n.raw7),features=array_sha(n.observed_features),
            anchor=n.anchor_ref,alias=n.alias_family,closure_turns=n.closure_turns,evidence=n.evidence_kind) for n in self.nodes]
        contexts=[dict(path=p,geometry=c.primitive.to_parameters(),all_q=array_sha(c.input.query_positions_mm),
            all_raw=array_sha(c.input.raw7),provenance=c.input.provenance.to_dict(),FG=c.fg.fingerprint())
            for p,c in sorted(self.contexts.items())]
        edges=sha256()
        for edge in self.iter_edges():edges.update(canonical_json(edge.__dict__).encode()+b'\n')
        return sha256(canonical_json(dict(version=VERSION,scope=self.scope.__dict__,nodes=nodes,
            phase_rule=PHASE_RULE,
            contexts=contexts,admission=self.admission_receipts_json,complete_edge_digest=edges.hexdigest(),
            relation_counts=self.relation_counts(),constraints='degree_DAG_direction_progress_anchor_221_v1',
            supervision='NOT_IMPLEMENTED',normalizer='NOT_FITTED',network='UNTRAINED')).encode()).hexdigest()


def validate_selected_edge_ids(graph, identities):
    edges=[graph.edge(identity) for identity in identities]
    if len({e.runtime_id for e in edges})!=len(edges):raise PrimitiveError('Repeated submitted edge ID')
    incoming,outgoing={},{}
    for edge in edges:
        if not edge.eligible:raise PrimitiveError(edge.reason)
        incoming.setdefault(edge.target,[]).append(edge)
        outgoing.setdefault(edge.source,[]).append(edge)
    used=set(incoming)|set(outgoing)
    for i in used:
        ni,no,role=len(incoming.get(i,[])),len(outgoing.get(i,[])),graph.nodes[i].role
        valid=(ni==0 and no==1) if role=='start' else ((ni==1 and no==0) if role=='end' else ni==no==1)
        if not valid:raise PrimitiveError('Illegal selected S/E/C degree')
    parents,visited=[],set()
    for start in sorted(i for i in used if graph.nodes[i].role=='start'):
        current,chain,order,progress=start,[start],[],[0.]
        while graph.nodes[current].role!='end':
            edge=outgoing[current][0]
            if edge.target in chain or edge.target in visited:raise PrimitiveError('Logical cycle or reused node')
            chain.append(edge.target);order.append(edge)
            progress.append(progress[-1]+edge.length_mm);current=edge.target
        if len({e.direction for e in order})!=1:raise PrimitiveError('Inconsistent parent traversal direction')
        context=graph.contexts[graph.nodes[start].path_id]
        period=context.primitive.length_mm
        tolerance=128.*np.finfo(np.float64).eps*max(period,progress[-1],1.)
        if progress[-1]>graph.scope.max_turns*period+tolerance:raise PrimitiveError('Parent exceeds declared traversal budget')
        end=graph.nodes[current]
        if end.closure_turns:
            if end.anchor_ref!=graph.nodes[start].runtime_id:raise PrimitiveError('Closure end references another start anchor')
            if abs(progress[-1]-end.closure_turns*period)>tolerance:raise PrimitiveError('Closure parent has not traversed its explicit full budget')
        visited.update(chain)
        parents.append(dict(runtime_parent_id=len(parents),node_ids=chain,
            ordered_edge_ids=[e.runtime_id for e in order],direction=order[0].direction,
            literal_progress_mm=progress,total_travel_mm=progress[-1],
            start_anchor_ref=list(graph.nodes[start].runtime_id),closure_turns=end.closure_turns))
    if visited!=used:raise PrimitiveError('Disconnected logical cut cycle')
    return dict(parents=parents,selected_children=len(edges),acyclic=True,
        business_scope_reviewed=graph.scope.business_scope_reviewed,
        gap_business_parent_schema='NOT_IMPLEMENTED_NO_GT_REMOVED')


def export_selected_edge_ids(graph, identities):
    identities=list(identities)
    try:
        validation=validate_selected_edge_ids(graph,identities);structural_error=''
    except (PrimitiveError,ValueError) as exc:
        validation=dict(parents=[],selected_children=len(identities));structural_error=str(exc)
    order={edge:(parent['runtime_parent_id'],j) for parent in validation['parents']
        for j,edge in enumerate(parent['ordered_edge_ids'])}
    ledger=[]
    for ordinal,identity in enumerate(identities):
        row=dict(submitted_child_ordinal=ordinal,edge_id=identity,counts_as_prediction=True,
            invalid_geometry_FP=True,status='INVALID_GEOMETRY_FP',segment=None)
        try:
            edge=graph.edge(identity)
            a,b=graph.nodes[edge.source],graph.nodes[edge.target]
            row.update(source_node_id=list(a.runtime_id),target_node_id=list(b.runtime_id),
                original_local_lifts_mm=[a.local_lift_mm,b.local_lift_mm],phases_mm=[a.phase_mm,b.phase_mm],
                original_xyz_mm=[a.xyz_mm,b.xyz_mm],direction=edge.direction,wrap_index=edge.wrap_index,
                traversal_kind=edge.traversal_kind)
            if structural_error:raise PrimitiveError(structural_error)
            segment=segment_traversal(graph.contexts[a.path_id].primitive,a.phase_mm,
                edge.direction,edge.length_mm,edge.traversal_kind)
            # Endpoint references are the original explicit logical anchors.
            segment['points'][0]=list(a.xyz_mm);segment['points'][-1]=list(b.xyz_mm)
            segment.update(runtime_child_edge_id=identity,start_node_id=list(a.runtime_id),
                end_node_id=list(b.runtime_id),endpoint_basis='explicit_logical_anchor_refs')
            row.update(invalid_geometry_FP=False,status='VALID_GEOMETRY_UNSCORED',segment=segment,
                runtime_parent_id=order[identity][0],child_order=order[identity][1])
        except (PrimitiveError,ValueError,OverflowError,IndexError) as exc:row['reason']=str(exc)
        ledger.append(row)
    invalid=sum(r['invalid_geometry_FP'] for r in ledger)
    return dict(schema=VERSION,graph_fingerprint=graph.fingerprint(),child_ledger=ledger,
        selected_children=len(identities),delivered_geometry_children=len(identities)-invalid,
        invalid_geometry_FP=invalid,prediction_count_including_invalid=len(identities),
        no_submitted_child_removed=True,structural_validation=validation,
        actual_neural_scoring='UNTRAINED',Good_Bad_solver='NOT_IMPLEMENTED',
        learned_accuracy_measured=False,full_contract_pass=False)
