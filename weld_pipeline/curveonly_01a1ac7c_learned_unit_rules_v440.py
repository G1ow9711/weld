"""Resolve native source conflicts using two frozen learned complete units.

Only an existing post-structural R35 proposal can be added. The second expert
provides independent learned unit boundaries, never replacement geometry.
"""
import copy,time
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_source_selection_v56 as provenance
from . import curveonly_01a1ac7c_runtime_v51 as exporter
from . import curveonly_01a1ac7c_rule_runtime_v345 as source_runtime
from . import curveonly_01a1ac7c_arc_publication_v6 as publication
from . import curveonly_01a1ac7c_corner_publication_v256 as corner

# Existing acceptance endpoint tolerance compares two learned unit hypotheses.
# It is not used as a source projection or material gap tolerance.
ENDPOINT_MATCH_MM=3.
PERIODIC_NUMERIC_MM=1e-6


def candidate(p):return int(p.get('candidate',p.get('source_candidate',-1)))


def same_category(a,b):
    return a['weld_type']==b['weld_type'] and a.get('weld_side')==b.get('weld_side')


class NativeUnits:
    def __init__(self,packet):
        self.packet=packet;self.lengths=[float(cad.arclength(c['points'])[-1]) for c in packet['candidates']]
        self.parents=[[] for c in self.lengths];self._embeddings={};self._spans={}
        for parent,c in enumerate(packet['candidates']):
            for m in c.get('source_material_path',[]):
                child=int(m['curve']);length=self.lengths[child]
                if abs(float(m['end_mm'])-float(m['start_mm'])-length)>1e-7*max(1.,length):
                    raise ValueError('Native source length mapping mismatch')
                sign=-1. if m['reversed'] else 1.
                offset=float(m['start_mm'])+(length if m['reversed'] else 0.)
                self.parents[child].append((parent,offset,sign))

    def embeddings(self,ci):
        if ci in self._embeddings:return self._embeddings[ci]
        out={}
        def visit(node,offset,sign,seen):
            if node in seen:raise ValueError('Cyclic native source provenance')
            out.setdefault(node,[]).append((offset,sign))
            for parent,a,b in self.parents[node]:visit(parent,a+b*offset,b*sign,seen+(node,))
        visit(ci,0.,1.,());self._embeddings[ci]=out;return out

    def interval(self,p):
        ci=candidate(p);q=p.get('interval')
        if not q or not 0<=ci<len(self.lengths):return None
        a,b=float(q['start_mm']),float(q['end_mm']);length=self.lengths[ci]
        if not np.isfinite([a,b]).all() or a<0. or b>length+1e-8 or b<=a:return None
        return ci,a,min(b,length)

    def open(self,p):
        values=self.interval(p)
        if values is None:return False
        ci,a,b=values;c=self.packet['candidates'][ci]
        if not(c.get('STEP') and c.get('pose')):return False
        # Native full periodic paths are protected at every physical scale.
        points=np.asarray(c['points'])
        if np.linalg.norm(points[0]-points[-1])<=PERIODIC_NUMERIC_MM and abs(b-a-self.lengths[ci])<=PERIODIC_NUMERIC_MM:
            return False
        ends=cad.slice_curve(points,a,b)[[0,-1]]
        return bool(np.linalg.norm(ends[0]-ends[-1])>PERIODIC_NUMERIC_MM)

    def views(self,a,b):
        ra,rb=self.interval(a),self.interval(b)
        if ra is None or rb is None:return []
        ca,lo,hi=ra;cb,left,right=rb;aa,bb=self.embeddings(ca),self.embeddings(cb);out=[]
        for root in sorted(set(aa)&set(bb),key=lambda ci:(self.lengths[ci],ci)):
            # Multiple occurrences cannot be identified by nearest coordinates.
            if len(aa[root])!=1 or len(bb[root])!=1:continue
            x,s=aa[root][0];y,t=bb[root][0]
            u=sorted((x+s*lo,x+s*hi));v=sorted((y+t*left,y+t*right))
            out.append(dict(common_native_source=root,left_mm=u,right_mm=v,
                left_reversed=s<0.,right_reversed=t<0.,
                endpoint_parameter_delta_mm=max(abs(u[0]-v[0]),abs(u[1]-v[1]))))
        return out

    def equivalent(self,a,b):
        if not same_category(a,b) or not(self.open(a) and self.open(b)):return None
        good=[v for v in self.views(a,b) if v['endpoint_parameter_delta_mm']<=ENDPOINT_MATCH_MM
            and min(v['left_mm'][1],v['right_mm'][1])>max(v['left_mm'][0],v['right_mm'][0])]
        return good[0] if good else None

    def contains(self,outer,inner):
        if not same_category(outer,inner) or not(self.open(outer) and self.open(inner)):return None
        for v in self.views(outer,inner):
            a,b=v['left_mm'];x,y=v['right_mm'];eps=1e-7*max(1.,self.lengths[v['common_native_source']])
            if x>=a-eps and y<=b+eps:return v
        return None

    def atoms(self,p):
        ci,a,b=self.interval(p)
        return provenance.primitive_intervals(ci,a,b,self.packet,self.lengths)

    def overlap(self,a,b):
        if self.interval(a) is None or self.interval(b) is None:return []
        return [dict(primitive=ci,start_mm=max(x,u),end_mm=min(y,v))
            for ci,x,y in self.atoms(a) for cj,u,v in self.atoms(b)
            if ci==cj and min(y,v)>max(x,u)]

    def disjoint(self,a,b):
        return not self.overlap(a,b)


def key(p):
    q=p['interval'];return candidate(p),float(q['start_mm']),float(q['end_mm']),float(p['confidence'])


def inspect(decoded,packet,raw,witnesses):
    graph=NativeUnits(packet);parents=decoded['parents']
    # Exact entries of the final source selector establish that all structural
    # processing already took place. No rejected raw range is manufactured here.
    rejected={key(q) for q in decoded.get('finite_source_selection',{}).get('rejected',[])}
    raw_kept=[]
    for p in raw:
        if key(p) in rejected or any(graph.equivalent(p,old) for old in parents):raw_kept.append(p)
    qualified=[w for w in witnesses if graph.open(w)]
    raw_matches=[]
    for p in raw_kept:
        matches=[i for i,w in enumerate(qualified) if graph.equivalent(p,w)]
        if len(matches)==1:raw_matches.append((p,matches[0]))
    parent_matches=[];parent_ambiguity=[]
    for p in parents:
        matches=[i for i,w in enumerate(qualified) if graph.equivalent(p,w)]
        parent_matches.append(matches[0] if len(matches)==1 else None)
        parent_ambiguity.append(len(matches)>1)
    groups={}
    for p,wi in raw_matches:groups.setdefault(wi,[]).append(p)
    additions=[];remove=set();certificates=[];declined=[]
    for wi,proposals in groups.items():
        witness=qualified[wi]
        anchors=[i for i,v in enumerate(parent_matches) if v==wi]
        eligible=proposals if anchors else [p for p in proposals if key(p) in rejected]
        if not eligible:
            declined.append(dict(expert_unit=wi,reason='NO_EXACT_POST_STRUCTURAL_RAW_REJECTION'));continue
        representative=max(eligible,key=lambda p:(p['confidence'],p['interval']['end_mm']-p['interval']['start_mm']))
        cover=parents[anchors[0]] if anchors else representative
        nested=[];compatible=[];blocked=[]
        for i,p in enumerate(parents):
            if i in anchors or not same_category(p,representative):continue
            overlap=graph.overlap(representative,p)
            if not overlap:continue
            other=parent_matches[i]
            if other is not None and other!=wi:
                if graph.disjoint(witness,qualified[other]):
                    compatible.append(dict(parent_index=i,other_expert_unit=other,
                        same_native_endpoint_overlap=overlap,other_complete_unit_match=graph.equivalent(p,qualified[other])))
                else:blocked.append(dict(parent_index=i,reason='DISTINCT_EXPERT_UNITS_OVERLAP_PRESERVE_CURRENT'))
            elif other is None and not parent_ambiguity[i] and graph.contains(cover,p):
                nested.append(dict(parent_index=i,native_containment=graph.contains(cover,p),
                    exported_closed_flag=p.get('is_closed'),native_interval_is_open=True))
            else:blocked.append(dict(parent_index=i,reason='PRESERVE_UNMATCHED_OR_LONGER_CURRENT_PARENT'))
        if not anchors and blocked:
            declined.append(dict(expert_unit=wi,raw_candidate=candidate(representative),reason='UNRESOLVED_CURRENT_PARENT_CONFLICT',blockers=blocked));continue
        if anchors:
            # Existing complete parent geometry and fields are never replaced.
            if not nested:continue
        else:additions.append(representative)
        remove.update(x['parent_index'] for x in nested)
        certificates.append(dict(expert_unit=wi,expert_candidate=candidate(witness),expert_interval=witness['interval'],
            raw_candidate=candidate(representative),raw_interval=representative['interval'],
            raw_confidence=representative['confidence'],expert_confidence=witness['confidence'],
            unit_agreement=graph.equivalent(representative,witness),existing_complete_parent_indices=anchors,
            remove_nested=nested,preserved_distinct_units=compatible,
            added_raw_proposal=not bool(anchors),raw_range_present_after_structural_processing=key(representative) in rejected,
            no_expert_geometry_copied=True,no_native_gap_or_source_projection_tolerance_change=True))
    return additions,remove,dict(branch='LEARNED_COMPLETE_UNIT_SOURCE_ARBITRATION_V440',
        certificates=certificates,declined=declined,raw_proposals=len(raw),expert_units=len(witnesses),
        unique_complete_unit_raw_matches=len(raw_matches),
        native_topology_instead_of_legacy_point_one_mm_endpoint_closeness=True,
        existing_complete_parent_geometry_and_fields_preserved=True,
        no_name_radius_coordinate_or_fixed_count_rule=True,no_new_fit_or_VALcalibration=True)


def correct(decoded,features,packet,raw,expert,expert_proposals=None):
    started=time.perf_counter()
    if expert_proposals is None:
        proposed=expert.predict(features[:,:34],packet);witnesses,selection=provenance.select(proposed,packet)
    else:witnesses=expert_proposals;selection=dict(exact_bound_expert_replay=True)
    additions,remove,audit=inspect(decoded,packet,raw,witnesses);result=copy.deepcopy(decoded)
    parents=[p for i,p in enumerate(result['parents']) if i not in remove]
    if additions:
        old=publication._fallback;publication._fallback=corner.fallback
        try:new=exporter.decode(features,packet,source_runtime.Proposals(additions,dict(two_frozen_models=True)))
        finally:publication._fallback=old
        assert len(new['parents'])==len(additions)
        parents.extend(new['parents'])
    if additions or remove:
        for i,p in enumerate(parents):p['parent_id']=f'curve_{i:05d}'
        result.update(parents=parents,geometry_units=len(parents),published_segments=sum(len(p['segments']) for p in parents))
    audit.update(replaced_upstream_parent_indices=sorted(remove),new_parent_count=len(additions),
        expert_native_source_selection=selection,seconds=time.perf_counter()-started)
    result['learned_unit_source_rule']=audit
    return result
