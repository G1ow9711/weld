"""Finite joint identity for newly added wall sections and curve-only scope.

This changes publication of proven duplicate representations, not should-weld
classification. Original models, inputs, true gaps and process cuts are kept.
"""
import copy,time
import numpy as np
from . import curveonly_01a1ac7c_step_joint_rules_v373 as step
from . import curveonly_01a1ac7c_representation_containment_v461 as conversion
from . import curveonly_01a1ac7c_native_join_rules_v418 as continuity
from . import curveonly_01a1ac7c_arc_publication_v6 as publication
from . import curveonly_01a1ac7c_sampler_v5 as sampler

cad=step.finite.cad
NUMERIC=step.NUMERIC
SOURCE_MM=sampler.TOLERANCE_MM
ENDPOINT_MM=continuity.ENDPOINT_UNCERTAINTY_MM
GEOMETRY_REASONS=conversion.ALLOWED_REASONS|{'FINITE_CARRIER_INTERVAL_SET','CIRCULAR_CONVERSION_ERROR_BOUND_EXCEEDED'}
PROCESS_KEYS=conversion.PROCESS_KEYS+('_confirmed_process_cut_ends','confirmed_process_cuts','confirmed_process_events')


def process_safe(parent):
    if parent.get('is_closed') or any(parent.get(k) for k in PROCESS_KEYS):return False
    segments=parent.get('segments',[])
    return bool(segments) and all(s.get('reason') in GEOMETRY_REASONS and not any(s.get(k) for k in PROCESS_KEYS) for s in segments)


def same_category(a,b):
    return a['weld_type']==b['weld_type'] and a.get('weld_side')==b.get('weld_side')


def polyline_conversion(parent,graph):
    """Independent continuous bound on every supported published polyline."""
    if not process_safe(parent):return dict(passed=False,reason='PROTECTED_PROCESS_OR_CLOSED_UNIT')
    if any(s['geometry_type']=='arc' for s in parent['segments']):return conversion.continuous_conversion(parent,graph)
    if any(s['geometry_type'] not in ('polyline','bspline_polyline','line') for s in parent['segments']):
        return dict(passed=False,reason='UNKNOWN_PUBLISHED_REPRESENTATION')
    pieces=[np.asarray(s['points'],float) for s in parent['segments']]
    if any(np.linalg.norm(a[-1]-b[0])>NUMERIC for a,b in zip(pieces[:-1],pieces[1:])):
        return dict(passed=False,reason='ACTUAL_PUBLISHED_GAP')
    joined=np.vstack([pieces[0],*[p[1:] for p in pieces[1:]]]);points=np.asarray(parent['points'],float)
    if joined.shape!=points.shape or np.max(np.linalg.norm(joined-points,axis=1))>NUMERIC:
        return dict(passed=False,reason='PARENT_CHILD_GEOMETRY_MISMATCH')
    value=graph.interval(parent)
    if value is None:return dict(passed=False,reason='UNKNOWN_SOURCE_RANGE')
    ci,a,b=value;native=cad.slice_curve(graph.packet['candidates'][ci]['points'],a,b)
    spacing=conversion.local.QUERY_SPACING
    def bound(left,right):
        q=cad.resample(left,spacing);s=cad.arclength(q)
        return float(publication.distance_upper(q,right).max()+np.diff(s).max()/2.+NUMERIC)
    upper=max(bound(native,points),bound(points,native))
    return dict(passed=upper<=publication.MAX_CONVERSION_ERROR_MM,continuous_error_upper_mm=upper,
        independent_bidirectional_polyline_bound=True,unchanged_publication_limit_mm=publication.MAX_CONVERSION_ERROR_MM)


def ordered_containment(query,reference):
    q=cad.resample(query,SOURCE_MM/2.);s=cad.arclength(q)
    mapped=step.aliases.coincident_parameters(q,reference)
    if mapped is None:return None
    station,error=mapped;sign=1. if station[-1]>=station[0] else -1.
    if np.min(sign*np.diff(station)) < -NUMERIC:return None
    if abs(abs(station[-1]-station[0])-s[-1])>2*SOURCE_MM:return None
    upper=float(error+np.diff(s).max()/2.+NUMERIC)
    if upper>SOURCE_MM:return None
    return dict(continuous_source_distance_upper_mm=upper,query_samples=len(q),
        ordered_reference_interval_mm=[float(station.min()),float(station.max())],
        reversed=sign<0,original_source_tolerance_mm=SOURCE_MM,no_repeated_or_reversed_subranges=True)


class Context:
    def __init__(self,packet,inventory,solids):
        self.original=step.Context(solids,packet,inventory);self.packet=packet;self.inventory=inventory
        self.R=self.original.geometry.R;self.t=self.original.geometry.t
        self.faces=[cad.TopoDS.Face_s(f) for solid in solids for f in cad.items(solid,cad.TopAbs_FACE)]
        self.rows=inventory['faces'];self.geometry=conversion.topology.NativeUnits(packet)

    def finite_face(self,world,face_index):
        """Distance to a closed finite face is 1-Lipschitz on each segment."""
        face=self.faces[face_index];p=(np.asarray(world)-self.t)@self.R;calls=0;cache={}
        def distance(q):
            nonlocal calls
            key=tuple(q)
            if key not in cache:
                calls+=1
                if calls>250000:raise RuntimeError('BOUNDED_GEOMETRY_QUERY_BUDGET_UNKNOWN')
                cache[key]=cad.finite_distance(face,q)+NUMERIC
            return cache[key]
        try:
            if any(distance(p[i])>SOURCE_MM for i in [0,len(p)//2,len(p)-1]):return None
            values=[distance(q) for q in p]
            if max(values)>SOURCE_MM:return None
            stack=[(a,b,x,y) for a,b,x,y in zip(p[:-1],p[1:],values[:-1],values[1:])];upper=0.
            while stack:
                a,b,x,y=stack.pop();length=float(np.linalg.norm(b-a))
                bound=max(x,y)+length/2.
                if bound<=SOURCE_MM:upper=max(upper,bound);continue
                if length<=NUMERIC:return None
                mid=(a+b)/2.;value=distance(mid)
                if value>SOURCE_MM:return None
                stack.extend([(a,mid,x,value),(mid,b,value,y)])
            return dict(face=face_index,material=int(self.rows[face_index]['material']),queries=calls,
                continuous_finite_face_distance_upper_mm=upper,adaptive_segment_Lipschitz_bound=True)
        except (RuntimeError,ValueError):return None

    def joint(self,primitive):
        c=self.packet['candidates'][primitive];base=c.get('base');wall=c.get('material');expected=c.get('base_face')
        if base not in self.original.panels or wall in self.original.panels or wall is None or wall<0:return None
        if not all(r.get('face_valid',False) for r in self.rows):return None
        points=np.asarray(c['points']);base_matches=[]
        for owner,panel in self.original.panels.items():
            for face in panel['dominant_faces']:
                proof=self.finite_face(points,face)
                if proof is not None:base_matches.append(proof)
        if len(base_matches)!=1 or (base_matches[0]['material'],base_matches[0]['face'])!=(base,expected):return None
        normal=np.asarray(self.rows[expected]['normal']);partners=[]
        for i,row in enumerate(self.rows):
            if row['material']==base:continue
            if row['status']=='CANONICAL_PLANE' and abs(abs(np.asarray(row['normal'])@normal)-1.)<=NUMERIC:continue
            proof=self.finite_face(points,i)
            if proof is not None:partners.append(proof)
        if len(partners)!=1 or partners[0]['material']!=wall:return None
        return dict(base=base,wall=wall,base_face=base_matches[0],wall_side_face=partners[0],
            unique_finite_base_and_wall_side=True,geometry_does_not_grant_should_weld=True)


def inspect(decoded,packet,inventory,solids):
    meta=packet.get('mixed_source_observations_v487',{});original_count=meta.get('original_candidates')
    if original_count is None or not solids or not packet.get('pose',{}).get('passed'):
        return [],dict(not_applicable='NO_NEW_MIXED_SOURCE_CONTEXT',certificates=[])
    parents=decoded['parents'];anchors=[i for i,p in enumerate(parents) if p['source_candidate']<original_count and process_safe(p)]
    if not anchors:return [],dict(certificates=[],reason='NO_EXISTING_CURVE_MODEL_ANCHOR')
    context=Context(packet,inventory,solids);aliases={};identity=[];joint_cache={};conversion_cache={}
    def converted(i):
        if i not in conversion_cache:conversion_cache[i]=polyline_conversion(parents[i],context.geometry)
        return conversion_cache[i]
    for ci,c in enumerate(packet['candidates'][original_count:],original_count):
        if c.get('native_geometry_kind')!='NATIVE_NONLINEAR' or c.get('source_kind')!='FINITE_WALL_SECTION_AT_OBSERVED_BASE_PLANE':continue
        if np.linalg.norm(np.asarray(c['points'])[0]-c['points'][-1])<=NUMERIC:continue
        possible=[]
        for i in anchors:
            parent=parents[i];old=packet['candidates'][parent['source_candidate']]
            if old.get('material')!=c.get('base') or not(old.get('STEP') and old.get('pose')):continue
            if not converted(i).get('passed'):continue
            native=step.native_parent(parent,context.original)
            mapping=ordered_containment(c['points'],native)
            if mapping is not None:possible.append((i,mapping))
        if len(possible)!=1:continue
        if ci not in joint_cache:joint_cache[ci]=context.joint(ci)
        physical=joint_cache[ci]
        if physical is None:continue
        i,mapping=possible[0];cert=dict(primitive=ci,anchor_parent_index=i,anchor_source=parents[i]['source_candidate'],
            finite_joint=physical,native_containment=mapping,anchor_conversion=converted(i))
        aliases[ci]=cert;identity.append(cert)
    removal=[];declined=[]
    for i,parent in enumerate(parents):
        ci=parent['source_candidate']
        if ci<original_count or not process_safe(parent):continue
        if ci in aliases:
            cert=aliases[ci];anchor=parents[cert['anchor_parent_index']]
            if same_category(parent,anchor) and converted(i).get('passed'):
                removal.append(dict(parent_index=i,kind='DUPLICATE_FINITE_JOINT_CURVE',identity=cert,published_conversion=converted(i)))
            continue
        c=packet['candidates'][ci];path=c.get('source_material_path',[])
        if c.get('source_kind')!='FINITE_CURVED_TO_LINEAR_DIFFERENT_WALL_BASE_PLANE_CANDIDATE' or len(path)!=2:continue
        a,b=path
        if a['curve'] not in aliases or packet['candidates'][b['curve']].get('native_geometry_kind')!='CERTIFIED_NATIVE_LINEAR':continue
        if any(c.get(k) for k in PROCESS_KEYS):continue
        if any(g.get('weld_gap_or_process_cut_status')!='UNKNOWN_NOT_AUTOMATIC' for g in c.get('nominal_clearance_intervals',[])):continue
        cert=aliases[a['curve']];anchor=parents[cert['anchor_parent_index']]
        if not same_category(parent,anchor):continue
        r=parent['interval'];curved=max(0.,min(r['end_mm'],a['end_mm'])-max(r['start_mm'],a['start_mm']))
        linear=max(0.,min(r['end_mm'],b['end_mm'])-max(r['start_mm'],b['start_mm']))
        external_uncovered=max(0.,r['start_mm']-a['start_mm'])
        # Use existing join/endpoint uncertainty only to identify model spill.
        # Whole hybrids and genuine short nonlinear primitives are protected.
        if not(0.<curved<=ENDPOINT_MM and linear>ENDPOINT_MM and external_uncovered>ENDPOINT_MM):
            declined.append(dict(parent_index=i,reason='PRESERVE_WHOLE_OR_NONTRIVIAL_CURVED_MODEL_RANGE'));continue
        if not converted(i).get('passed'):continue
        removal.append(dict(parent_index=i,kind='STRAIGHT_TAIL_WITH_ALREADY_REPRESENTED_CURVED_ENDPOINT_SPILL',
            identity=cert,curved_endpoint_spill_mm=curved,linear_native_length_mm=linear,
            original_endpoint_uncertainty_mm=ENDPOINT_MM,published_conversion=converted(i),
            original_nominal_clearances=copy.deepcopy(c.get('nominal_clearance_intervals',[])),
            straight_should_weld_and_process_status='UNKNOWN_OUTSIDE_CURVE_MODEL_SCOPE'))
    return removal,dict(certificates=removal,finite_curve_identities=identity,declined=declined,
        old_anchors_preserved=True,no_new_fit_or_VALcalibration=True,no_spatial_only_or_part_lookup=True)


def correct(decoded,packet,inventory,solids):
    start=time.perf_counter();removal,audit=inspect(decoded,packet,inventory,solids)
    result=copy.deepcopy(decoded);indices={r['parent_index'] for r in removal}
    if indices:
        result['parents']=[p for i,p in enumerate(result['parents']) if i not in indices]
        for i,p in enumerate(result['parents']):p['parent_id']=f'curve_{i:05d}'
        result['geometry_units']=len(result['parents']);result['published_segments']=sum(len(p['segments']) for p in result['parents'])
    result['finite_curve_alias_scope_rule']=dict(audit,removed_parent_indices=sorted(indices),seconds=time.perf_counter()-start,
        source_precision_mm=SOURCE_MM,publication_conversion_mm=publication.MAX_CONVERSION_ERROR_MM,
        full_hybrids_short_curves_true_gaps_sides_and_confirmed_cuts_preserved=True)
    return result
