"""Typed native-curve continuation geometry for the pending parent head.

Inputs are predicted curves and input-generated face associations only. Labels,
STEP, source identity and material numeric IDs are not read by this module.
Both traversal orientations are explicit candidate states, never an evaluation
permission to flip individual reference children.
"""
import numpy as np
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier


def continuation_inputs(children,profiles):
    curves=[CurveCarrier(c['representation']) for c in children]
    count=len(curves)
    if not count:return np.empty((0,4),np.int64),np.empty((0,20),np.float32)
    endpoints=np.array([c.at(np.array([0.,c.length])) for c in curves])
    tangents=np.array([c.tangent(np.array([0.,c.length])) for c in curves])
    length=np.array([c.length for c in curves]);centers=np.array([c.at(c.length/2) for c in curves])
    pid=np.array([c['profile'] for c in children],np.int64)
    face_sets=[set(profiles[int(i)]['face_indices']) for i in pid]
    kind=np.array([c.kind for c in curves]);pair_parts=[];feature_parts=[]
    for reverse_a in (0,1):
        for reverse_b in (0,1):
            da=tangents[:,1-reverse_a]*(1-2*reverse_a)
            db=tangents[:,reverse_b]*(1-2*reverse_b)
            for i,curve in enumerate(curves):
                delta=endpoints[:,reverse_b]-endpoints[i,1-reverse_a]
                gap=np.linalg.norm(delta,axis=1);along=delta@da[i]
                perp=np.linalg.norm(delta-along[:,None]*da[i],axis=1);tangent=db@da[i]
                ahead=(centers-centers[i])@da[i]
                straight=(kind=='line')&(curve.kind=='line')&(tangent>.999)&(along>=-3.)&(perp<=3.)&(ahead>1e-6)
                circular=np.zeros(count,bool)
                if curve.kind=='arc':
                    axis=curve.arc.plane_normal*np.sign(curve.arc.sweep_rad)*(1-2*reverse_a)
                    for j,other in enumerate(curves):
                        if other.kind!='arc':continue
                        other_axis=other.arc.plane_normal*np.sign(other.arc.sweep_rad)*(1-2*reverse_b)
                        circular[j]=(np.linalg.norm(curve.arc.center_mm-other.arc.center_mm)<=3.
                            and abs(curve.arc.radius_mm-other.arc.radius_mm)<=3. and axis@other_axis>.999)
                legal=(gap<=6.)|straight|circular;legal[i]=False;js=np.flatnonzero(legal)
                if not len(js):continue
                shared=np.array([2*len(face_sets[i]&face_sets[j])/max(1,len(face_sets[i])+len(face_sets[j])) for j in js])
                geom=np.c_[delta[js]/1000.,gap[js]/1000.,along[js]/1000.,perp[js]/3.,tangent[js],
                    np.full(len(js),np.log1p(length[i])/10.),np.log1p(length[js])/10.,pid[js]==pid[i],shared,kind[js]==kind[i],
                    np.full(len(js),reverse_a),np.full(len(js),reverse_b),np.tile(da[i],(len(js),1)),db[js]]
                pair_parts.append(np.c_[np.full(len(js),i),np.full(len(js),reverse_a),js,np.full(len(js),reverse_b)])
                feature_parts.append(geom)
    pairs=np.concatenate(pair_parts).astype(np.int64) if pair_parts else np.empty((0,4),np.int64)
    relation=np.concatenate(feature_parts).astype(np.float32) if feature_parts else np.empty((0,20),np.float32)
    assert relation.shape==(len(pairs),20) and np.isfinite(relation).all()
    return pairs,relation


def controls():
    from weld_pipeline.contract_curve_geometry import CircularArc
    def line(a,b,profile=0):return dict(representation=dict(kind='line',endpoints_mm=[a,b]),profile=profile)
    profiles=[dict(face_indices=[0,1]),dict(face_indices=[1])]
    children=[line([0,0,0],[10,0,0]),line([10,0,0],[20,0,0]),line([100,0,0],[120,0,0]),line([0,40,0],[10,40,0],1)]
    pairs,features=continuation_inputs(children,profiles);rows={tuple(p) for p in pairs}
    assert (0,0,1,0) in rows and (1,0,2,0) in rows
    assert not any(p[0]==0 and p[2]==3 for p in pairs)
    a=CircularArc([0,0,0],20,[0,0,1],[1,0,0],np.pi/2,False)
    b=CircularArc([0,0,0],20,[0,0,1],[-1,0,0],np.pi/2,False)
    arcs=[dict(representation=dict(kind='arc',arc_parameters=v.to_parameters()),profile=1) for v in (a,b)]
    pairs,features=continuation_inputs(arcs,profiles)
    assert (0,0,1,0) in {tuple(p) for p in pairs} # positive circular gap exceeds 6 mm
    pairs,features=continuation_inputs([],profiles);assert pairs.shape==(0,4) and features.shape==(0,20)
    return dict(shared_endpoint_candidate=True,long_collinear_gap_candidate=True,
        displaced_parallel_lines_excluded=True,circular_positive_gap_candidate=True,
        both_traversal_states_explicit=True,empty_output_supported=True,
        parent_head_not_trained=True,optimizer_updates=0)
