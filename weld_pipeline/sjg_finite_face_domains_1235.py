"""Observed-face convex outer domains; NOT physical or weldable truth.

Analytic clipping supports both finite lines and circular arcs. Convex hulls
cannot resolve holes, concavities, disconnected materials or process setbacks.
They provide boundary proposals only; every unresolved case stays explicit.
"""
import argparse,gc,json,math,time,traceback
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_relative_geometry_1135 import transformations
from weld_pipeline.sjg_joint_graph_capacity_1105 import ContinuousGraphBank
from weld_pipeline.sjg_parent_set_runtime_1081 import decode_choices


def face_halfspaces(points,normal):
    normal=np.asarray(normal,float);norm=np.linalg.norm(normal)
    if len(points)<3 or norm<1e-10:raise ValueError('Unknown or degenerate face')
    normal/=norm;axis=np.eye(3)[np.argmin(np.abs(normal))]
    u=axis-normal*(axis@normal);u/=np.linalg.norm(u);basis=np.column_stack((u,np.cross(normal,u)))
    origin=points.mean(0);projected=(points-origin)@basis
    hull=ConvexHull(projected);A=hull.equations[:,:2]@basis.T;b=hull.equations[:,2]-A@origin
    return A,b,dict(points=len(points),vertices=len(hull.vertices),
        maximum_plane_residual_mm=float(np.max(np.abs((points-origin)@normal))))


def domains(curve,A,b):
    """Intersect every projected-face halfspace with the original carrier."""
    tol=1e-8 # floating arithmetic tolerance in millimetres, not match tolerance
    if curve.kind=='line':
        rate=A@curve.direction;start=A@curve.ends[0]+b;lo=0.;hi=curve.length
        for slope,value in zip(rate,start):
            if abs(slope)<1e-12:
                if value>tol:return []
            elif slope>0:hi=min(hi,-value/slope)
            else:lo=max(lo,-value/slope)
        return [[max(0.,lo),min(curve.length,hi)]] if hi-lo>tol else []
    arc=curve.arc;sign=np.sign(arc.sweep_rad);span=abs(arc.sweep_rad)
    u=arc.start_direction;v=sign*np.cross(arc.plane_normal,u)
    aa=arc.radius_mm*(A@u);bb=arc.radius_mm*(A@v);cc=A@arc.center_mm+b
    cuts=[0.,span]
    for a,d,c in zip(aa,bb,cc):
        radius=math.hypot(a,d)
        if radius<1e-12:
            if c>tol:return []
            continue
        value=-c/radius
        if -1<=value<=1:
            phase=math.atan2(d,a);delta=math.acos(value)
            for root in (phase-delta,phase+delta):
                for k in range(-2,3):
                    q=root+2*math.pi*k
                    if 0<q<span:cuts.append(q)
    cuts=sorted(set(cuts));result=[]
    for left,right in zip(cuts[:-1],cuts[1:]):
        if right-left<1e-12:continue
        middle=(left+right)/2
        if np.max(aa*np.cos(middle)+bb*np.sin(middle)+cc)<=tol:
            lo,hi=left*arc.radius_mm,right*arc.radius_mm
            if result and abs(result[-1][1]-lo)<tol:result[-1][1]=hi
            else:result.append([lo,hi])
    return result


def controls(dest):
    points=np.array([[-5,-2,0],[-5,2,0],[5,-2,0],[5,2,0]],float)
    ends=np.array([[-10.,0,0],[10.,0,0]])
    A,b,_=face_halfspaces(points,[0,0,1]);line=CurveCarrier(dict(kind='line',endpoints_mm=ends.tolist()))
    reference=domains(line,A,b);assert np.allclose(reference,[[5,15]],atol=1e-12)
    errors=[]
    for name,R,shift in transformations():
        aa,bb,_=face_halfspaces(points@R.T+shift,np.array([0,0,1])@R.T)
        moved=CurveCarrier(dict(kind='line',endpoints_mm=(ends@R.T+shift).tolist()))
        error=float(np.max(np.abs(np.array(domains(moved,aa,bb))-reference)));assert error<1e-10;errors.append(error)
    from weld_pipeline.contract_curve_geometry import CircularArc
    arc=CircularArc([0,0,0],3,[0,0,1],[1,0,0],2*np.pi,True)
    curve=CurveCarrier(dict(kind='arc',arc_parameters=arc.to_parameters()));intervals=domains(curve,A,b)
    assert len(intervals)==3 # first/last domains share the circular anchor
    q=np.linspace(0,curve.length,1001);inside=np.max(curve.at(q)@A.T+b,axis=1)<=1e-8
    actual=np.any([(q>=lo)&(q<=hi) for lo,hi in intervals],axis=0)
    assert np.array_equal(actual,inside)
    reverse=CurveCarrier(dict(kind='line',endpoints_mm=ends[::-1].tolist()))
    assert np.allclose(domains(reverse,A,b),reference)
    write(dest/'finite_domain_controls.json',dict(line_exact_interval=True,rigid_max_error=max(errors),
        circular_multi_interval_matches_halfspaces=True,line_reversal=True,
        convex_hull_holes_not_resolved=True,formal_contract_pass=False))


def decode_shared(choices,bank):
    """Keep every prediction; share only identical immutable geometry arrays."""
    result=dict(parents=[],invalid_predictions=[],total_prediction_children=len(choices),
        presence_threshold=.5,GT_reads=0,NMS=False,output_union=False,GT_count_used=False)
    cache={}
    for choice in choices:
        c=choice['children'][0];key=(c['profile'],*c['intervals'],*c['fractions'],c['closed'],
            c.get('wrap',False),c['forward'],c['weld_type'])
        if key not in cache:cache[key]=decode_choices([choice],bank)
        base=cache[key]
        for parent in base['parents']:
            p=dict(parent,predicted_parent_query=choice['query'])
            p['segments']=[dict(parent['segments'][0],prediction_id=f'{choice["query"]}:0')]
            result['parents'].append(p)
        for invalid in base['invalid_predictions']:
            result['invalid_predictions'].append(dict(invalid,parent_query=choice['query']))
    # Shared geometry is an allocation optimization, not prediction deduplication.
    return result


def prepare(key,dest):
    start=time.perf_counter();case=load_case(key);g=case['graph'];case['x']._mmap.close()
    source=OUT/f'run384/proposal_{key}';prior=read(source/'prediction_receipt.json')
    assert digest(source/'decoded.json')==prior['decoded_sha256'] and prior['GT_reads']==0
    decoded=read(source/'decoded.json');profiles=sorted({s['profile'] for p in decoded['parents'] for s in p['segments']})
    face_ids=sorted({int(f) for pid in profiles for f in g.profiles[pid].get('face_indices',[])[:2]})
    cohort=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/cohort['input_file'])==cohort['input_sha256']
    xyz=np.load(OUT/cohort['input_file'],mmap_mode='r');faces={};unresolved=[]
    with np.load(OUT/f'run185/source_{key}/surface_support.npz') as support:
        for fid in face_ids:
            surface=case['surfaces'][fid];sk=surface.get('point_ids_key')
            try:
                if sk is None or sk not in support:raise ValueError('No observed face support')
                pts=np.asarray(xyz[support[sk]],float);faces[fid]=face_halfspaces(pts,surface.get('normal',[0,0,0]))
            except (ValueError,QhullError) as error:unresolved.append(dict(face=fid,reason=str(error)[:200]))
    intervals={};records=[]
    for pid in profiles:
        fids=g.profiles[pid].get('face_indices',[])[:2]
        if len(fids)!=2 or any(f not in faces for f in fids):intervals[pid]=None
        else:
            A=np.concatenate([faces[f][0] for f in fids]);b=np.concatenate([faces[f][1] for f in fids])
            intervals[pid]=domains(g.curve(pid),A,b)
        records.append(dict(profile=pid,faces=fids,outer_domains_mm=intervals[pid]))
    choices=[];fallback=[];empty=[]
    for parent in decoded['parents']:
        assert parent['child_count']==1
        segment=parent['segments'][0];pid=segment['profile'];domain=intervals[pid]
        if domain is None:fallback.append(parent);continue
        segment.pop('points',None) # no longer needed for replaced geometry
        if not domain:empty.append(dict(profile=pid,prediction_id=segment['prediction_id']));continue
        curve=g.curve(pid)
        for lo,hi in domain:
            closed=curve.kind=='arc' and curve.arc.is_closed and lo==0 and hi==curve.length
            choices.append(dict(query=len(choices),probability=1.,child_count=1,closed=closed,
                children=[dict(profile=pid,intervals=[0,0],fractions=[lo/curve.length,hi/curve.length],
                    forward=True,closed=closed,wrap=False,weld_type=int(segment['weld_type']=='vertical'),
                    side='UNKNOWN',connection='UNKNOWN')]))
    bank=ContinuousGraphBank(g)
    control=choices[:3]+choices[:3]
    assert decode_shared(control,bank)==decode_choices(control,bank)
    gc.collect();result=decode_shared(choices,bank)
    result['parents'].extend(fallback);result['invalid_predictions'].extend(decoded['invalid_predictions'])
    result['total_prediction_children']+=sum(p['child_count'] for p in fallback)+sum(p['counted_false_children'] for p in decoded['invalid_predictions'])
    result.update(scope='CONVEX_OUTER_DOMAIN_CANDIDATE_DIAGNOSTIC_NOT_FINAL_OUTPUT',GT_reads=0,STEP_reads=0)
    folder=dest/f'proposal_{key}';folder.mkdir(exist_ok=True)
    assert not (folder/'prediction_receipt.json').exists()
    # Stream JSON tokens instead of allocating a second multi-GB string.
    with (folder/'decoded.json').open('w',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    write(folder/'finite_domains.json',dict(records=records,unresolved_faces=unresolved,empty_domain_predictions=empty,
        face_diagnostics=[dict(face=f,**v[2]) for f,v in faces.items()],holes_and_concavities_unresolved=True))
    write(folder/'prediction_receipt.json',dict(status='PURE_INPUT_PROPOSALS_FROZEN_BEFORE_EVALUATION',source=key,
        source_prediction_sha256=prior['decoded_sha256'],decoded_sha256=digest(folder/'decoded.json'),input_sha256=cohort['input_sha256'],
        profiles=len(profiles),computed_faces=len(faces),fallback_parents=len(fallback),empty_domain_parents=len(empty),
        output_children=result['total_prediction_children'],GT_reads=0,STEP_reads=0,model_fits=0,seconds=time.perf_counter()-start,
        convex_outer_support_not_weldable_truth=True,shared_decoder_small_control_exact=True,
        prediction_duplicates_retained=True,streamed_json=True))
    xyz._mmap.close();print('FINITE_DOMAINS_READY',key,'OUTPUT',result['total_prediction_children'],'SECONDS',round(time.perf_counter()-start,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--sources',nargs='+',required=True);ap.add_argument('--resume-source-10',action='store_true');a=ap.parse_args()
    dest=OUT/a.run
    if a.resume_source_10:
        assert a.sources==['10'] and not (dest/'proposal_10/prediction_receipt.json').exists()
        assert read(dest/'serialization_attempt_01.json')['status']=='STOPPED_OWN_SERIALIZATION_RESOURCE_GROWTH'
        for key in ('64','21','BM79A_native_local'):
            rr=read(dest/f'proposal_{key}/prediction_receipt.json')
            assert digest(dest/f'proposal_{key}/decoded.json')==rr['decoded_sha256']
        write(dest/'serialization_resume.json',dict(source_sha256=digest(__file__),sources=a.sources,geometry_rules_changed=False))
    else:dest.mkdir(exist_ok=False)
    try:
        controls(dest)
        if not a.resume_source_10:
            write(dest/'registered_protocol.json',dict(source_sha256=digest(__file__),sources=a.sources,
                purpose='Finite observed-face outer boundary diagnostic; frozen384 presence/category, no fitting',
                STEP_reads=0,GT_reads=0,baseline379_preserved=True,external_consultation_paused=True))
        for key in a.sources:prepare(key,dest);gc.collect()
    except BaseException:write(dest/'failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
