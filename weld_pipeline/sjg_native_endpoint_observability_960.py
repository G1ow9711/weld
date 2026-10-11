"""Input-only third-plane event inventory, then TRAIN endpoint observability audit."""
import argparse
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, utc, check_resources
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier


def arc_plane_queries(curve, normal, offset):
    a=curve.arc;u=a.start_direction;v=np.cross(a.plane_normal,u)
    first=a.radius_mm*float(normal@u);second=a.radius_mm*float(normal@v)
    radius=np.hypot(first,second);value=float(offset-normal@a.center_mm)
    if radius<1e-10 or abs(value)>radius+1e-7:return np.empty(0)
    phi=np.arctan2(second,first);theta=np.arccos(np.clip(value/radius,-1,1))
    angles=np.array([phi-theta,phi+theta]);q=np.mod(np.sign(a.sweep_rad)*angles,2*np.pi)*a.radius_mm
    if a.is_closed:q[np.abs(q-curve.length)<1e-7]=0
    return np.unique(np.round(q[q<=curve.length+1e-7].clip(0,curve.length),9))


def controls():
    from weld_pipeline.contract_curve_geometry import CircularArc
    arc=CircularArc([0,0,0],10,[0,0,1],[1,0,0],2*np.pi,True)
    c=CurveCarrier(dict(kind='arc',arc_parameters=arc.to_parameters()))
    q=arc_plane_queries(c,np.array([1.,0,0]),0.)
    assert len(q)==2 and np.allclose(q,[5*np.pi,15*np.pi])
    assert not len(arc_plane_queries(c,np.array([0.,0,1]),0.))
    assert not len(arc_plane_queries(c,np.array([1.,0,0]),11.))
    return dict(circle_crossings_correct=True,coplanar_not_false_event=True,nonintersecting_rejected=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    write(dest/'registered_protocol.json',dict(stage='NATIVE_ENDPOINT_OBSERVABILITY',source='21',
        scope='TRAIN diagnostic, all input carriers, both supported types',
        STEP_reads=0,targets_read_only_after_all_input_events_frozen=True,
        convex_support_is_not_certified_boundary=True,hole_and_material_ownership_unknown=True,
        source_sha256=digest(__file__),optimizer_updates=0,full_contract_pass=False,start_utc=utc()))
    write(dest/'controls.json',controls());case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='21')
    folder=OUT/'run132/source_21';freeze=read(folder/'input_freeze.json')
    assert digest(OUT/case['input_file'])==case['input_sha256']
    for file in ('surfaces.json','surface_support.npz','profiles.json'):assert digest(folder/file)==freeze['files'][file]
    xyz=np.load(OUT/case['input_file'],mmap_mode='r');faces=read(folder/'surfaces.json');profiles=read(folder/'profiles.json')
    curves=[CurveCarrier(p['representation']) for p in profiles];line_ids=np.array([i for i,c in enumerate(curves) if c.kind=='line'],np.int64)
    arc_ids=[i for i,c in enumerate(curves) if c.kind=='arc'];origin=np.array([curves[i].ends[0] for i in line_ids]);direction=np.array([curves[i].direction for i in line_ids]);length=np.array([curves[i].length for i in line_ids])
    events=[[] for _ in profiles];normals=np.array([f['normal'] for f in faces])
    with np.load(folder/'surface_support.npz') as support:
        for fi,f in enumerate(faces):
            check_resources();n=normals[fi];points=xyz[support[f['point_ids_key']]]
            basis=np.array(f['basis']);center=np.array(f['center']);uv=(points-center)@basis
            if len(points)<3 or np.linalg.matrix_rank(uv-uv.mean(0))<2:continue
            hull=ConvexHull(uv).equations;tree=cKDTree(points)
            denominator=direction@n;eligible=abs(denominator)>1e-6
            qq=np.zeros(len(line_ids));qq[eligible]=(float(f['offset'])-origin[eligible]@n)/denominator[eligible]
            eligible&=(qq>=0)&(qq<=length)
            eligible&=np.array([fi not in profiles[i]['face_indices'] for i in line_ids])
            selected=np.flatnonzero(eligible);records=[(int(line_ids[j]),float(qq[j]),float(abs(denominator[j]))) for j in selected]
            for i in arc_ids:
                if fi in profiles[i]['face_indices']:continue
                for q in arc_plane_queries(curves[i],n,float(f['offset'])):
                    records.append((i,float(q),float(abs(curves[i].tangent(q)@n))))
            if not records:continue
            p=np.array([curves[i].at(q) for i,q,cosine in records]);in_box=np.all((p>=points.min(0)-3)&(p<=points.max(0)+3),axis=1)
            possible=np.flatnonzero(in_box)
            if len(possible):
                projected=(p[possible]-center)@basis
                inside=np.all(projected@hull[:,:2].T+hull[:,2]<=3.,axis=1);possible=possible[inside]
            if len(possible):
                distance=tree.query(p[possible],workers=2)[0]
                for index,nearest in zip(possible,distance):
                    i,q,cosine=records[int(index)];events[i].append(dict(q_mm=q,face=fi,
                        axial_cosine=cosine,nearest_observed_point_mm=float(nearest),finite_convex_support=True,holes_unknown=True))
            if fi%25==0:print('INPUT_THIRD_FACE',fi+1,'/',len(faces),flush=True)
    for values in events:values.sort(key=lambda e:(e['q_mm'],e['face']))
    write(dest/'input_events.json',dict(events=events,profile_sha256=freeze['files']['profiles.json'],STEP_reads=0,target_reads=0))
    event_sha=digest(dest/'input_events.json');print('INPUT_EVENTS_FROZEN',sum(map(len,events)),event_sha,flush=True)
    # Existing TRAIN supervision is read only after input events are immutable.
    sup=read(OUT/'run133/source_21_supervision.json');diagnosis=[]
    for key,values in sup['by_profile'].items():
        index=int(key);available=events[index]
        for e in values:
            for name in ('lo','hi'):
                nearest=sorted(available,key=lambda v:abs(v['q_mm']-e[name]))[:5]
                diagnosis.append(dict(profile=index,original_child_id=e['original_child_id'],endpoint=name,target_q_mm=e[name],
                    nearest_input_events=[dict(v,endpoint_distance_mm=abs(v['q_mm']-e[name])) for v in nearest]))
    write(dest/'receipt.json',dict(status='NATIVE_ENDPOINT_OBSERVABILITY_DIAGNOSED',input_events_sha256=event_sha,
        profiles=len(profiles),event_count=sum(map(len,events)),endpoint_diagnosis=diagnosis,
        not_extraction_accuracy=True,full_contract_pass=False,end_utc=utc()))
    print('ENDPOINT_OBSERVABILITY',diagnosis,flush=True)

if __name__=='__main__':main()
