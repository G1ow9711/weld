"""Pure-point <=5 mm local joint sequences on frozen admitted carriers.

28 channels: two faces x (7 transverse + 4 symmetric longitudinal), observed
face masks, absolute normal dot, arc flag, curvature, closed flag. No positions,
piece/face IDs, GT, STEP or whole-face dimensions enter the feature channels.
"""
import argparse,gc,time,traceback
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_local_cross_section_1233 import face_sites
from weld_pipeline.sjg_relative_geometry_1135 import transformations

SCHEMA='LOCAL_JOINT_SEQUENCE_28_STEP5_V1'
STEP=5.


def local_face(tree,positions,tangent,normal,spacing):
    transverse=face_sites(tree,positions,tangent,normal,spacing)
    query=positions[:,None,:]+tangent[:,None,:]*np.array([-20.,-5.,5.,20.])[None,:,None]
    d=tree.query(query.reshape(-1,3),workers=2)[0].reshape(-1,4)
    along=np.column_stack((np.sort(d[:,[1,2]],axis=1),np.sort(d[:,[0,3]],axis=1)))
    return np.column_stack((transverse,np.log1p(along/max(1.,float(spacing)))))


def ordered_features(x,side):
    if side==0:return np.asarray(x)
    y=np.array(x,copy=True);y[:,:11]=x[:,11:22];y[:,11:22]=x[:,:11]
    y[:,22:24]=x[:,22:24][:,::-1]
    if x.shape[1]==56:
        y[:,52:54]=x[:,54:56];y[:,54:56]=x[:,52:54]
    return y


def controls(dest):
    xx,yy=np.meshgrid(np.arange(-50.,51.,2.),np.arange(0.,51.,2.))
    points=np.column_stack((xx.ravel(),yy.ravel(),np.zeros(xx.size)))
    query=np.array([[-49.,0,0],[0,0,0],[49,0,0]]);t=np.tile([1.,0,0],(3,1));normal=np.array([0.,0,1.])
    ref=local_face(cKDTree(points),query,t,normal,2.)
    assert np.array_equal(ref,local_face(cKDTree(points),query,-t,-normal,2.))
    maximum=0.
    for name,R,shift in transformations():
        value=local_face(cKDTree(points@R.T+shift),query@R.T+shift,t@R.T,normal@R.T,2.)
        delta=float(np.max(np.abs(value-ref)));assert delta<1e-10;maximum=max(maximum,delta)
    x=np.arange(84,dtype=np.float32).reshape(3,28)
    assert np.array_equal(ordered_features(ordered_features(x,1),1),x)
    write(dest/'sequence_controls.json',dict(status='LOCAL_SEQUENCE_CONTROLS_PASSED',rigid_max_error=maximum,
        tangent_and_normal_reversal_exact=True,face_order_involution_exact=True,fit_calls=0,GT_reads=0))


def prepare(key,dest,admission_run='run384',whole_run='run380'):
    start=time.perf_counter();check_resources();case=load_case(key);g=case['graph'];case['x']._mmap.close()
    prior=OUT/f'{admission_run}/proposal_{key}';r=read(prior/'prediction_receipt.json')
    assert digest(prior/'decoded.json')==r['decoded_sha256'] and r['GT_reads']==0
    with np.load(prior/'presence_scores.npz') as z:rows=z['selected_rows'].copy()
    pids=np.unique(rows//2);offsets=[0];queries=[];positions=[];tangents=[];groups=defaultdict(list);normals={}
    for pid in pids:
        pid=int(pid);curve=g.curve(pid);q=curve.grid(STEP);queries.append(q);positions.append(curve.at(q));tangents.append(curve.tangent(q));offsets.append(offsets[-1]+len(q))
        for side,fid in enumerate(g.profiles[pid].get('face_indices',[])[:2]):
            normal=np.asarray(case['surfaces'][fid].get('normal',[0,0,0]),float);norm=np.linalg.norm(normal)
            if norm>1e-10:groups[int(fid)].append((len(offsets)-2,side));normals[fid]=normal/norm
    q=np.concatenate(queries);positions=np.concatenate(positions);tangents=np.concatenate(tangents);offsets=np.array(offsets,np.int64)
    del queries;sites=np.zeros((len(q),2,11),np.float32);known=np.zeros((len(q),2),bool)
    cohort=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/cohort['input_file'])==cohort['input_sha256']
    xyz=np.load(OUT/cohort['input_file'],mmap_mode='r')
    stats={s['face']:s['spacing_median_mm'] for s in read(OUT/f'{whole_run}/source_{key}/receipt.json')['face_statistics']}
    with np.load(OUT/f'run185/source_{key}/surface_support.npz') as support:
        for number,(fid,entries) in enumerate(groups.items()):
            sk=case['surfaces'][fid].get('point_ids_key')
            if sk is None or sk not in support or len(support[sk])<1:continue
            pts=np.asarray(xyz[support[sk]],float);tree=cKDTree(pts)
            indices=np.concatenate([np.arange(offsets[j],offsets[j+1]) for j,side in entries])
            sides=np.concatenate([np.full(offsets[j+1]-offsets[j],side,np.int8) for j,side in entries])
            for first in range(0,len(indices),8192):
                ix=indices[first:first+8192];ss=sides[first:first+8192]
                valid=np.linalg.norm(np.cross(tangents[ix],normals[fid]),axis=1)>1e-10;ix=ix[valid];ss=ss[valid]
                if len(ix):sites[ix,ss]=local_face(tree,positions[ix],tangents[ix],normals[fid],stats.get(fid,1.));known[ix,ss]=True
            del tree,pts
            if number%32==0:print('LOCAL_SEQUENCE',key,'FACES',number+1,'/',len(groups),'SITES',len(q),flush=True)
    features=np.zeros((len(q),28),np.float32);features[:,:22]=sites.reshape(len(q),22);features[:,22:24]=known
    for j,pid in enumerate(pids):
        curve=g.curve(int(pid));faces=g.profiles[int(pid)].get('face_indices',[])[:2]
        dot=abs(float(normals[faces[0]]@normals[faces[1]])) if len(faces)==2 and all(f in normals for f in faces) else 0.
        features[offsets[j]:offsets[j+1],24:]=[dot,curve.kind=='arc',0 if curve.kind=='line' else 1/curve.arc.radius_mm,
            curve.kind=='arc' and curve.arc.is_closed]
    assert np.isfinite(features).all();folder=dest/f'source_{key}';folder.mkdir(exist_ok=False)
    np.save(folder/'features.npy',features);np.savez(folder/'sequence_map.npz',profile_ids=pids,offsets=offsets,q_mm=q,admitted_rows=rows)
    write(folder/'receipt.json',dict(status='DENSE_LOCAL_SEQUENCE_READY',schema=SCHEMA,source=key,profiles=len(pids),
        unique_sites=len(q),ordered_carriers=len(rows),columns=28,maximum_grid_step_mm=STEP,
        features_sha256=digest(folder/'features.npy'),map_sha256=digest(folder/'sequence_map.npz'),
        input_sha256=cohort['input_sha256'],admission_prediction_sha256=r['decoded_sha256'],GT_reads=0,STEP_reads=0,
        fit_calls=0,physical_boundaries_not_inferred=True,seconds=time.perf_counter()-start))
    xyz._mmap.close();print('LOCAL_SEQUENCE_READY',key,'SITES',len(q),'SECONDS',round(time.perf_counter()-start,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--sources',nargs='+',required=True);a=ap.parse_args()
    dest=OUT/a.run;dest.mkdir(exist_ok=False);controls(dest)
    write(dest/'registered_protocol.json',dict(schema=SCHEMA,sources=a.sources,step_mm=STEP,source_sha256=digest(__file__),
        TRAIN=['21','BM79A_native_local','10'],DEV=['64'],GT_reads=0,STEP_reads=0,fit_calls=0,external_consultation_paused=True))
    try:
        for key in a.sources:prepare(key,dest);gc.collect()
    except BaseException:write(dest/'failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
