"""128 CPU geometric features for every frozen finite joint fragment.

RoboWeldAR-inspired two-face point support: 16 ordered positions x 5 channels.
No target/STEP access, candidate deletion, coordinate move, or model import.
Saved rows use unresolved side A; side B is an exact role permutation on load.
"""
import argparse
from collections import defaultdict
import gc
import time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph,specification


def side_rows(base, fragments, sides):
    result=np.array(base[fragments],dtype=np.float32,copy=True)
    swap=np.asarray(sides)==1
    if np.any(swap):
        selected=result[swap].copy();bins=selected[:,:80].reshape(-1,16,5)
        bins[:,:,[0,1]]=bins[:,:,[1,0]]
        selected[:,99:102],selected[:,102:105]=selected[:,102:105].copy(),selected[:,99:102].copy()
        selected[:,110]=-1
        selected[:,117],selected[:,118]=selected[:,118].copy(),selected[:,117].copy()
        selected[:,122]=0;selected[:,123]=1
        result[swap]=selected
    return result


def prepare(key,run,geometry_run='run285'):
    from scipy.spatial import cKDTree
    import sys
    assert 'torch' not in sys.modules
    t=time.perf_counter();check_resources()
    dest=OUT/run/f'source_{key}';dest.mkdir(parents=True,exist_ok=False)
    row=next(r for r in read(OUT/geometry_run/'all_geometry_frozen_before_labels.json')['cases'] if r['source']==key)
    base=OUT/f'run185/source_{key}'
    for name in ('profiles.json','surfaces.json','surface_support.npz'):
        assert digest(base/name)==row['source_files'][name]
    assert digest(OUT/geometry_run/row['geometry_file'])==row['geometry_sha256']
    profiles=read(base/'profiles.json');surfaces=read(base/'surfaces.json')
    with np.load(OUT/geometry_run/row['geometry_file']) as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/case['input_file'])==case['input_sha256']
    xyz=np.load(OUT/case['input_file'],mmap_mode='r')
    n=len(graph.spans);features=np.lib.format.open_memmap(dest/'object_features.npy',mode='w+',dtype=np.float32,shape=(n,128))
    features[:]=0;queries=np.empty((n,16,3),np.float64);groups=defaultdict(list)
    valid=np.zeros((n,2),bool);counts=np.diff(graph.offsets)
    for pid,p in enumerate(profiles):
        curve=graph.curve(pid);start,stop=graph.offsets[pid:pid+2]
        faces=p.get('face_indices',[])[:2];normals=[]
        for side in range(2):
            if side<len(faces):
                face=int(faces[side]);raw=np.asarray(surfaces[face].get('normal',[0,0,0]),np.float64)
                norm=float(np.linalg.norm(raw));normals.append(raw/norm if norm else np.zeros(3))
                groups[face].extend((int(i),side) for i in range(start,stop))
            else:normals.append(np.zeros(3))
        for fragment in range(start,stop):
            lo,hi,supported,void=graph.spans[fragment];length=hi-lo
            q=lo+(np.arange(16)+.5)/16*length;queries[fragment]=curve.at(q)
            points=curve.at(np.array([lo,hi,(lo+hi)/2]));tangents=curve.tangent(np.array([lo,hi]))
            geometry=[np.log1p(length),np.log1p(curve.length),lo/curve.length,hi/curve.length,
                      *(points.reshape(-1)/1000),*tangents.reshape(-1),*normals[0],*normals[1],
                      float(normals[0]@normals[1]),0 if curve.kind=='line' else 1/curve.arc.radius_mm,
                      0 if curve.kind=='line' else np.sign(curve.arc.sweep_rad)*length/curve.arc.radius_mm,
                      supported,void,1.,length/curve.length]
            assert len(geometry)==32
            features[fragment,80:112]=geometry
            features[fragment,112:128]=[curve.kind=='line',curve.kind=='arc',
                curve.kind=='arc' and curve.arc.is_closed,supported,void,0,0,0,0,0,1,0,
                lo==0,hi==curve.length,1,1]
        if pid%8192==0:check_resources()
    distances=np.zeros((n,16,2),np.float64);nearest=np.zeros((n,16,2,3),np.float64)
    with np.load(base/'surface_support.npz') as supports:
        for index,(face,entries) in enumerate(groups.items()):
            support_key=surfaces[face].get('point_ids_key')
            if support_key is None or support_key not in supports:continue
            ids=supports[support_key]
            if not len(ids):continue
            points=np.asarray(xyz[ids],np.float64);tree=cKDTree(points)
            for first in range(0,len(entries),1024):
                batch=np.asarray(entries[first:first+1024],np.int64);fragments,sides=batch.T
                query=queries[fragments].reshape(-1,3);unique,inverse=np.unique(query,axis=0,return_inverse=True)
                d,ix=tree.query(unique,k=1,workers=2)
                distances[fragments,:,sides]=d[inverse].reshape(-1,16)
                nearest[fragments,:,sides]=points[ix[inverse]].reshape(-1,16,3)
                valid[fragments,sides]=True
            del points,tree,ids
            if index%64==0:
                features.flush();gc.collect();check_resources()
                print('FRAGMENT_PAIR_FEATURES',key,index+1,'/',len(groups),flush=True)
    for first in range(0,n,4096):
        last=min(n,first+4096);d=distances[first:last];v=valid[first:last].all(-1)
        bins=features[first:last,:80].reshape(-1,16,5)
        bins[:,:,0:2]=np.where(v[:,None,None],np.log1p(d/10.),0.)
        bins[:,:,2]=np.where(v[:,None],np.exp(-np.square(d.max(-1)/10.)),0.)
        separation=np.linalg.norm(nearest[first:last,:,0]-nearest[first:last,:,1],axis=-1)
        bins[:,:,3]=np.where(v[:,None],np.log1p(separation/10.),0.)
        bins[:,:,4]=v[:,None]
        features[first:last,117:119]=valid[first:last]
        features[first:last,119]=v
    assert np.isfinite(features).all()
    # Exact role permutation is involutive on known/unknown support channels.
    sample=np.arange(min(32,n));a=np.asarray(features[sample]);b=side_rows(features,sample,np.ones(len(sample)))
    assert np.array_equal(a[:,:80].reshape(-1,16,5)[:,:,0],b[:,:80].reshape(-1,16,5)[:,:,1])
    assert np.array_equal(a[:,99:102],b[:,102:105])
    features.flush();del features
    write(dest/'receipt.json',dict(status='ALL_FINITE_FRAGMENT_OBJECT_FEATURES_READY',source=key,
        shape=[n,128],explicit_side_states=2,profiles=len(profiles),source_input_sha256=case['input_sha256'],
        geometry_sha256=row['geometry_sha256'],features_sha256=digest(dest/'object_features.npy'),
        specification=specification(),feature_model='ordered original two-face point support + finite analytic geometry',
        raw_xyz_moved=False,candidate_deletion=False,STEP_reads=0,target_reads=0,Torch_imported='torch' in sys.modules,
        seconds=time.perf_counter()-t,utc=utc(),extraction_correct_rate=None,extraction_precision_mm=None))
    print('FRAGMENT_FEATURES_READY',key,n,'SECONDS',time.perf_counter()-t,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run287');args=ap.parse_args()
    for key in ('21','BM79A_native_local','10'):prepare(key,args.run);gc.collect()


if __name__=='__main__':main()
