"""RoboWeldAR-inspired finite surface-pair evidence, independent implementation.

Reference: ikh-innovation/roboweldar-weld-seam-detection, commit
8cf0e93a6cc24a1372cdad7234eb4e0f8b563dfa, pipeline.py points_intersection.
Replace box membership intersection with continuous distances to both observed
surface supports. No point removal, candidate filtering, or CAD/label access.
"""
import argparse
import gc
import time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, utc, check_resources, reclaim_own_idle_pages
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_parent_input_1078 import ParentInput

FEATURE_RUN='run274'
CHANNELS=('face_a_distance_log1p_over_10mm','face_b_distance_log1p_over_10mm',
          'joint_support_exp_max_distance_over_10mm','nearest_face_points_distance_log1p_over_10mm',
          'face_pair_support_known')


def encode_pair(distances, nearest, known):
    d=np.asarray(distances,np.float64); q=np.asarray(nearest,np.float64)
    valid=np.asarray(known,bool).all(-1)
    result=np.zeros((*d.shape[:-1],5),np.float32)
    result[...,0:2]=np.where(valid[...,None],np.log1p(d/10.),0.)
    result[...,2]=np.where(valid,np.exp(-np.square(d.max(-1)/10.)),0.)
    result[...,3]=np.where(valid,np.log1p(np.linalg.norm(q[...,0,:]-q[...,1,:],axis=-1)/10.),0.)
    result[...,4]=valid
    if not np.isfinite(result).all():raise ValueError('Nonfinite pair evidence')
    return result


def controls():
    # Both surfaces must support the point; one nearby surface alone is insufficient.
    d=np.array([[[0.,0.],[0.,100.],[0.,0.]]])
    q=np.zeros((1,3,2,3));known=np.ones((1,3,2),bool);known[0,2,1]=False
    out=encode_pair(d,q,known)
    assert out[0,0,2]==1 and out[0,1,2]<1e-20 and np.all(out[0,2]==0)
    swapped=encode_pair(d[...,::-1],q[...,::-1,:],known[...,::-1])
    assert np.array_equal(out[...,0],swapped[...,1]) and np.array_equal(out[...,2:],swapped[...,2:])
    moved=encode_pair(d,q+np.array([1000.,-300.,40.]),known)
    assert np.array_equal(out,moved)
    return dict(status='PAIR_SUPPORT_CONTROLS_PASSED',both_faces_required=True,unknown_mask=True,
                ordered_sides_preserved=True,optimizer_updates=0)


def prepare(key):
    from scipy.spatial import cKDTree
    dest=OUT/FEATURE_RUN/('source_'+key);dest.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();check_resources()
    base=OUT/f'run185/source_{key}';binding=read(base/'input_freeze.json')
    for name in ('profiles.json','surfaces.json','surface_support.npz'):
        assert digest(base/name)==binding['files'][name]
    profiles=read(base/'profiles.json');surfaces=read(base/'surfaces.json')
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/case['input_file'])==case['input_sha256']
    xyz=np.load(OUT/case['input_file'],mmap_mode='r')
    # Midpoints coincide with the existing sixteen ordered encoding bins. This
    # samples evidence only; complete anchors, intervals and candidates survive.
    queries=np.empty((len(profiles),16,3),np.float64)
    groups={}
    for pid,p in enumerate(profiles):
        curve=CurveCarrier(p['representation'])
        queries[pid]=curve.at((np.arange(16)+.5)/16*curve.length)
        faces=p.get('face_indices',[])
        for side,face in enumerate(faces[:2]):groups.setdefault(int(face),[]).append((pid,side))
    distances=np.zeros((len(profiles),16,2),np.float64)
    nearest=np.zeros((len(profiles),16,2,3),np.float64)
    known=np.zeros((len(profiles),16,2),bool)
    query_count=0
    with np.load(base/'surface_support.npz') as supports:
        for number,(face,items) in enumerate(groups.items()):
            support_key=surfaces[face].get('point_ids_key')
            if support_key not in supports:continue
            ids=supports[support_key]
            if not len(ids):continue
            points=np.asarray(xyz[ids],np.float64);tree=cKDTree(points)
            for first in range(0,len(items),1024):
                batch=np.asarray(items[first:first+1024],np.int64);pi,side=batch.T
                query=queries[pi].reshape(-1,3)
                unique,inverse=np.unique(query,axis=0,return_inverse=True)
                d,ix=tree.query(unique,k=1,workers=2);query_count+=len(unique)
                distances[pi,:,side]=d[inverse].reshape(-1,16)
                nearest[pi,:,side]=points[ix[inverse]].reshape(-1,16,3)
                known[pi,:,side]=True
            del points,tree,ids
            if number%32==0:
                gc.collect();reclaim_own_idle_pages();check_resources()
                print('ROBOWELDAR_PAIR_SUPPORT',key,number+1,'/',len(groups),flush=True)
    result=encode_pair(distances,nearest,known)
    np.save(dest/'pair_bins.npy',result)
    assert digest(OUT/case['input_file'])==case['input_sha256']
    write(dest/'receipt.json',dict(status='ALL_PROFILE_PAIR_SUPPORT_READY',source=key,
        source_input_sha256=case['input_sha256'],profiles=len(profiles),shape=list(result.shape),
        channels=CHANNELS,scale_mm=10.,scale_is_feature_not_acceptance_threshold=True,
        known_fraction=float(result[...,4].mean()),unique_queries_per_face_batch=query_count,
        candidate_count_before=len(profiles),candidate_count_after=len(profiles),
        geometry_changed=False,STEP_reads=0,target_reads=0,optimizer_updates=0,
        upstream_source_commit='8cf0e93a6cc24a1372cdad7234eb4e0f8b563dfa',
        upstream_method='surface pair joint point support; continuous adaptation, not original VoteNet execution',
        files={name:digest(base/name) for name in ('profiles.json','surfaces.json','surface_support.npz')},
        pair_bins_sha256=digest(dest/'pair_bins.npy'),seconds=time.perf_counter()-start,utc=utc()))
    print('PAIR_SUPPORT_READY',key,len(profiles),'SECONDS',round(time.perf_counter()-start,3),flush=True)


class PairParentInput(ParentInput):
    def __init__(self,key,normalization=None,verify=True):
        super().__init__(key,normalization,verify)
        self.pair_path=OUT/FEATURE_RUN/('source_'+key)
        self.pair_receipt=read(self.pair_path/'receipt.json')
        assert self.pair_receipt['status']=='ALL_PROFILE_PAIR_SUPPORT_READY'
        assert digest(self.pair_path/'pair_bins.npy')==self.pair_receipt['pair_bins_sha256']
        self.pair_bins=np.load(self.pair_path/'pair_bins.npy',mmap_mode='r')
        assert self.pair_bins.shape==(len(self),16,5)

    def arrays(self,indices):
        c,b=super().arrays(indices)
        return c,np.concatenate((b,self.pair_bins[indices]),axis=-1)

    def binding(self):
        result=super().binding()
        result.update(pair_receipt_sha256=digest(self.pair_path/'receipt.json'),
                      pair_bins_sha256=self.pair_receipt['pair_bins_sha256'],combined_shape=[len(self),16,247])
        return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source');a=ap.parse_args()
    if a.source:prepare(a.source)
    else:
        dest=OUT/FEATURE_RUN;dest.mkdir(exist_ok=False)
        write(dest/'controls.json',controls())
        for key in ('21','BM79A_native_local','10'):prepare(key)
