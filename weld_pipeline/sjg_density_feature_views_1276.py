"""TRAIN-only alternate sampling views of dimensional kNN descriptors.

Candidate geometry, admission, parent labels and the original feature files are
immutable. Only 15 dimensional global-neighborhood channels get a second view.
This is isolated feature augmentation, not a full resampled-input benchmark.
"""
import gc
import shutil
import time
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_global_neighborhood_sequence_1239 import neighborhood,controls

TRAIN=('21','BM79A_native_local','10','25','36')


def prepare():
    check_resources();dest=OUT/'run436';dest.mkdir(exist_ok=False)
    snapshot=dest/'source_snapshot';snapshot.mkdir();bindings={}
    for name in ('sjg_density_feature_views_1276.py','sjg_global_neighborhood_sequence_1239.py'):
        path=ROOT/'weld_pipeline'/name;bindings[name]=digest(path);shutil.copy2(path,snapshot/name)
    write(dest/'registered_protocol.json',dict(status='TRAIN_INPUT_DENSITY_VIEW_PREPARATION',TRAIN=list(TRAIN),DEV=[],
        source_run='run421/sequence56',sequence_columns=list(range(28,43)),density_ratio=.5,
        sampling='Fixed original indices 0,2,4,...',module_sha256=bindings,fit_calls=0,GT_reads=0,STEP_reads=0,
        pure_input_features_only=True,labels_and_candidate_positions_unchanged=True,
        no_extra_independent_workpieces=True,not_full_pipeline_resampling=True,full_contract_pass=False))
    controls(dest);cohort=read(OUT/'run98/local_cohort.json');cases=[]
    for key in TRAIN:
        begin=time.perf_counter();check_resources();print('DENSITY_VIEW_LOAD',key,flush=True)
        source=OUT/f'run421/sequence56/source_{key}';sr=read(source/'receipt.json')
        assert sr['GT_reads']==sr['STEP_reads']==0
        assert digest(source/'features.npy')==sr['features_sha256'] and digest(source/'sequence_map.npz')==sr['map_sha256']
        with np.load(source/'sequence_map.npz') as z:m={k:z[k] for k in z.files}
        binding=next(v for v in cohort['cases'] if v['source_key']==key)
        assert digest(OUT/binding['input_file'])==sr['input_sha256']==binding['input_sha256']
        xyz=np.load(OUT/binding['input_file']);thin=np.ascontiguousarray(xyz[::2])
        original_count=len(xyz);full_tree=cKDTree(xyz);thin_tree=cKDTree(thin)
        case=load_case(key);graph=case['graph'];case['x']._mmap.close()
        base=np.load(source/'features.npy',mmap_mode='r');folder=dest/f'source_{key}';folder.mkdir()
        view=np.lib.format.open_memmap(folder/'dimensional_knn_half.npy',mode='w+',dtype=np.float32,shape=(len(base),15))
        recompute_max=0.;delta_max=0.;delta_sum=np.zeros(15);written=0
        for j,pid in enumerate(m['profile_ids']):
            curve=graph.curve(int(pid));a,b=map(int,m['offsets'][j:j+2]);q=m['q_mm'][a:b]
            for first in range(0,len(q),4096):
                qq=q[first:first+4096];positions=curve.at(qq);tangent=curve.tangent(qq);length=np.full(len(qq),curve.length)
                alternate=neighborhood(thin_tree,thin,positions,tangent,qq,length)[:,:15]
                assert np.isfinite(alternate).all()
                view[a+first:a+first+len(qq)]=alternate
                old=base[a+first:a+first+len(qq),28:43]
                delta=np.abs(alternate-old);delta_max=max(delta_max,float(delta.max()));delta_sum+=delta.sum(0,dtype=np.float64)
                written+=len(qq)
            # Verify original feature semantics on every carrier, up to eight sites.
            ids=np.unique(np.linspace(0,len(q)-1,min(8,len(q)),dtype=int));qq=q[ids]
            ref=neighborhood(full_tree,xyz,curve.at(qq),curve.tangent(qq),qq,np.full(len(qq),curve.length))[:,:15]
            error=float(np.max(np.abs(ref-base[a+ids,28:43])));recompute_max=max(recompute_max,error)
            assert error<3e-5,(key,int(pid),error)
            if (j+1)%100==0 or j+1==len(m['profile_ids']):
                print('DENSITY_VIEW',key,'PROFILES',j+1,'/',len(m['profile_ids']),flush=True)
        assert written==len(base)
        view.flush();view._mmap.close();base._mmap.close()
        shutil.copy2(source/'sequence_map.npz',folder/'sequence_map.npz')
        assert digest(folder/'sequence_map.npz')==sr['map_sha256']
        assert digest(source/'features.npy')==sr['features_sha256']
        result=dict(source=key,unique_sites=written,profiles=len(m['profile_ids']),original_points=original_count,
            alternate_points=len(thin),columns=list(range(28,43)),dtype='float32',shape=[written,15],
            original_features_sha256=sr['features_sha256'],original_input_sha256=sr['input_sha256'],
            alternate_sha256=digest(folder/'dimensional_knn_half.npy'),map_sha256=sr['map_sha256'],
            original_feature_recompute_max_abs=recompute_max,maximum_feature_delta=delta_max,
            mean_absolute_feature_delta=(delta_sum/written).tolist(),seconds=time.perf_counter()-begin,
            GT_reads=0,STEP_reads=0,fit_calls=0,not_final_weld_accuracy=True)
        write(folder/'receipt.json',result);cases.append(result)
        write(dest/'progress.json',dict(completed_cases=cases,total_cases=len(TRAIN)))
        del full_tree,thin_tree,xyz,thin,base,view,case,graph,m,positions,tangent,alternate,old,delta,ref
        gc.collect();print('DENSITY_VIEW_COMPLETE',key,round(result['seconds'],2),flush=True)
    write(dest/'receipt.json',dict(status='FIVE_TRAIN_DENSITY_FEATURE_VIEWS_FROZEN_NO_FIT',cases=cases,
        TRAIN=list(TRAIN),DEV=[],GT_reads=0,STEP_reads=0,fit_calls=0,full_contract_pass=False))


if __name__=='__main__':prepare()
