"""Diagnostic exact-neighbour reuse; retains all query positions and candidates.

The old logical unique-XYZ budget is reported separately and is not redefined.
"""
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_hierarchical_features_1029 import positions


def main():
    dest=OUT/'run255';dest.mkdir(exist_ok=False);start=time.perf_counter()
    try:
        check_resources();key='10';folder=OUT/'run252/source_10';receipt=read(folder/'receipt.json')
        for name in ('cells.f64','family_offsets.npy'):
            assert digest(folder/name)==receipt['files'][name]
        domain_path=OUT/'run201/source_10_joint_domain.json'
        assert digest(domain_path)==receipt['domain_sha256']
        domain=read(domain_path)
        offsets=np.load(folder/'family_offsets.npy')
        cells=np.memmap(folder/'cells.f64',dtype='<f8',mode='r',shape=(receipt['cells'],4))
        raw=np.empty((len(cells)*3,3),np.float64)
        for fid,family in enumerate(domain['families']):
            a,b=offsets[fid:fid+2];part=cells[a:b]
            q=np.c_[part[:,0],(part[:,0]+part[:,1])/2,part[:,1]].ravel()
            raw[a*3:b*3]=positions(family,q)
        raw[raw==0]=0
        queries,inverse=np.unique(raw,axis=0,return_inverse=True)
        np.save(dest/'all_query_xyz.npy',queries)
        np.save(dest/'cell_query_ids.npy',inverse.reshape(-1,3))
        del raw,inverse,domain
        case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
        assert digest(OUT/case['input_file'])==case['input_sha256']==receipt['source_input_sha256']
        xyz=np.load(OUT/case['input_file'],mmap_mode='r')
        tree=cKDTree(xyz);query_tree=cKDTree(queries)
        assigned=np.full(len(queries),-1,np.int32)
        representatives=[];neighbor_rows=[];radii=[]
        # Lipschitz distance certificate: same 16-neighbour set for displacement
        # strictly less than (d17-d16)/2. Equal-distance ties get no reuse.
        scale=max(1.,float(np.max(np.abs(xyz))),float(np.max(np.abs(queries))))
        tolerance=128*np.finfo(np.float64).eps*scale
        for i,q in enumerate(queries):
            if assigned[i]>=0:continue
            if len(representatives)>=400000:break
            distance,ids=tree.query(q,k=17)
            radius=max(0.,float((distance[16]-distance[15])/2-tolerance))
            group=query_tree.query_ball_point(q,radius) if radius>0 else [i]
            group=np.asarray(group,np.int64)
            group=group[assigned[group]<0]
            # Strict inequality at the certificate boundary, with a roundoff margin.
            group=group[np.linalg.norm(queries[group]-q,axis=1)<radius] if radius>0 else group
            if i not in group:group=np.r_[group,i]
            assigned[group]=len(representatives)
            representatives.append(i);neighbor_rows.append(ids[:16]);radii.append(radius)
            if len(representatives)%20000==0:
                check_resources()
                print('CERTIFIED_INPUT_QUERY_REUSE',len(representatives),'KD_QUERIES',int((assigned>=0).sum()),'/',len(queries),flush=True)
        np.savez(dest/'certified_neighbors.npz',query_to_representative=assigned,
                 representative_query=np.asarray(representatives,np.int64),
                 neighbors=np.asarray(neighbor_rows,np.int64),radius_mm=np.asarray(radii,np.float64))
        covered=np.flatnonzero(assigned>=0)
        # Independent deterministic direct-neighbour audit; diagnostic queries are
        # recorded separately and never hidden in extraction timing or budgets.
        shared=covered[np.asarray(representatives,np.int64)[assigned[covered]]!=covered]
        selected=shared[np.linspace(0,len(shared)-1,min(2048,len(shared)),dtype=np.int64)] if len(shared) else shared
        mismatches=0;max_distance_error=0.
        for first in range(0,len(selected),512):
            test=selected[first:first+512]
            d,actual=tree.query(queries[test],k=16,workers=2)
            reused=np.asarray(neighbor_rows,np.int64)[assigned[test]]
            mismatches+=int(np.any(np.sort(actual,axis=1)!=np.sort(reused,axis=1),axis=1).sum())
            derived=np.sort(np.linalg.norm(xyz[reused]-queries[test,None],axis=-1),axis=1)
            max_distance_error=max(max_distance_error,float(np.max(np.abs(d-derived),initial=0.)))
        assert mismatches==0,(mismatches,len(selected))
        write(dest/'receipt.json',dict(status='CERTIFIED_QUERY_REUSE_DIAGNOSTIC_COMPLETE',source=key,
            logical_unique_XYZ_queries=len(queries),original_logical_query_budget=400000,
            original_logical_query_budget_pass=len(queries)<=400000,
            actual_original_point_KD_queries=len(representatives),assigned_queries=len(covered),
            unassigned_queries=int((assigned<0).sum()),shared_query_count=len(shared),
            all_queries_computed_under_actual_KD_limit=bool((assigned>=0).all()),
            numerical_margin_mm=tolerance,direct_control_queries=len(selected),
            direct_control_neighbor_mismatches=mismatches,direct_control_distance_max_error_mm=max_distance_error,
            changes_query_coordinates=False,changes_anchors=False,changes_candidate_count=False,
            model_feature_integration=False,old_logical_budget_not_redefined=True,
            target_reads=0,STEP_reads=0,optimizer_updates=0,
            files={name:digest(dest/name) for name in ('all_query_xyz.npy','cell_query_ids.npy','certified_neighbors.npz')},
            source_input_sha256=case['input_sha256'],seconds=time.perf_counter()-start,utc=utc()))
        print('CERTIFIED_QUERY_REUSE_COMPLETE',len(queries),len(representatives),len(shared),flush=True)
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),optimizer_updates=0,utc=utc()));raise


if __name__=='__main__':main()
