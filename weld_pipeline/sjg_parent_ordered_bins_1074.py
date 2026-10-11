"""Length-weighted profile encoding; exact cells/anchors remain authoritative."""
import argparse
import gc
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, utc, check_resources, resources, reclaim_own_idle_pages
from weld_pipeline.sjg_hierarchical_features_1029 import IntervalFeatures, positions


def profile_bins(cells, values, transforms, circumference=0., count=16):
    """Integrate piecewise-constant input features, keeping profile orientation."""
    cells = np.asarray(cells, np.float64)
    values = np.asarray(values, np.float64)
    widths = cells[:,1]-cells[:,0]
    prefix = np.concatenate((np.zeros((1, values.shape[1])), np.cumsum(widths[:,None]*values, axis=0)))

    def primitive(q):
        q = np.asarray(q, np.float64)
        if circumference:
            cycle = np.floor(q/circumference)
            q = q-cycle*circumference
        else:
            cycle = np.zeros_like(q)
        i = np.searchsorted(cells[:,0], q, side='right')-1
        j = np.maximum(i, 0)
        result = prefix[j] + np.clip(q-cells[j,0], 0, widths[j])[...,None]*values[j]
        result = np.where((i<0)[...,None], 0., result)
        return result + cycle[...,None]*prefix[-1]

    t = np.asarray(transforms, np.float64)
    q = t[:,0,None] + t[:,1,None]*t[:,2,None]*np.linspace(0., 1., count+1)
    integrated = primitive(q)
    result = np.diff(integrated, axis=1)*t[:,1,None,None]/(t[:,2,None,None]/count)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite profile encoding')
    return result.astype(np.float32)


def controls():
    cell = np.asarray([[0., 1., 1., 0.], [1., 9., 0., 1.], [9., 10., 1., 0.]])
    val = np.asarray([[2.], [10.], [4.]])
    forward = profile_bins(cell, val, [[0.,1.,10.]], count=2)[0,:,0]
    reverse = profile_bins(cell, val, [[10.,-1.,10.]], count=2)[0,:,0]
    assert np.allclose(forward, [8.4, 8.8]) and np.array_equal(forward, reverse[::-1])
    # Split an interval 100 ways: physical weights, not row counts, determine encoding.
    ends = np.linspace(1.,9.,101)
    split = np.r_[cell[:1], np.c_[ends[:-1], ends[1:], np.zeros(100), np.ones(100)], cell[-1:]]
    split_values = np.r_[val[:1], np.full((100,1),10.), val[-1:]]
    assert np.array_equal(profile_bins(split, split_values, [[0.,1.,10.]], count=2)[0,:,0], forward)
    circle = profile_bins([[0.,5.],[5.,10.]], [[2.],[4.]], [[8.,1.,4.],[2.,-1.,4.]], 10., count=2)
    assert np.array_equal(circle[:,:,0], [[4.,2.],[2.,4.]])
    # Gap survives in its own mask even when it occupies only a tiny portion of a bin.
    gap = profile_bins([[0.,1.],[1.,1.000001],[1.000001,2.]], [[0.],[1.],[0.]], [[0.,1.,2.]], count=1)
    assert gap[0,0,0]>0
    return dict(length_not_cell_count_weighted=True, orientation_preserved=True,
                periodic_wrap_preserved=True, micro_gap_not_deleted=True,
                anchors_not_replaced_by_bins=True, optimizer_updates=0)


def prepare(key,run='run253'):
    dest = OUT/run/('source_'+key)
    dest.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    write(dest/'controls.json', controls())
    check_resources()
    db = next(c for c in read(OUT/'run201/all_domains_frozen_before_labels.json')['cases'] if c['source_key']==key)
    assert digest(OUT/'run201'/db['json_file']) == db['json_sha256']
    assert digest(OUT/'run201'/db['mapping_file']) == db['mapping_sha256']
    domain = read(OUT/'run201'/db['json_file'])
    with np.load(OUT/'run201'/db['mapping_file']) as z:
        transform = z['local_to_canonical']; pf = z['profile_family']
    context_binding = next(c for c in read(OUT/'run243/input_freeze.json')['cases'] if c['source']==key)
    context = OUT/'run243'/context_binding['file']
    assert digest(context)==context_binding['sha256']
    context_shape = np.load(context, mmap_mode='r').shape
    assert context_shape == (len(domain['families']),669)
    output = np.lib.format.open_memmap(dest/'ordered_bins.npy', mode='w+', dtype=np.float32, shape=(len(pf),16,188))
    factory = None
    certified = None
    logical_budget_pass = True
    direct_query_count = None
    if key=='21':
        binding = read(OUT/'run224/input_freeze.json')
        for name in ('features.npy','input_domain.npz'):
            assert digest(OUT/'run224'/name)==binding['files'][name]
        with np.load(OUT/'run224/input_domain.npz') as z:
            cells = z['cells']; families = z['family']
        starts = np.searchsorted(families, np.arange(len(domain['families'])+1))
        old = np.load(OUT/'run224/features.npy', mmap_mode='r')
        cell_source = 'run224/input_domain.npz'
        query_count = 173754
    else:
        folder = OUT/'run252'/('source_'+key)
        receipt = read(folder/'receipt.json')
        logical_budget_pass = receipt['query_budget_pass']
        if not logical_budget_pass:
            # Preserve the old logical-population failure. This preprocessing-only
            # path reuses certified identical neighbour sets, with no input removal.
            # It does not silently claim that the old population limit passed.
            assert key=='10'
            certified = read(OUT/'run255/receipt.json')
            assert certified['all_queries_computed_under_actual_KD_limit']
            assert certified['direct_control_neighbor_mismatches']==0
            assert certified['source_input_sha256']==receipt['source_input_sha256']
            for name,expected in certified['files'].items():
                assert digest(OUT/'run255'/name)==expected
        for name, expected in receipt['files'].items():
            assert digest(folder/name)==expected
        starts = np.load(folder/'family_offsets.npy')
        cells = np.memmap(folder/'cells.f64', mode='r', dtype='<f8', shape=(receipt['cells'],4))
        queries = np.load(OUT/'run255/all_query_xyz.npy' if certified else folder/'query_xyz.npy', mmap_mode='r')
        query_count = len(queries)
        case = next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
        assert digest(OUT/case['input_file'])==case['input_sha256']==receipt['source_input_sha256']
        xyz = np.load(OUT/case['input_file'], mmap_mode='r')
        fb = read(OUT/f'run166/source_{key}_freeze.json')
        fp = OUT/'run166'/fb['feature_file']
        assert digest(fp)==fb['feature_sha256']==receipt['point_feature_sha256']
        with np.load(fp) as z:
            point = {k:z[k] for k in ('embedding','normal','boundary')}
        base = OUT/f'run185/source_{key}'
        source_binding = read(base/'input_freeze.json')
        assert digest(base/'profiles.json')==source_binding['files']['profiles.json']
        profiles = read(base/'profiles.json')
        factory = IntervalFeatures(domain, profiles, transform, xyz, point, max_queries=0)
        summaries = np.empty((len(queries),42), np.float32)
        if certified:
            with np.load(OUT/'run255/certified_neighbors.npz') as z:
                neighbor_assignment=z['query_to_representative']; neighbor_table=z['neighbors']
                representative=z['representative_query'];radius=z['radius_mm']
            assert np.all(neighbor_assignment>=0)
            ties=np.flatnonzero(radius==0)
            # k=17 can choose a different subset at a kth-distance tie than k=16.
            # Uncertifiable representatives must use the original exact k=16 call.
            for first in range(0,len(ties),4096):
                rows=ties[first:first+4096]
                _,neighbor_table[rows]=factory.tree.query(queries[representative[rows]],k=16,workers=2)
            direct_query_count=certified['actual_original_point_KD_queries']+len(ties)
            assert direct_query_count<=400000,'Actual original-point searches exceed unchanged cost budget'
            np.save(dest/'tie_corrected_neighbors.npy',neighbor_table)
        for first in range(0,len(queries),4096):
            q=queries[first:first+4096]
            if certified:
                ids=neighbor_table[neighbor_assignment[first:first+len(q)]]
                dist=np.linalg.norm(xyz[ids]-q[:,None],axis=-1)
                order=np.argsort(dist,axis=1,kind='stable')
                ids=np.take_along_axis(ids,order,axis=1);dist=np.take_along_axis(dist,order,axis=1)
            else:
                dist,ids=factory.tree.query(q,k=16,workers=2)
            emb=point['embedding'][ids]; nor=point['normal'][ids]; b=point['boundary'][ids]
            summaries[first:first+len(q)] = np.c_[emb.mean(1),emb.std(1),nor.mean(1),b.mean(1),b.max(1),
                dist[:,0]/50,dist.mean(1)/50,(xyz[ids]-q[:,None]).mean(1)/50]
        if not certified:direct_query_count=len(queries)
        if certified:
            # Numerical feature parity uses independent direct queries, recorded
            # as controls, not as model inference or training operations.
            sample=np.linspace(0,len(queries)-1,2048,dtype=np.int64)
            direct=[]
            for first in range(0,len(sample),512):
                ix=sample[first:first+512];q=queries[ix];d,ids=factory.tree.query(q,k=16,workers=2)
                e=point['embedding'][ids];n=point['normal'][ids];b=point['boundary'][ids]
                direct.append(np.c_[e.mean(1),e.std(1),n.mean(1),b.mean(1),b.max(1),
                    d[:,0]/50,d.mean(1)/50,(xyz[ids]-q[:,None]).mean(1)/50].astype(np.float32))
            delta=np.abs(np.concatenate(direct)-summaries[sample])
            write(dest/'certified_feature_parity.json',dict(direct_control_queries=len(sample),
                feature_count=42,max_absolute_error=float(delta.max()),rtol=2e-6,atol=2e-6,
                old_logical_budget_pass=False,actual_KD_budget_pass=direct_query_count<=400000,
                tie_corrected_representatives=len(ties),actual_original_point_KD_queries=direct_query_count,
                allclose=bool(np.allclose(np.concatenate(direct),summaries[sample],atol=2e-6,rtol=2e-6)),
                optimizer_updates=0))
            assert np.allclose(np.concatenate(direct),summaries[sample],atol=2e-6,rtol=2e-6),float(delta.max())
        lookup = {tuple(float(v) for v in q):i for i,q in enumerate(queries)}
        cell_source = str((folder/'cells.f64').relative_to(OUT))
        interval_output = np.lib.format.open_memmap(dest/'cell_interval_features.npy',mode='w+',dtype=np.float32,shape=(len(cells),186))
        # All original-point summaries are frozen now. batch() must only consult
        # its supplied cache (max_queries=0), so release the no-longer-used tree,
        # point arrays and profile dictionaries before the long encoding loop.
        factory.tree=None;factory.xyz=None;factory.point=None
        del xyz,point,profiles,queries
        if certified:del neighbor_table,neighbor_assignment,representative,radius
        gc.collect();reclaim_own_idle_pages();check_resources()
    covered = np.zeros(len(pf),bool)
    for fid, family in enumerate(domain['families']):
        first,last = int(starts[fid]),int(starts[fid+1])
        part = cells[first:last]
        assert len(part)>0
        if factory is None:
            features = np.c_[old[first:last,:186],part[:,2:]]
        else:
            q = np.c_[part[:,0],(part[:,0]+part[:,1])/2,part[:,1]]
            coords = positions(family,q.ravel()); coords[coords==0]=0
            rows = [lookup[tuple(float(v) for v in p)] for p in coords]
            factory.cache = {(fid,float(v)):summaries[index] for v,index in zip(q.ravel(),rows)}
            interval = factory.batch(np.full(len(part),fid,np.int64),part[:,0],part[:,1])
            assert factory.query_count==0
            interval_output[first:last]=interval
            features = np.c_[interval,part[:,2:]]
        ids = np.asarray(family['profiles'],np.int64)
        assert np.all(pf[ids]==fid) and not covered[ids].any()
        output[ids] = profile_bins(part,features,transform[ids],family['circumference_mm'])
        covered[ids]=True
        if fid%512==0:
            if resources()['available_physical_bytes']<1879048192:
                output.flush()
                if factory is not None:interval_output.flush()
                reclaim_own_idle_pages()
            check_resources()
            print('ORDERED_PROFILE_BINS',key,fid+1,'/',len(domain['families']),'PROFILES',int(covered.sum()),flush=True)
    assert covered.all()
    output.flush()
    if factory is not None:
        interval_output.flush()
    write(dest/'receipt.json',dict(status='ALL_PROFILE_ORDERED_BINS_READY',source=key,profiles=len(pf),
        families=len(domain['families']),shape=list(output.shape),context_file=str(context.relative_to(OUT)),
        context_sha256=context_binding['sha256'],context_shape=list(context_shape),
        bins_sha256=digest(dest/'ordered_bins.npy'),cell_source=cell_source,
        mapping_sha256=db['mapping_sha256'],domain_sha256=db['json_sha256'],
        queries=query_count,actual_original_point_KD_queries=direct_query_count or query_count,
        old_logical_query_budget_pass=logical_budget_pass,
        certified_computational_reuse=certified is not None,
        training_protocol_budget_decision_pending=certified is not None,
        cell_anchors_retained=True,original_XYZ_retained=True,
        target_reads=0,STEP_reads=0,optimizer_updates=0,seconds=time.perf_counter()-start,utc=utc()))
    print('ALL_PROFILE_BINS_DONE',key,len(pf),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--run',default='run253');a=ap.parse_args()
    try:
        prepare(a.source,a.run)
    except BaseException:
        dest=OUT/a.run/('source_'+a.source)
        if dest.exists():write(dest/'failure.json',dict(error=traceback.format_exc(),optimizer_updates=0,utc=utc()))
        raise


if __name__=='__main__':main()
