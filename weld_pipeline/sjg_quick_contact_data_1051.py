"""Input-first, all-family finite-cell data for the real minimal-workpiece trial."""
import argparse,gc,time,shutil,traceback
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_sparse_contact_domain_1048 import build
from weld_pipeline.sjg_projected_boundary_anchors_1050 import ProjectedBoundaryAnchors
from weld_pipeline.sjg_hierarchical_features_1029 import IntervalFeatures
from weld_pipeline.sjg_hierarchical_data_1030 import family_supervision
from weld_pipeline.sjg_sparse_contact_preflight_1049 import capacity

def labels(domain,a,supervised):
    n=len(a['family']);event=np.zeros((n,2,4),np.float32);pos=np.zeros((n,2),np.float64)
    pm=np.zeros((n,2),bool);occ=np.zeros((n,3),np.float32)
    direction=np.zeros(n,np.float32);dm=np.zeros(n,bool);kind=np.full(n,-1,np.int64)
    closure=np.zeros(n,np.float32);cm=np.zeros(n,bool)
    for fid in np.unique(a['family']):
        ids=np.flatnonzero(a['family']==fid);cells=a['cells'][ids,:2];children=supervised.get(int(fid),{})
        events={};segments=[]
        for child,meta in children.items():
            segments.extend((float(l),float(r),meta) for l,r in meta['pieces'])
            if meta['closed']:continue
            for l,r in meta['pieces']:
                for q,t in ((l,0),(r,1)):
                    # A wrapped circle's artificial coordinate origin is not an event.
                    c=domain['families'][int(fid)]['circumference_mm']
                    if c and len(meta['pieces'])==2 and (abs(q)<1e-8 or abs(q-c)<1e-8):continue
                    events.setdefault(round(float(q),8),[]).append((t,meta))
        bycell={}
        for q,e in sorted(events.items()):
            j=int(np.searchsorted(cells[:,0],q,side='right')-1)
            if j<0 and abs(q-cells[0,0])<=1e-8:j=0
            assert j>=0 and q<=cells[j,1]+1e-7
            bycell.setdefault(j,[]).append((q,e))
        for j,i in enumerate(ids):
            lo,hi=cells[j];ev=bycell.get(j,[]);assert len(ev)<=2
            # Unused slots split a cell only computationally; they are not predicted breaks.
            qs=[v[0] for v in ev]
            if len(qs)==0:qs=[lo+(hi-lo)/3,lo+2*(hi-lo)/3]
            elif len(qs)==1:qs=[qs[0],(qs[0]+hi)/2]
            qs=np.clip(qs,lo,hi);pos[i]=(np.asarray(qs)-lo)/max(hi-lo,1e-12)
            for k,(q,rows) in enumerate(ev):
                pm[i,k]=True
                for t,m in rows:event[i,k,t]=1
                event[i,k,2]=float(event[i,k,0]>0 and event[i,k,1]>0)
                starts=[m for t,m in rows if t==0];ends=[m for t,m in rows if t==1]
                event[i,k,3]=float(any(s['parent']==e['parent'] and abs(s['segment']-e['segment'])==1 for s in starts for e in ends))
            cuts=[lo,*qs,hi]
            for k in range(3):
                q=(cuts[k]+cuts[k+1])/2
                owners=[m for l,r,m in segments if l<q<r]
                occ[i,k]=float(bool(owners))
            active=[m for l,r,m in segments if min(hi,r)-max(lo,l)>1e-8]
            if active:
                directions={m['forward'] for m in active};types={m['weld_type'] for m in active};closed={m['closed'] for m in active}
                if len(directions)==1:direction[i]=float(next(iter(directions)));dm[i]=True
                if len(types)==1:kind[i]=next(iter(types))
                if len(closed)==1:closure[i]=float(next(iter(closed)));cm[i]=True
    return dict(event=event,positions=pos,position_mask=pm,occupancy=occ,direction=direction,direction_mask=dm,weld_type=kind,closure=closure,closure_mask=cm)

def prepare(run,key='21'):
    dest=OUT/run;dest.mkdir(exist_ok=False);start=time.perf_counter();check_resources()
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert case['role']=='TRAIN';base=OUT/f'run185/source_{key}';binding=read(base/'input_freeze.json')
    domains={r['source_key']:r for r in read(OUT/'run201/all_domains_frozen_before_labels.json')['cases']};db=domains[key]
    for file,k in ((db['json_file'],'json_sha256'),(db['mapping_file'],'mapping_sha256')):assert digest(OUT/'run201'/file)==db[k]
    for name in ('profiles.json','surfaces.json','surface_support.npz'):assert digest(base/name)==binding['files'][name]
    assert digest(OUT/case['input_file'])==case['input_sha256']==binding['input_sha256']
    domain=read(OUT/'run201'/db['json_file']);profiles=read(base/'profiles.json');faces=read(base/'surfaces.json')
    with np.load(OUT/'run201'/db['mapping_file']) as z:transforms=z['local_to_canonical'];pf=z['profile_family']
    xyz=np.load(OUT/case['input_file'],mmap_mode='r');fb=read(OUT/f'run166/source_{key}_freeze.json')
    fp=OUT/'run166'/fb['feature_file'];assert fb['input_sha256']==case['input_sha256'] and digest(fp)==fb['feature_sha256']
    with np.load(fp) as z:point={k:z[k] for k in ('embedding','normal','boundary')}
    with np.load(base/'surface_support.npz') as support:
        provider=ProjectedBoundaryAnchors(xyz,point['boundary'],faces,support)
        meta,a=build(domain,profiles,transforms,np.arange(len(domain['families'])),provider)
    meta.update(source=key,all_relations_preserved=len(domain['relations']),all_families_preserved=len(domain['families']),learned_root_gates=0)
    np.savez(dest/'input_domain.npz',**a);write(dest/'input_domain.json',meta)
    print('ALL_FAMILY_INPUT',key,'families',len(domain['families']),'cells',len(a['family']),'sharedXYZ',len(a['query_xyz']),flush=True)
    # A new all-relation trial uses bounded batching, no hypothesis deletion.
    # The document supersedes the selected-family construction. Actual costs stay explicit.
    if len(a['family'])>200000 or len(a['query_xyz'])>400000:raise RuntimeError('Whole input exceeds preregistered all-relation memory budget; no truncation')
    factory=IntervalFeatures(domain,profiles,transforms,xyz,point);tree=factory.tree;summaries=[]
    for startq in range(0,len(a['query_xyz']),4096):
        queries=a['query_xyz'][startq:startq+4096];dist,ids=tree.query(queries,k=16,workers=2)
        e=point['embedding'][ids];normal=point['normal'][ids];b=point['boundary'][ids]
        summaries.append(np.c_[e.mean(1),e.std(1),normal.mean(1),b.mean(1),b.max(1),dist[:,0]/50,dist.mean(1)/50,(xyz[ids]-queries[:,None]).mean(1)/50].astype(np.float32))
    summaries=np.concatenate(summaries)
    for fid,(lo,hi,_,_),qid in zip(a['family'],a['cells'],a['query_ids']):
        for q,idx in zip((lo,(lo+hi)/2,hi),qid):factory.cache[int(fid),float(q)]=summaries[idx]
    interval=factory.batch(a['family'],a['cells'][:,0],a['cells'][:,1]);assert factory.query_count==0
    sb=next(r for r in read(OUT/'run206/training_manifest.json')['static_bindings'] if r['source_key']==key)
    assert digest(OUT/'run206'/sb['file'])==sb['sha256']
    with np.load(OUT/'run206'/sb['file']) as z:
        f=a['family'];context=np.c_[z['relation_x'][z['family_relation'][f]],z['family_x'][f]]
    features=np.c_[interval,context,a['cells'][:,2:]].astype(np.float32)
    assert np.isfinite(features).all();np.save(dest/'features.npy',features)
    write(dest/'input_freeze.json',dict(source=key,source_input_sha256=case['input_sha256'],target_reads=0,STEP_reads=0,
        files={name:digest(dest/name) for name in ('features.npy','input_domain.npz','input_domain.json')},
        provenance=dict(domain=db,point_feature_sha256=fb['feature_sha256'],static_sha256=sb['sha256']),
        reused_pure_input_components=['run166 frozen point features','run185 finite surface candidates','run201 joint identities','run206 static graph features'],
        no_old_weld_predictions_as_input=True,input_preparation_seconds=time.perf_counter()-start))
    del point,xyz,factory,tree,context,features,summaries;gc.collect();check_resources()
    assert digest(OUT/case['target_file'])==case['target_sha256'];target=read(OUT/case['target_file'])
    supervised,_=family_supervision(key,domain,pf,transforms,target);report=capacity(domain,a,supervised)
    report.update(G=case['original_G'],covered_G=len(report['covered_ids']))
    write(dest/'TRAIN_capacity.json',report)
    if report['overflow_families'] or report['outside_domains'] or report['covered_G']!=report['G']:raise RuntimeError('Full native single-source event capacity failed')
    y=labels(domain,a,supervised);np.savez(dest/'labels.npz',**y)
    write(dest/'receipt.json',dict(status='QUICK_CONTACT_DATA_READY',source=key,G=case['original_G'],all_G=500,
        token_count=len(a['family']),query_count=len(a['query_xyz']),target_sha256=case['target_sha256'],label_sha256=digest(dest/'labels.npz'),
        seconds=time.perf_counter()-start,full_contract_pass=False,utc=utc()))
    print('REAL_QUICK_DATA_READY',key,report['covered_G'],'/',report['G'],flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    try:prepare(args.run)
    except BaseException:
        write(OUT/args.run/'failure.json',dict(error=traceback.format_exc(),utc=utc()));raise
if __name__=='__main__':main()
