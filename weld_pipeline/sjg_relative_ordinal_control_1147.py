"""Bind frozen seven-case compatible K12 supervision to unchanged relative inputs."""
import argparse,gc,hashlib,shutil,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources,utc
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS,feature_path
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.sjg_stream_profile_geometry_1149 import read_profile_geometry


def intersect(a,b):
    return [(max(lo,lc),min(hi,hc)) for lo,hi in a for lc,hc in b if max(lo,lc)<=min(hi,hc)]


def worker(dest):
    started=time.perf_counter();ready=read(OUT/'run344/receipt.json');assert ready['status']=='SEVEN_RELATIVE_INPUTS_FROZEN'
    old=read(OUT/'run318/receipt.json');assert old['status']=='ORDERED_INTERVAL_SUPERVISION_READY' and old['K']==12
    prior=read(OUT/'run325/tables/receipt.json');tables={};seen={};groups={};conflicts=[];count=0;cases=[];E=0;N=0;G=0;parents=0
    for key in KEYS:
        folder=feature_path(key);receipt=read(folder/'receipt.json')
        assert digest(folder/'features.npy')==receipt['features_sha256']
        tables[key]=np.load(folder/'features.npy',mmap_mode='r')
    for key in KEYS:
        check_resources();x=tables[key];binding=next(v for v in prior['cases'] if v['source']==key)
        labels=OUT/f'run318/source_{key}_count_targets.npz';instancefile=OUT/f'run318/source_{key}_interval_instances.json'
        assert digest(labels)==binding['sha256']['labels'] and digest(instancefile)==binding['sha256']['instances']
        with np.load(labels) as z:y=z['labels'];w=z['weights']
        assert len(x)==len(y)==len(w) and np.all(w[y<0]==0)
        with np.load(OUT/f'run288/source_{key}_alias_map.npz') as z:canonical=z['canonical_fragments']
        geometry=next(g for g in read(OUT/'run285/all_geometry_frozen_before_labels.json')['cases'] if g['source']==key)
        profilefile=OUT/f'run185/source_{key}/profiles.json'
        assert digest(profilefile)==geometry['source_files']['profiles.json']
        assert digest(OUT/'run285'/geometry['geometry_file'])==geometry['geometry_sha256']
        profiles=read_profile_geometry(profilefile)
        with np.load(OUT/'run285'/geometry['geometry_file']) as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
        case=dict(canonical=canonical,graph=graph);assert len(x)==2*len(canonical)
        instances=read(instancefile)['instances'];known=[v for v in instances if v['count_is_known']]
        assert len(known)==binding['endpoint_stop']-binding['endpoint_start']
        row_hash={};local_count=0
        for row in np.flatnonzero(y>=0):
            v=np.array(x[row],copy=True);v[v==0]=0
            h=hashlib.blake2b(v.tobytes(),digest_size=16).digest();label=int(y[row]);previous=seen.get(h)
            if previous is None:seen[h]=(key,int(row),label)
            else:
                pk,pr,pl=previous;assert np.array_equal(tables[pk][pr],x[row]),'Digest collision'
                if pl!=label:
                    count+=1;local_count+=1
                    if len(conflicts)<50:conflicts.append(dict(first=dict(source=pk,row=pr,label=pl),second=dict(source=key,row=int(row),label=label)))
            if label>0:row_hash[int(row)]=h
        owners=set();mapping={}
        for event in known:
            row=int(event['row']);k=int(event['k']);assert 1<=k<=12 and event['count']<=12
            assert int(y[row])==(event['count']-1)*16+event['label']
            fragment=int(case['canonical'][row//2]);curve=case['graph'].curve(case['graph'].profile(fragment))
            width=float(np.diff(case['graph'].spans[fragment,:2])[0]);assert width==event['width_mm']
            value=dict(source=key,row=row,k=k,ratio=event['ratio'],width_mm=width,
                radius_mm=None if curve.kind=='line' else curve.arc.radius_mm,owner=event['owner'])
            groups.setdefault((row_hash[row],k),[]).append(value);owners.add(tuple(event['owner']))
            mapping[row,k]=event
        witnessfile=OUT/f'run318/source_{key}_original_instance_witnesses.json';witness=read(witnessfile)
        assert not witness['collisions'] and not witness['count_unresolved_original_instances']
        for parent in witness['witnesses']:
            for node in parent['instances']:
                event=mapping[node['object_row'],node['k']]
                assert tuple(event['owner'])==tuple(node['owner'])
                oldnode=node['original_node'];fragment=int(case['canonical'][event['row']//2])
                assert fragment==oldnode['fragment'] and event['row']%2==oldnode['side']
                lo,hi=case['graph'].spans[fragment,:2];pred=lo+np.asarray(event['ratio'])*(hi-lo)
                # No new witness search; original interval targets must retain their finite geometry.
                assert np.max(np.abs(pred-np.array([oldnode['lo'],oldnode['hi']])))<=3.
        N+=int((y>=0).sum());E+=len(known);G+=witness['G'];parents+=witness['parents']
        cases.append(dict(source=key,input_rows=len(x),known_rows=int((y>=0).sum()),unknown_rows=int((y<0).sum()),
            endpoint_instances=len(known),G=witness['G'],parents=witness['parents'],classification_conflicts=local_count,
            features_sha256=digest(feature_path(key)/'features.npy'),labels_sha256=digest(labels),instances_sha256=digest(instancefile),
            original_witness_sha256=digest(witnessfile),original_witness_reindex_checked=True))
        del case,graph,profiles,canonical,instances,known,y,w,row_hash,mapping;gc.collect()
        print('RELATIVE_K12_SUPERVISION',key,'COUNT_CONFLICTS',count,'E',E,flush=True)
    assert (N,E,G,parents)==(1134182,13533,363,222)
    endpoint_conflicts=[];endpoint_status='NOT_RUN_CLASSIFICATION_GATE_FAILED'
    if count==0:
        endpoint_status='PASSED'
        for members in groups.values():
            allowed=[[(0.,1.)],[(0.,1.)]]
            for member in members:
                radius=member['radius_mm'];width=member['width_mm']
                if radius is not None and 2*radius<=3:continue
                delta=3/width if radius is None else 2*radius*np.arcsin(3/(2*radius))/width
                period=None if radius is None else 2*np.pi*radius/width
                for end,q in enumerate(member['ratio']):
                    parts=[]
                    for k in ([0] if period is None else [-2,-1,0,1,2]):
                        center=q+(0 if period is None else k*period);lo=max(0.,center-delta);hi=min(1.,center+delta)
                        if lo<=hi:parts.append((lo,hi))
                    allowed[end]=intersect(allowed[end],parts)
            if not all(allowed) or min(a for a,b in allowed[0])>=max(b for a,b in allowed[1]):
                endpoint_status='FAILED';endpoint_conflicts.append(members)
    for x in tables.values():x._mmap.close()
    write(dest/'worker_receipt.json',dict(status='SEVEN_RELATIVE_K12_SUPERVISION_READY' if count==0 and endpoint_status=='PASSED' else 'RELATIVE_K12_CONFLICT_STOP',
        cases=cases,N=N,E=E,G=G,parents=parents,K=12,classification_conflicts=count,classification_examples=conflicts,
        endpoint_feasibility=endpoint_status,endpoint_conflicts=endpoint_conflicts,fit_calls=0,DEV_reads=0,STEP_reads=0,
        seconds=time.perf_counter()-started,no_label_or_weight_changes=True,no_rows_deleted=True,new_capacity_searches=0))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');args=ap.parse_args();dest=OUT/'run342'
    if args.worker:
        try:worker(dest)
        except BaseException:write(dest/'worker_failure.json',dict(error=traceback.format_exc(),fit_calls=0));raise
        return
    dest.mkdir(exist_ok=False);shutil.copy2(__file__,dest/'source_snapshot.py')
    write(dest/'registered_protocol.json',dict(status='PRO336_FULL_RELATIVE_K12_SUPERVISION_CONTROL',N=1134182,E=13533,G=363,parents=222,
        sources=list(KEYS),K=12,wall_seconds=180,fit_calls=0,DEV_reads=0,utc=utc()))
    from weld_pipeline.sjg_owned_watchdog_1101 import supervise
    result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_relative_ordinal_control_1147','--worker'],dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='RELATIVE_K12_CONTROL_INCOMPLETE',watchdog=result));raise RuntimeError('Supervision control incomplete')
    r=read(dest/'worker_receipt.json');r['external_wall_seconds']=result['wall_seconds'];write(dest/'receipt.json',r)
    print('RELATIVE_K12_CONTROL',r['status'],r['classification_conflicts'],r['endpoint_feasibility'],flush=True)


if __name__=='__main__':main()
