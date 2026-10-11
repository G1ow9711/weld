"""Input-only numerical coalescing; retain group membership and feature extrema."""
from pathlib import Path
import time,shutil
import numpy as np
from weld_pipeline.server_arc_event_groups import coincident_event_groups
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    # Reversal and transitive-chain controls independent of real workpiece labels.
    test=np.array([0.,1e-10,2e-10,1.,1.0000000001,3.]);g,q=coincident_event_groups(test);r,p=coincident_event_groups(3-test[::-1]);assert np.allclose(q,3-p[::-1],atol=1e-14);assert np.array_equal(np.diff(g),np.diff(r)[::-1]);g,q=coincident_event_groups(np.array([0.,.8e-6,1.6e-6]));assert len(q)==3;assert len(coincident_event_groups(np.array([0.,2e-6]))[1])==2
    started=time.perf_counter();key='new_fcaf2af1abee';events=BASE/'curved_probe_vertex_events_v1'/key;features=BASE/'curved_probe_event_features_v1'/key;er=read(events/'receipt.json');fr=read(features/'receipt.json');assert digest(events/'events.npz')==er['events_sha256'];assert digest(features/'features.npy')==fr['feature_sha256'];assert read(features/'protocol.json')['event_receipt_sha256']==digest(events/'receipt.json')
    with np.load(events/'events.npz') as z:oldoffsets=z['offsets'];oldq=z['positions_mm']
    x=np.load(features/'features.npy',mmap_mode='r');offsets=[0];members=[0];positions=[];rows=[]
    for profile,(a,b) in enumerate(zip(oldoffsets[:-1],oldoffsets[1:])):
        groups,q=coincident_event_groups(oldq[a:b]);assert np.all(np.diff(groups)>0);positions.extend(q.tolist());members.extend((groups[1:]+a).tolist());offsets.append(len(positions));rows.append(dict(profile=profile,original_events=int(b-a),groups=len(q)))
    out=BASE/'curved_probe_event_groups_v1'/key;out.mkdir(parents=True,exist_ok=False);needed=len(positions)*83*4*3;assert shutil.disk_usage(BASE).free>=8*1024**3+needed
    write(out/'protocol.json',dict(event_receipt_sha256=digest(events/'receipt.json'),feature_receipt_sha256=digest(features/'receipt.json'),module_sha256=digest('weld_pipeline/server_arc_event_groups.py'),builder_sha256=digest(__file__),tolerance_mm=1e-6,controls='PASS reversal, bounded component span, transitive chain nonmerge, positive gap preservation',scope='Input-only numerical coincidence, no weld minimum length or reference reads. Group center is mean of original event coordinates; all original source rows retained by contiguous membership. Cache mean/min/max of all83 features for later controlled learning.',GT_reads=0))
    members=np.asarray(members,np.int64);group_positions=np.asarray(positions);offsets=np.asarray(offsets,np.int64);np.savez_compressed(out/'events.npz',offsets=offsets,positions_mm=group_positions,member_offsets=members)
    mean=np.lib.format.open_memmap(out/'mean.npy',mode='w+',dtype=np.float32,shape=(len(positions),83));minimum=np.lib.format.open_memmap(out/'min.npy',mode='w+',dtype=np.float32,shape=mean.shape);maximum=np.lib.format.open_memmap(out/'max.npy',mode='w+',dtype=np.float32,shape=mean.shape)
    for i,(a,b) in enumerate(zip(members[:-1],members[1:])):
        assert oldq[b-1]-oldq[a]<=1e-6 or b-a==1
        mean[i]=x[a:b].mean(0,dtype=np.float64);minimum[i]=x[a:b].min(0);maximum[i]=x[a:b].max(0)
    for array in [mean,minimum,maximum]:array.flush();array._mmap.close()
    write(out/'receipt.json',dict(status='INPUT_NUMERICAL_EVENT_GROUPS_FROZEN',original_events=len(oldq),groups=len(positions),profiles=len(rows),rows=rows,protocol_sha256=digest(out/'protocol.json'),files={name:digest(out/name) for name in ['events.npz','mean.npy','min.npy','max.npy']},seconds=time.perf_counter()-started,GT_reads=0,no_automatic_promotion=True));print('groups',len(positions),'from',len(oldq),flush=True)

if __name__=='__main__':main()
