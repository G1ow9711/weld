"""TRAIN coverage probe: union existing events with observed point projections."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_raw_tube_endpoint_events import RawTubeEndpointEvents
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';OUT=BASE/'raw_tube_anchor_training_probe_v1'
def main():
 OUT.mkdir(exist_ok=False);controls=read(BASE/'raw_tube_event_controls_v1/receipt.json');assert controls['status']=='PASS' and controls['module_sha256']==digest('weld_pipeline/server_raw_tube_endpoint_events.py');keys=['new_034230635b7a','new_181887f85097'];write(OUT/'protocol.json',dict(keys=keys,radius_mm=3.,source='All raw XYZ projections within fixed finite line tube union original input events',runtime_profile_selection=False,profile_scope='Previously registered reliable TRAIN query profiles; diagnostic conditional capacity only',GT_event_insertion=False,STEP_reads=0,DEV_reads=0,module_sha256=controls['module_sha256']))
 for key in keys:
  started=time.perf_counter();cache=BASE/'sampled_boundary_anchor_training_v1'/key;r=read(cache/'receipt.json');assert digest(cache/'events.npz')==r['events_sha256'];z=np.load(cache/'events.npz');pids=z['profile_ids'];old_offsets=z['offsets'];old_q=z['positions_mm'];source=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);freeze=read(source/'input_freeze.json');assert digest(source/'profiles.json')==freeze['files']['profiles.json'];profiles=read(source/'profiles.json');xyzpath=BASE/'inputs'/key/'xyz_mm.npy';assert digest(xyzpath)==freeze['input_sha256'];xyz=np.load(xyzpath,mmap_mode='r');generator=RawTubeEndpointEvents(xyz);chunks=[];offsets=[0];rawcounts=[]
  for i,pid in enumerate(pids):
   curve=CurveCarrier(profiles[int(pid)]['representation']);q,_=generator.events(curve);a,b=old_offsets[i:i+2];union=np.union1d(old_q[a:b],q);chunks.append(union);offsets.append(offsets[-1]+len(union));rawcounts.append(len(q))
   if i%100==0:print(key,i,'/',len(pids),'union_events',offsets[-1],flush=True)
  dest=OUT/key;dest.mkdir();np.savez(dest/'events.npz',profile_ids=pids,offsets=np.asarray(offsets,np.int64),positions_mm=np.concatenate(chunks));write(dest/'receipt.json',dict(key=key,events_sha256=digest(dest/'events.npz'),query_receipt_sha256=r['query_receipt_sha256'],original_event_receipt_sha256=digest(cache/'receipt.json'),profiles=len(pids),events=offsets[-1],old_events=len(old_q),raw_projection_events=int(sum(rawcounts)),max_events_per_profile=int(np.diff(offsets).max()),input_sha256=freeze['input_sha256'],GT_endpoint_reads=0,GT_event_insertion=False,fit_calls=0,DEV_reads=0,seconds=time.perf_counter()-started));xyz._mmap.close();print('COMPLETE',key,offsets[-1],flush=True)
 write(OUT/'receipt.json',dict(status='TWO_TRAIN_INPUT_EVENT_PROBES_READY',fit_calls=0,DEV_reads=0))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
