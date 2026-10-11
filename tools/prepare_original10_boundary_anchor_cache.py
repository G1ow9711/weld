"""Offline TRAIN input-event cache for endpoint selection; no GT-created events."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_sampled_surface_boundary_events import SampledSurfaceBoundaryEvents
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';DEST=BASE/'original_boundary_anchor_training_v1'
def main():
 DEST.mkdir(exist_ok=False);keys=['10'];write(DEST/'protocol.json',dict(keys=keys,scope='Input boundary events cached on profiles appearing in reliable TRAIN positive query tables, for subsequent endpoint-selector supervision; not runtime profile selection',event_generator_sha256=digest('weld_pipeline/server_sampled_surface_boundary_events.py'),kind='line only for this new branch; existing arc work and final arc requirement retained',event_positions='Input geometry only, no GT endpoints inserted, exact float64 unique sorting, no proximity merging',physical_contact_certified=False,sampling_edge_cutoff='fixed6 times per-face median positive nearest-neighbor spacing, no GT parameter fit',labelled_query_ids_not_deployment_input=True,fit_calls=0,DEV_reads=0))
 for key in keys:
  start=time.perf_counter();dest=DEST/key;dest.mkdir();data=BASE/'positive_query_head_v1'/key;r=read(data/'receipt.json');assert digest(data/'records.json')==r['files']['records.json'];records=read(data/'records.json');source=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime'/('source_'+key);f=source/'run185'/('source_'+key);freeze=read(f/'input_freeze.json');xyzpath=ROOT/'server_rebuild/inputs_v1'/key/'xyz_mm.npy';assert digest(xyzpath)==freeze['input_sha256']
  for name in ['surfaces.json','surface_support.npz','profiles.json']:assert digest(f/name)==freeze['files'][name]
  xyz=np.load(xyzpath,mmap_mode='r');faces=read(f/'surfaces.json');inp=FixedQueries(source,key)
  with np.load(f/'surface_support.npz') as support:generator=SampledSurfaceBoundaryEvents(xyz,faces,support)
  pids=sorted({v['profile'] for v in records if inp.profiles[v['profile']]['representation']['kind']=='line'});offsets=[0];positions=[];feature_indices=[];lengths=[]
  for index,pid in enumerate(pids):
   curve=CurveCarrier(inp.profiles[pid]['representation']);events,_=generator.events(curve);events=np.unique(events);assert np.isfinite(events).all() and np.all((events>=0)&(events<=curve.length));a,b=inp.offsets[pid:pid+2];spans=inp.spans[a:b,:2];local=np.searchsorted(spans[:,1],events,side='right');local=np.minimum(local,len(spans)-1);fragment=a+local;rep=inp.representative[fragment];rows=np.searchsorted(inp.canonical,rep);assert np.array_equal(inp.canonical[rows],rep);positions.append(events);feature_indices.append(rows.astype(np.int64));lengths.append(curve.length);offsets.append(offsets[-1]+len(events))
   if index%50==0:print(key,index,'/',len(pids),'events',offsets[-1],flush=True)
  np.savez(dest/'events.npz',profile_ids=np.asarray(pids,np.int64),offsets=np.asarray(offsets,np.int64),positions_mm=np.concatenate(positions),feature_indices=np.concatenate(feature_indices),profile_length_mm=np.asarray(lengths));write(dest/'face_statistics.json',generator.boundary_statistics);write(dest/'receipt.json',dict(key=key,status='INPUT_LINE_ANCHOR_CACHE_FOR_TRAIN_ONLY',profiles=len(pids),events=offsets[-1],maximum_events_per_profile=int(np.diff(offsets).max()),events_sha256=digest(dest/'events.npz'),query_receipt_sha256=digest(data/'receipt.json'),input_bindings=inp.bindings,input_sha256=freeze['input_sha256'],fit_calls=0,DEV_reads=0,GT_endpoints_in_event_values=False,seconds=time.perf_counter()-start));inp.close();xyz._mmap.close();print('COMPLETE',key,offsets[-1],flush=True)
 write(DEST/'receipt.json',dict(status='ORIGINAL10_TRAIN_LINE_EVENT_CACHE_COMPLETE',fit_calls=0,DEV_reads=0))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
