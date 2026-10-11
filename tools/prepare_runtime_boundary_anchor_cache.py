"""Input-only boundary events for profiles selected by a frozen query detector."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.server_sampled_surface_boundary_events import SampledSurfaceBoundaryEvents
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';OUT=BASE/'runtime_boundary_anchor_cache_v1'
def main():
 OUT.mkdir(exist_ok=False)
 for key in ('new_cfc732e5f068','new_ceeae49d6777'):
  start=time.perf_counter();dest=OUT/key;dest.mkdir();prediction=BASE/'joint_reference_full_output'/key;freeze=read(prediction/'prediction_freeze.json');assert digest(prediction/'predictions.npz')==freeze['predictions_sha256'];z=np.load(prediction/'predictions.npz');ids=np.flatnonzero(z['valid']&z['line_eligible']&(z['reference_occupancy_score']>=.5));assert len(ids)==freeze['selected_queries'];source=BASE/'frozen_features/legacy_runtime'/('source_'+key);f=source/'run185'/('source_'+key);binding=read(f/'input_freeze.json');xyzpath=BASE/'inputs'/key/'xyz_mm.npy';assert digest(xyzpath)==binding['input_sha256']
  for name in ['surfaces.json','surface_support.npz','profiles.json']:assert digest(f/name)==binding['files'][name]
  xyz=np.load(xyzpath,mmap_mode='r');inp=FixedQueries(source,key)
  with np.load(f/'surface_support.npz') as support:generator=SampledSurfaceBoundaryEvents(xyz,read(f/'surfaces.json'),support)
  pids=np.unique(inp.pid[ids//4]);offsets=[0];positions=[];feature_indices=[];lengths=[]
  for index,pid in enumerate(pids):
   curve=CurveCarrier(inp.profiles[pid]['representation']);assert curve.kind=='line';events,_=generator.events(curve);events=np.unique(events);assert np.isfinite(events).all() and np.all((events>=0)&(events<=curve.length));a,b=inp.offsets[pid:pid+2];spans=inp.spans[a:b,:2];local=np.minimum(np.searchsorted(spans[:,1],events,side='right'),len(spans)-1);rep=inp.representative[a+local];rows=np.searchsorted(inp.canonical,rep);assert np.array_equal(inp.canonical[rows],rep);positions.append(events);feature_indices.append(rows);lengths.append(curve.length);offsets.append(offsets[-1]+len(events))
   if index%50==0:print(key,index,'/',len(pids),'events',offsets[-1],flush=True)
  np.savez(dest/'events.npz',selected_query_ids=ids,profile_ids=pids,offsets=np.asarray(offsets,np.int64),positions_mm=np.concatenate(positions) if positions else np.array([],float),feature_indices=np.concatenate(feature_indices) if feature_indices else np.array([],np.int64),profile_length_mm=np.asarray(lengths));write(dest/'receipt.json',dict(key=key,status='INPUT_ONLY_FROZEN_DETECTOR_PROFILE_EVENT_CACHE',selected_queries=len(ids),profiles=len(pids),events=offsets[-1],events_sha256=digest(dest/'events.npz'),detector_freeze_sha256=digest(prediction/'prediction_freeze.json'),input_bindings=inp.bindings,input_sha256=binding['input_sha256'],event_generator_sha256=digest('weld_pipeline/server_sampled_surface_boundary_events.py'),GT_reads=0,STEP_reads=0,fit_calls=0,source_sha256=digest(__file__),seconds=time.perf_counter()-start));inp.close();xyz._mmap.close();print('COMPLETE',key,flush=True)
 write(OUT/'receipt.json',dict(status='TWO_RUNTIME_INPUT_EVENT_CACHES_COMPLETE',GT_reads=0,STEP_reads=0,fit_calls=0))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
