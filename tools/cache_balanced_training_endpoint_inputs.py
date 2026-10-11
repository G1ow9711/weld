"""Input event/provenance cache for both frozen detectors; share geometry computation."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from tools.cache_boundary_event_provenance import describe,COLUMNS
from weld_pipeline.server_sampled_surface_boundary_events import SampledSurfaceBoundaryEvents
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';OUT=BASE/'reference_detector_training_endpoint_cache_followup'
def main():
 OUT.mkdir(exist_ok=False)
 for key in read(BASE/'reference_query_task_v1/protocol.json')['training_keys']:
  start=time.perf_counter();source=BASE/'frozen_features/legacy_runtime'/('source_'+key);inp=FixedQueries(source,key);selected={};pools={};freezes={}
  for arm in ['balanced']:
   folder=BASE/('reference_detector_'+arm+'_full_output_v1')/key;r=read(folder/'prediction_freeze.json');assert r['input_bindings']==inp.bindings and digest(folder/'predictions.npz')==r['predictions_sha256'];z=np.load(folder/'predictions.npz');ids=np.flatnonzero(z['valid']&z['line_eligible']&(z['reference_occupancy_score']>=.5));assert len(ids)==r['selected_queries'];selected[arm]=ids;pools[arm]=np.unique(inp.pid[ids//4]);freezes[arm]=digest(folder/'prediction_freeze.json')
  f=source/'run185'/('source_'+key);r=read(f/'input_freeze.json');xyzpath=BASE/'inputs'/key/'xyz_mm.npy';assert digest(xyzpath)==r['input_sha256']
  for name in ['surfaces.json','surface_support.npz','profiles.json']:assert digest(f/name)==r['files'][name]
  xyz=np.load(xyzpath,mmap_mode='r');faces=read(f/'surfaces.json')
  with np.load(f/'surface_support.npz') as support:generator=SampledSurfaceBoundaryEvents(xyz,faces,support);counts=np.array([len(support[x['point_ids_key']]) if x.get('point_ids_key') in support else 0 for x in faces])
  bank={};union=np.unique(np.concatenate(list(pools.values())))
  for index,pid in enumerate(union):
   curve=CurveCarrier(inp.profiles[pid]['representation']);assert curve.kind=='line';events,provenance=describe(generator,curve,counts);a,b=inp.offsets[pid:pid+2];local=np.minimum(np.searchsorted(inp.spans[a:b,1],events,side='right'),b-a-1);rep=inp.representative[a+local];rows=np.searchsorted(inp.canonical,rep);assert np.array_equal(inp.canonical[rows],rep);bank[int(pid)]=(events,rows,curve.length,provenance)
   if index%100==0:print(key,index,'/',len(union),flush=True)
  for arm in ['balanced']:
   dest=BASE/('reference_detector_'+arm+'_event_cache_v1')/key;dest.mkdir(parents=True,exist_ok=False);items=[bank[int(pid)] for pid in pools[arm]];offsets=np.r_[0,np.cumsum([len(x[0]) for x in items])];np.savez(dest/'events.npz',selected_query_ids=selected[arm],profile_ids=pools[arm],offsets=offsets,positions_mm=np.concatenate([x[0] for x in items]) if items else np.empty(0),feature_indices=np.concatenate([x[1] for x in items]) if items else np.empty(0,np.int64),profile_length_mm=np.array([x[2] for x in items]));np.save(dest/'provenance.npy',np.concatenate([x[3] for x in items]) if items else np.empty((0,12),np.float32));write(dest/'receipt.json',dict(key=key,arm=arm,selected_queries=len(selected[arm]),profiles=len(items),events=int(offsets[-1]),events_sha256=digest(dest/'events.npz'),features_sha256=digest(dest/'provenance.npy'),columns=COLUMNS,detector_freeze_sha256=freezes[arm],input_bindings=inp.bindings,input_sha256=r['input_sha256'],event_generator_sha256=digest('weld_pipeline/server_sampled_surface_boundary_events.py'),provenance_implementation_sha256=digest('tools/cache_boundary_event_provenance.py'),GT_reads=0,STEP_reads=0,fit_calls=0,source_sha256=digest(__file__),shared_geometry_computation_only=True,seconds=time.perf_counter()-start))
  inp.close();xyz._mmap.close();print('COMPLETE',key,flush=True)
 write(OUT/'receipt.json',dict(status='BALANCED_DETECTOR_SIX_TRAIN_INPUT_ENDPOINT_CACHES_COMPLETE',GT_reads=0,STEP_reads=0))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
