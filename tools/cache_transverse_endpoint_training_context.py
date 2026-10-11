"""Observed finite-face transverse context at frozen TRAIN predicted endpoints."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_sampled_surface_boundary_events import SampledSurfaceBoundaryEvents
from weld_pipeline.server_transverse_face_context import transverse_endpoint_context
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';OUT=BASE/'transverse_endpoint_training_context_v1'
def main():
 OUT.mkdir(exist_ok=False);controls=read(BASE/'transverse_face_context_controls_v1/receipt.json');assert controls['status']=='PASS' and controls['module_sha256']==digest('weld_pipeline/server_transverse_face_context.py');proto=read(BASE/'predicted_endpoint_attribute_features_v1/protocol.json');keys=proto['keys'];extra={x['key'] for x in read(BASE/'additional_batch02_protocol.json')['cases']};write(OUT/'protocol.json',dict(keys=keys,columns=86,per_endpoint='43: face count then componentwise mean/min/max14 over the two input profile faces',per_face='normal valid, plane distance/1000, abs normal dot tangent, boundary valid, zero crossing, collinear flag, two sign-invariant (has,count log1p,nearest/1000,farthest/1000) tuples',plane_search_band_mm=3.,sampled_boundary_rule='Existing6 times median nearest spacing triangulation cutoff',source_endpoint_model_sha256=proto['endpoint_model_sha256'],source_feature_protocol_sha256=digest(BASE/'predicted_endpoint_attribute_features_v1/protocol.json'),module_sha256=controls['module_sha256'],not_physical_height_hole_or_contact_labels=True,GT_reads=0,fit_calls=0,DEV_reads=0))
 for key in keys:
  started=time.perf_counter();source=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);freeze=read(source/'input_freeze.json');xyzpath=BASE/'inputs'/key/'xyz_mm.npy';assert digest(xyzpath)==freeze['input_sha256']
  for name in ['surfaces.json','surface_support.npz','profiles.json']:assert digest(source/name)==freeze['files'][name]
  base=BASE/'predicted_endpoint_attribute_features_v1'/key;br=read(base/'receipt.json');predpath=BASE/'boundary_provenance_batch02_expanded8_v1'/(key+'_training_event_predictions.npz');assert digest(predpath)==br['prediction_sha256'];pred=np.load(predpath);chosen=pred['chosen_event_ids'];index=np.load(base/'index.npz');assert digest(base/'index.npz')==br['index_sha256'] and np.array_equal(index['query_rows'],pred['query_rows']);cache=BASE/('batch02_boundary_anchor_training_v1' if key in extra else 'sampled_boundary_anchor_training_v1')/key;assert digest(cache/'receipt.json')==br['event_receipt_sha256'];cr=read(cache/'receipt.json');assert digest(cache/'events.npz')==cr['events_sha256'];ev=np.load(cache/'events.npz');offsets=ev['offsets'];positions=ev['positions_mm'];pids=ev['profile_ids'];profiles=read(source/'profiles.json');faces=read(source/'surfaces.json');xyz=np.load(xyzpath,mmap_mode='r')
  with np.load(source/'surface_support.npz') as support:generator=SampledSurfaceBoundaryEvents(xyz,faces,support)
  unique=np.unique(chosen);profile_index=np.searchsorted(offsets,unique,side='right')-1;context=np.empty((len(unique),43),np.float32)
  for j,pi in enumerate(np.unique(profile_index)):
   ix=np.flatnonzero(profile_index==pi);pid=int(pids[pi]);curve=CurveCarrier(profiles[pid]['representation']);assert curve.kind=='line';face_ids=[int(i) for i in profiles[pid].get('face_indices',[])[:2] if 0<=int(i)<len(faces)];points=curve.at(positions[unique[ix]])
   for row,point in zip(ix,points):context[row]=transverse_endpoint_context(point,curve.direction,face_ids,generator)
   if j%200==0:print(key,j,'profiles',len(unique),'unique endpoints',flush=True)
  features=context[np.searchsorted(unique,chosen)].reshape(len(chosen),86);assert np.isfinite(features).all();dest=OUT/key;dest.mkdir();np.save(dest/'features.npy',features);write(dest/'receipt.json',dict(key=key,rows=len(chosen),columns=86,unique_selected_events=len(unique),features_sha256=digest(dest/'features.npy'),source_feature_receipt_sha256=digest(base/'receipt.json'),prediction_sha256=digest(predpath),input_sha256=freeze['input_sha256'],profile_sha256=freeze['files']['profiles.json'],GT_reads=0,fit_calls=0,DEV_reads=0,seconds=time.perf_counter()-started));del generator;xyz._mmap.close();print('COMPLETE',key,len(chosen),flush=True)
 write(OUT/'receipt.json',dict(status='EIGHT_TRAIN_TRANSVERSE_ENDPOINT_INPUT_CONTEXT_READY',fit_calls=0,GT_reads=0,DEV_reads=0))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
