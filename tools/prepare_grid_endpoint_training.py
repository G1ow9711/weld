"""Freeze input grids, then bind owner-separated TRAIN endpoint supervision."""
from pathlib import Path
import numpy as np
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.server_finite_carrier_grid import finite_carrier_grid
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 out=BASE/'grid_endpoint_training_v1';out.mkdir(exist_ok=False)
 keys=read(BASE/'bounded_raw_training_inputs_v1/protocol.json')['keys']
 write(out/'protocol.json',dict(keys=keys,spacing_mm=1.,profile_scope='Existing positive TRAIN profile scope; deployment has no positive-query gate',runtime_feature_module_sha256=digest('weld_pipeline/server_grid_feature_stream.py'),GT_coordinate_insertion=False,DEV_reads=0))
 for key in keys:
  dest=out/key;dest.mkdir();old=BASE/'bounded_raw_training_inputs_v1'/key;receipt=read(old/'receipt.json')
  with np.load(old/'events.npz') as z:pids=z['profile_ids'].copy()
  inp=FixedQueries(BASE/'frozen_features/legacy_runtime'/('source_'+key),key);assert inp.bindings==receipt['input_bindings']
  offsets=[0];positions=[];rows=[];lengths=[]
  for pid in pids:
   a,b=inp.offsets[pid:pid+2];spans=inp.spans[a:b,:2];length=float(spans[-1,1]);q=finite_carrier_grid(length,[])['positions_mm']
   local=np.minimum(np.searchsorted(spans[:,1],q,side='right'),len(spans)-1);rep=inp.representative[a+local];fr=np.searchsorted(inp.canonical,rep);assert np.array_equal(inp.canonical[fr],rep)
   positions.append(q);rows.append(fr);lengths.append(length);offsets.append(offsets[-1]+len(q))
  positions=np.concatenate(positions);rows=np.concatenate(rows);offsets=np.asarray(offsets)
  np.savez_compressed(dest/'events.npz',profile_ids=pids,offsets=offsets,positions_mm=positions,feature_rows=rows,lengths_mm=lengths)
  write(dest/'input_freeze.json',dict(input_bindings=inp.bindings,events_sha256=digest(dest/'events.npz'),GT_endpoint_reads=0,scope='Input grids frozen before endpoint targets; positive training profile scope inherited'))
  data=BASE/'exhaustive_carrier_supervision_v1/positive_query'/key;records=read(data/'records.json');dr=read(data/'receipt.json');truthpath=BASE/'labels'/key/'targets.json';assert digest(truthpath)==dr['target_sha256'];truth=read(truthpath)
  pmap={int(pid):i for i,pid in enumerate(pids)};rids=[];interpretations=[];owners=[];ioff=[0];qpid=[];qm=[]
  for ri,record in enumerate(records):
   pid=record['profile']
   if pid not in pmap:continue
   pi=pmap[pid];a,b=offsets[pi:pi+2];q=positions[a:b];length=lengths[pi];query=record['carrier_query_ratio']*length
   ends=np.asarray(inp.profiles[pid]['representation']['endpoints_mm']);direction=ends[1]-ends[0];direction/=np.linalg.norm(direction)
   for parent,child in record['owners']:
    ref=truth['parents'][parent]['children'][child]['representation'];assert ref['kind']=='line';xyz=np.asarray(ref['endpoints_mm']);t=(xyz-ends[0])@direction;order=np.argsort(t);t=t[order];xyz=xyz[order];perp2=np.square(xyz-(ends[0]+t[:,None]*direction)).sum(1)
    if np.any(perp2>9.):continue
    radius=np.sqrt(np.maximum(0,9.-perp2));low=t-radius;high=t+radius;high[0]=min(high[0],query);low[1]=max(low[1],query)
    starts=np.searchsorted(q,low,side='left');stops=np.searchsorted(q,high,side='right')
    if np.any(starts>=stops) or starts[0]>=stops[1]-1:continue
    interpretations.append(np.column_stack((starts,stops)));owners.append([parent,child])
   rids.append(ri);qpid.append(pi);qm.append(query);ioff.append(len(interpretations))
  np.savez_compressed(dest/'supervision.npz',query_rows=rids,query_profile_indices=qpid,query_mm=qm,interpretation_offsets=ioff,endpoint_ranges=np.asarray(interpretations,dtype=np.int64).reshape(-1,2,2),owners=np.asarray(owners,dtype=np.int64).reshape(-1,2))
  report=dict(key=key,profiles=len(pids),grid_events=len(positions),queries=len(rids),queries_with_joint_interpretation=int((np.diff(ioff)>0).sum()),interpretations=len(owners),represented_children=len(set(map(tuple,owners))),events_sha256=digest(dest/'events.npz'),supervision_sha256=digest(dest/'supervision.npz'),target_sha256=digest(truthpath),query_receipt_sha256=digest(data/'receipt.json'),fit_calls=0,DEV_reads=0,scope='Same-owner line endpoint supervision only, not full parent targets; empty interpretations unknown, not negative')
  write(dest/'receipt.json',report);inp.close();print(report,flush=True)
 write(out/'receipt.json',dict(status='EIGHT_TRAIN_GRID_OWNER_SUPERVISION_READY',keys=keys,source_sha256=digest(__file__)))
if __name__=='__main__':main()
