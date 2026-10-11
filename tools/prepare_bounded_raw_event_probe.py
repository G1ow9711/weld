"""Freeze fixed1mm input event covers before diagnostic TRAIN target reads."""
from pathlib import Path
import sys,subprocess
import numpy as np
from weld_pipeline.server_bounded_endpoint_event_cover import bounded_event_cover
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'bounded_raw_anchor_training_probe_v1'
def main():
 OUT.mkdir(exist_ok=False);controls=read(BASE/'bounded_event_cover_controls_v1/receipt.json');assert controls['status']=='PASS' and controls['module_sha256']==digest('weld_pipeline/server_bounded_endpoint_event_cover.py');keys=read(BASE/'raw_tube_anchor_training_probe_v1/protocol.json')['keys'];write(OUT/'protocol.json',dict(keys=keys,cell_mm=1.,maximum_omitted_event_nearest_cover_mm=.5,all_original_anchors_preserved=True,no_new_coordinates=True,GT_reads_during_candidate_freeze=0,not_guaranteed_to_preserve_reference3mm_gate=True,DEV_reads=0))
 for key in keys:
  source=BASE/'raw_tube_anchor_training_probe_v1'/key;r=read(source/'receipt.json');assert digest(source/'events.npz')==r['events_sha256'];old=BASE/'sampled_boundary_anchor_training_v1'/key;orr=read(old/'receipt.json');assert digest(old/'events.npz')==orr['events_sha256'];z=np.load(source/'events.npz');o=np.load(old/'events.npz');assert np.array_equal(z['profile_ids'],o['profile_ids']);ids=z['profile_ids'];pos=z['positions_mm'];offsets=z['offsets'];opos=o['positions_mm'];ooffsets=o['offsets'];lengths=o['profile_length_mm'];chunks=[];new_offsets=[0];maxgap=0.
  for i,pid in enumerate(ids):
   q=pos[offsets[i]:offsets[i+1]];anchors=opos[ooffsets[i]:ooffsets[i+1]];cover=bounded_event_cover(q,lengths[i],anchors);assert np.isin(anchors,cover).all() and np.isin(cover,q).all();index=np.searchsorted(cover,q);distance=np.minimum(abs(q-cover[np.minimum(index,len(cover)-1)]),abs(q-cover[np.maximum(index-1,0)]));maxgap=max(maxgap,float(distance.max()));assert maxgap<=.5+1e-8;chunks.append(cover);new_offsets.append(new_offsets[-1]+len(cover))
  dest=OUT/key;dest.mkdir();np.savez(dest/'events.npz',profile_ids=ids,positions_mm=np.concatenate(chunks),offsets=np.asarray(new_offsets,np.int64));write(dest/'receipt.json',dict(key=key,events_sha256=digest(dest/'events.npz'),query_receipt_sha256=r['query_receipt_sha256'],input_sha256=r['input_sha256'],raw_event_receipt_sha256=digest(source/'receipt.json'),old_event_receipt_sha256=digest(old/'receipt.json'),events=new_offsets[-1],full_union_events=len(pos),max_events_per_profile=int(np.diff(new_offsets).max()),maximum_nearest_cover_mm=maxgap,GT_reads=0,DEV_reads=0));print('FROZEN',key,new_offsets[-1],maxgap,flush=True)
 # Both input caches frozen before the first target access in subprocess.
 rows=[]
 for key in keys:
  with (OUT/(key+'_supervision.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/prepare_bounded_raw_anchor_supervision_probe.py'),'--key',key],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  r=read(BASE/'bounded_raw_anchor_supervision_probe_v1'/key/'receipt.json');rows.append(dict(key=key,children=r['line_reference_children'],both_available=r['children_with_both_endpoints_available']));print(rows[-1],flush=True)
 write(OUT/'receipt.json',dict(status='BOUNDED_RAW_INPUT_COVER_AND_TRAIN_CAPACITY_COMPLETE',rows=rows,fit_calls=0,DEV_reads=0,not_full_extraction=True))
if __name__=='__main__':main()
