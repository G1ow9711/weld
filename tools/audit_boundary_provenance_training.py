"""TRAIN-only endpoint event source association, not a weld detector or accuracy."""
from pathlib import Path
import time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';OUT=BASE/'boundary_provenance_training_audit_v1'
def main():
 OUT.mkdir(exist_ok=False);source=BASE/'boundary_event_provenance_v1';keys=read(source/'protocol.json')['keys'];proc=Path('/proc/640627');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;deadline=time.monotonic()+3600;reports=[]
 for key in keys:
  cache=source/key
  while not (cache/'receipt.json').exists():
   if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Input cache producer terminal without receipt')
   if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
   time.sleep(5)
  r=read(cache/'receipt.json');x=np.load(cache/'features.npy');assert digest(cache/'features.npy')==r['features_sha256'];events=BASE/'sampled_boundary_anchor_training_v1'/key;er=read(events/'receipt.json');assert digest(events/'receipt.json')==r['events_receipt_sha256'];sup=BASE/'boundary_anchor_supervision_v1'/key;sr=read(sup/'receipt.json');assert sr['events_receipt_sha256']==digest(events/'receipt.json');assert digest(sup/'targets.npz')==sr['supervision_sha256'];s=np.load(sup/'targets.npz');good=np.zeros(len(x),bool);good[s['left_event_ids']]=True;good[s['right_event_ids']]=True;rows=[]
  for j,name in enumerate(r['columns'][:4]):
   mask=x[:,j]>0;rows.append(dict(source=name,all_events=int(mask.sum()),positive_endpoint_events=int((mask&good).sum()),fraction_endpoint_positive=float(good[mask].mean()) if mask.any() else None))
  patterns=(x[:,:4]>0).astype(np.int64)@np.array([1,2,4,8]);groups=[]
  for pattern in np.unique(patterns):
   mask=patterns==pattern;groups.append(dict(source_bitmask=int(pattern),all_events=int(mask.sum()),positive_endpoint_events=int((mask&good).sum())))
  report=dict(key=key,total_events=len(x),endpoint_positive_events=int(good.sum()),sources=rows,source_combinations=groups,scope='TRAIN union of endpoint-positive event sets across reliable positive queries; association only, not physical weldability or full extraction; categories overlap',input_receipt_sha256=digest(cache/'receipt.json'),supervision_receipt_sha256=digest(sup/'receipt.json'),fit_calls=0,heldout_reads=0,DEV_reads=0);write(OUT/(key+'.json'),report);reports.append(report);print(key,rows,flush=True)
 write(OUT/'summary.json',dict(cases=reports,source_sha256=digest(__file__),scope='TRAIN-only association, no source category rule selected'))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
