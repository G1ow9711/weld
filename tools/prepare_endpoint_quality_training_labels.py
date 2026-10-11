"""TRAIN-only geometry-quality labels plus all compatible reference-child owners."""
import time,traceback,collections
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';OUT=BASE/'endpoint_quality_training_labels_v1'
def main():
 OUT.mkdir(exist_ok=False);keys=read(BASE/'reference_query_task_v1/protocol.json')['training_keys'];write(OUT/'protocol.json',dict(training_keys=keys,target='Actual compact prediction matches at least one reference child geometry and type; duplicates may share owner',zero_not_physical_absence=True,quality_positive_not_final_unique_output=True,weights='Half class mass; positives equal covered-reference-child mass divided over compatible compact predictions; backgrounds uniform; equal families during fit',upstream_predictions_in_sample=True,quality_layer_family_holdout_not_full_chain_OOF=True,heldout_label_reads=0,DEV_reads=0));pids=[691475,711615];identities={pid:((Path('/proc')/str(pid)/'stat').read_text().split()[21] if (Path('/proc')/str(pid)).exists() else None) for pid in pids};deadline=time.monotonic()+10800;reports=[]
 for key in keys:
  feature=BASE/'endpoint_candidate_quality_features_v1'/key;output=BASE/'reference_detector_balanced_endpoint_full_output_v1'/key
  for pid,ready in [(691475,output/'child_reference_capacity.json'),(711615,feature/'receipt.json')]:
   proc=Path('/proc')/str(pid)
   while not ready.exists():
    if identities[pid] is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identities[pid]:raise RuntimeError(f'Producer {pid} terminal without {ready}')
    if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
    time.sleep(5)
  f=read(feature/'receipt.json');r=read(output/'child_reference_capacity.json');assert f['prediction_freeze_sha256']==r['runtime_receipt_sha256']==digest(output/'prediction_freeze.json');assert digest(feature/'query_to_compact.npy')==f['query_to_compact_sha256'];assert digest(output/'reference_child_quality.npy')==r['reference_quality_sha256'];group=np.load(feature/'query_to_compact.npy');query_quality=np.load(output/'reference_child_quality.npy');owners=[set() for _ in range(f['rows'])]
  for owner,row in enumerate(r['rows']):
   for compact in np.unique(group[row['compatible_query_ids']]):owners[int(compact)].add(owner)
  y=np.array([bool(o) for o in owners],np.int8);assert np.array_equal(y[group],query_quality),'Exact duplicate predictions have inconsistent geometry labels';counts=collections.Counter(o for row in owners for o in row);weights=np.zeros(len(y),np.float64);positive=np.flatnonzero(y);negative=np.flatnonzero(y==0);assert len(positive) and len(negative)
  for i in positive:weights[i]=.5*sum(1/counts[o] for o in owners[i])/len(counts)
  weights[negative]=.5/len(negative);assert np.isclose(weights.sum(),1);offsets=np.r_[0,np.cumsum([len(v) for v in owners])];dest=OUT/key;dest.mkdir();np.savez(dest/'labels.npz',quality=y,weights=weights,owner_offsets=offsets,owner_ids=np.array([o for v in owners for o in sorted(v)],np.int64));report=dict(key=key,candidates=len(y),positive_candidates=len(positive),reference_unmatched_candidates=len(negative),covered_reference_children=len(counts),multiple_owner_candidates=sum(len(v)>1 for v in owners),feature_receipt_sha256=digest(feature/'receipt.json'),quality_report_sha256=digest(output/'child_reference_capacity.json'),labels_sha256=digest(dest/'labels.npz'),fit_calls=0,DEV_reads=0);write(dest/'receipt.json',report);reports.append(report);print(report,flush=True)
 write(OUT/'receipt.json',dict(status='SIX_TRAIN_REFERENCE_QUALITY_LABELS_COMPLETE',cases=reports,heldout_label_reads=0,fit_calls=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
