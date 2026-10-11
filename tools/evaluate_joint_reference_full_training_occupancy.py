"""Frozen detector TRAIN full-query fit audit; no threshold selection or extraction claim."""
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.server_joint_reference_query_model import JointReferenceModel
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';OUT=BASE/'joint_reference_full_training_occupancy_v1'
def main():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(2);folder=BASE/'positive_query_head_v1/joint_reference_v1';r=read(folder/'receipt.json');assert digest(folder/'model.pt')==r['model_sha256'];s=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model=JointReferenceModel().cuda().eval();model.load_state_dict(s['state_dict']);reports=[]
 for key in s['presence_training_keys']:
  started=time.perf_counter();dest=OUT/key;dest.mkdir();inp=FixedQueries(BASE/'frozen_features/legacy_runtime'/('source_'+key),key);scores=np.empty(inp.n,np.float32)
  with torch.no_grad():
   for a in range(0,inp.n,4096):
    ids=np.arange(a,min(a+4096,inp.n));x=inp.features(ids);z=torch.as_tensor(np.clip((x-s['mean'])/s['std'],-20,20),device='cuda');scores[ids]=torch.sigmoid(model.reference(model.trunk(z)).squeeze(1)).cpu().numpy()
  assert np.isfinite(scores).all();np.save(dest/'scores.npy',scores);write(dest/'prediction_freeze.json',dict(model_sha256=r['model_sha256'],input_bindings=inp.bindings,scores_sha256=digest(dest/'scores.npy'),labels_read_during_prediction=0,source_sha256=digest(__file__)))
  data=BASE/'reference_query_task_v1'/key;rr=read(data/'receipt.json');assert digest(data/'receipt.json')==r['presence_bindings'][key];assert rr['input_bindings']==inp.bindings;assert digest(data/'reference_query_labels.npy')==rr['files']['reference_query_labels.npy'];y=np.load(data/'reference_query_labels.npy');assert len(y)==len(scores);known=y>=0;positive=y==1;selected=known&(scores>=.5);tp=int((selected&positive).sum());fp=int((selected&~positive).sum());fn=int((~selected&positive).sum());queries=BASE/'exhaustive_carrier_supervision_v1/positive_query'/key;records=read(queries/'records.json');owners=set();hit=set()
  for rec in records:
   q=rec['feature_rows'][0]//2*4+rec['query_index'];os={tuple(v) for v in rec['owners']};owners|=os
   if selected[q]:hit|=os
  report=dict(key=key,known_queries=int(known.sum()),positive_queries=int(positive.sum()),selected_queries=int(selected.sum()),reference_query_TP=tp,reference_query_FP=fp,reference_query_FN=fn,reference_query_precision=tp/max(1,tp+fp),reference_query_recall=tp/max(1,tp+fn),reference_query_average_precision=float(average_precision_score(positive[known],scores[known])),represented_positive_children=len(owners),children_with_selected_positive_query=len(hit),positive_score_quantiles=np.quantile(scores[positive],[0,.25,.5,.75,.9,.99,1]).tolist(),scope='Full TRAIN query occupancy audit, not endpoint/parent extraction or physical-negative labeling; threshold remains0.5',fit_calls=0,threshold_changes=0,DEV_reads=0,heldout_reads=0,seconds=time.perf_counter()-started);write(dest/'receipt.json',report);reports.append(report);print(report,flush=True);inp.close()
 write(OUT/'summary.json',dict(cases=reports,model_sha256=r['model_sha256'],no_training_or_threshold_selection=True))
if __name__=='__main__':main()
