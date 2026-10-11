"""Offline TRAIN diagnosis of class/structure coverage and held-out utilities."""
from pathlib import Path
import collections
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1';KEYS=['21','BM79A_native_local','10','25','36']
def signature(children,gaps,closed,truth=False):
 kinds=sorted(set(c['weld_type']+'/'+c['original_kind' if truth else 'kind'] for c in children))
 return str((kinds,len(children),tuple(sorted(collections.Counter(gaps).items())),bool(closed)))
def main():
 targets={};data={};bindings={}
 for k in KEYS:
  f=BASE/('source_'+k);t=ROOT/'server_rebuild/labels_v1'/k/'targets.json';tr=read(t.parent/'receipt.json');assert tr['role']=='TRAIN' and tr['targets_sha256']==digest(t);targets[k]=collections.Counter(signature(p['children'],p['gaps'],p['is_closed'],True) for p in read(t)['parents'])
  data[k]=(read(f/'parents.json')['parents'],read(f/'d3_supervision_support_v2.json')['rows'],np.load(BASE/'count_head_support_v2'/(k+'_heldout_counts.npy')));bindings[k]=dict(target_sha256=digest(t),supervision_sha256=digest(f/'d3_supervision_support_v2.json'),heldout_counts_sha256=digest(BASE/'count_head_support_v2'/(k+'_heldout_counts.npy')))
 cases=[]
 for k,(parents,rows,pred) in data.items():
  othertruth=sum((targets[j] for j in KEYS if j!=k),collections.Counter());classes={}
  for p,r,y in zip(parents,rows,pred):
   if not r['parent_loss_valid'] or r['D3_TP']!=r['children']:continue
   kind=','.join(sorted(set(s['weld_type']+'/'+s['kind'] for s in p['segments'])));z=classes.setdefault(kind,dict(correct_candidates=0,positive_utility=0,utilities=[],child_counts=collections.Counter(),structure_seen_in_other_GT=0));z['correct_candidates']+=1;z['positive_utility']+=int(y[0]>y[1]);z['utilities'].append(float(y[0]-y[1]));z['child_counts'][p['child_count']]+=1;z['structure_seen_in_other_GT']+=int(signature(p['segments'],p['gaps'],p['is_closed']) in othertruth)
  for z in classes.values():z['utility_quantiles']=np.quantile(z.pop('utilities'),[0,.25,.5,.75,1]).tolist();z['child_counts']=dict(z['child_counts'])
  unseen={s:n for s,n in targets[k].items() if s not in othertruth};row=dict(source=k,correct_candidate_classes=classes,GT_parent_structure_counts=dict(targets[k]),unseen_GT_parent_structures=unseen,unseen_GT_parent_count=sum(unseen.values()));cases.append(row);print(k,classes,'unseen_GT_parents',sum(unseen.values()),flush=True)
 write(BASE/'count_head_generalization_audit.json',dict(cases=cases,bindings=bindings,candidate_counts_are_not_independent_objects=True,no_new_fit=True,DEV_reads=0,scope='Conditional heldout quality-head utility diagnosis; target structure excludes lengths, so seen structure does not prove geometric coverage',source_sha256=digest(__file__)))
if __name__=='__main__':main()
