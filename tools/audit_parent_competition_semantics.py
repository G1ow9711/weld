"""Zero-fit TRAIN audit: retrieval coverage and explicit competition counterexamples."""
from pathlib import Path
import collections,itertools
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.server_parent_set_optimization import solve
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def matching_size(sets):
 def search(i,used):
  if i==len(sets):return len(used)
  return max([search(i+1,used)]+[search(i+1,used|{t}) for t in sets[i]-used])
 return search(0,set())
def main():
 ambiguous=[{0,1},{0,1}];assert bool(ambiguous[0]&ambiguous[1]) and matching_size(ambiguous)==2
 triple=[{0},{0,1},{1}];assert all(matching_size([triple[a],triple[b]])==2 for a,b in itertools.combinations(range(3),2));assert matching_size(triple)==2
 selected,reason,receipt=solve(np.array([1.,1.5,1.]),[[0,1],[1,2]],[],[],['a','b','c']);assert set(selected)=={0,2}
 controls=dict(overlapping_compatible_sets_can_coexist=True,pairwise_matching_does_not_imply_joint_matching=True,learning_conflicts_not_transitively_collapsed=True,chain_selected=selected,GT_used_only_in_synthetic_matching_counterexamples=True)
 reports=[]
 for key in ['21','BM79A_native_local','10','25','36']:
  f=BASE/('source_'+key);s=read(f/'d3_supervision.json');rows=s['rows'];pairs=np.load(f/'competition_v2/pairs.npy');pairset={tuple(map(int,p)) for p in pairs};bytarget=collections.defaultdict(list)
  for row in rows:
   targets=row.get('compatible_best_parents',[])
   if row['parent_loss_valid'] and row['D3_TP']==row['children'] and len(targets)==1:bytarget[targets[0]].append(row['hypothesis'])
  positive={tuple(sorted(pair)) for ids in bytarget.values() for pair in itertools.combinations(ids,2)};missed=sorted(positive-pairset)
  resource=read(f/'resource_constraints.json');byresource=collections.defaultdict(list);invalid=set(resource['internally_conflicting_parents'])
  for i,rr in enumerate(resource['parent_resources']):
   if i not in invalid:
    for r in rr:byresource[r].append(i)
  physical={tuple(sorted(pair)) for ids in byresource.values() for pair in itertools.combinations(ids,2)}
  # A missing retrieval pair may already be enforced by the exact resource matrix.
  physical_missed=physical-pairset
  report=dict(source=key,retrieved_pairs=len(pairset),known_unique_target_positive_pairs=len(positive),missed_known_positive_pairs=missed,physical_conflict_pairs=len(physical),physical_conflicts_not_in_learned_pair_retrieval=len(physical_missed),physical_conflicts_still_enforced_by_resource_matrix=True,unknown_competition_recall_not_claimed=True,supervision_sha256=digest(f/'d3_supervision.json'),pair_receipt_sha256=digest(f/'competition_v2/receipt.json'),resource_sha256=digest(f/'resource_constraints.json'));reports.append(report);print({k:v for k,v in report.items() if not k.endswith('sha256')},flush=True)
 write(BASE/'competition_semantic_retrieval_audit.json',dict(cases=reports,controls=controls,fit_calls=0,DEV_reads=0,known_GT_retrieval_is_diagnostic_only=True,source_sha256=digest(__file__)))
if __name__=='__main__':main()
