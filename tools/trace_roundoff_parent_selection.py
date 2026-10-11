"""Replay frozen greedy selection; record deductions without changing decisions."""
import pickle
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
from weld_pipeline.sjg_precision_residual_1225 import IncompatibilityScorer,DuplicateRelations,load_parents,probability,deduction

def main():
 base=ROOT/'server_rebuild/roundoff_gap_baseline_probe_v1';dest=base/'legacy_runtime/source_10/run379';folder=dest/'test_10';reference=read(folder/'final_prediction.json')['selected_candidate_indices'];watch=set(reference[i] for i in [12,66,79,117]);model=IncompatibilityScorer.load(dest/'selector.npz',dest/'frozen_duplicate.npz',dest/'frozen_residual.npz',dest/'base_selector.npz')
 with np.load(dest/'selector_normalization.npz') as z:u=(np.load(folder/'unary.npy')-z['mean'])/z['scale']
 with (dest/'precision_model.pkl').open('rb') as f:correction=pickle.load(f)
 penalty=deduction(probability(correction['model'],u));logits,_=model.logits(u);parents,points=load_parents(folder);context=DuplicateRelations(parents,read(folder/'parent_metadata.json'),model.frozen);trace=[]
 with threadpool_limits(limits=2):
  for step in range(len(parents)+1):
   scores=model.forward(u,context.features(),context.selected,logits);scores[:-1]-=penalty;chosen=model.choose_next(scores,context.selected)
   if chosen is None:break
   assert chosen==reference[step]
   if chosen in watch:
    trace.append(dict(step=step,candidate=chosen,net_score=float(scores[chosen]),stop_score=float(scores[-1]),duplicate_max_deduction=float(context.head.maximum[chosen]),duplicate_argmax=int(context.head.argmax[chosen]),incompatibility_deduction=float(max(0,logits[chosen])),precision_deduction=float(penalty[chosen]),previous_watched_candidates=[i for i in context.selected if i in watch]))
   context.add(chosen)
  assert context.selected==reference
 points._mmap.close();write(base/'selection_deduction_trace.json',dict(status='EXACT_SELECTION_REPLAY',selected_count=len(reference),watched=trace,prediction_receipt_sha256=digest(folder/'final_prediction.json'),scope='Frozen model deductions, no score changes or GT-based runtime selection. Watch indices chosen for offline error attribution.'));print(trace,flush=True)

if __name__=='__main__':main()
