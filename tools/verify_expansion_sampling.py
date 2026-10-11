"""Actual frozen-parent deduplication and stored-traversal controls."""
from pathlib import Path
from weld_pipeline.server_expansion_sampling import eligible_rows,round_robin,canonical_parent_signature
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def main():
 rows=[]
 for key in ['21','BM79A_native_local','10','25','36']:
  f=BASE/('source_'+key);parents=read(f/'parents.json')['parents'];sup=read(f/'d3_supervision.json')['rows'];items,duplicates=eligible_rows(parents,sup,key)
  for p in parents:
   # Reverse stored points exactly, without re-evaluating/resampling arc geometry.
   rev=dict(p,segments=[dict(s,points=s['points'][::-1]) for s in p['segments'][::-1]],gaps=p['gaps'][::-1]);assert canonical_parent_signature(p)==canonical_parent_signature(rev)
   if p['is_closed'] and len(p['segments'])>1:
    ring=p['gaps']+['SHARED'];rotated=dict(p,segments=p['segments'][1:]+p['segments'][:1],gaps=(ring[1:]+ring[:1])[:-1]);assert canonical_parent_signature(p)==canonical_parent_signature(rotated)
  selected=round_robin({key:items},len(items));assert len(selected)==len({r['signature'] for r in selected})
  rows.append(dict(key=key,known=sum(r['parent_loss_valid'] for r in sup),unique=len(items),duplicates=len(duplicates),stored_parent_reversal_and_closed_shift_equal=True))
 write(ROOT/'server_rebuild/expansion_v1/routing_controls/sampler_receipt.json',dict(status='PASS',cases=rows,scope='Actual frozen TRAIN parents; exact stored geometry deduplication, traversal and cyclic ordering. Analytically re-sampled arcs with floating-point differences deliberately remain separate. No rounded or averaged geometry.',earlier_control_failure='Using legacy reverse_segment resamples arcs; it is not bitwise involutive. Replaced signature construction by exact stored point reversal, without modifying predictions.',source_sha256=digest(ROOT/'weld_pipeline/server_expansion_sampling.py'),fit_calls=0));print(rows)
if __name__=='__main__':main()
