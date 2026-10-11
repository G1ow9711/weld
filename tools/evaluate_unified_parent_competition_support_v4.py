"""Original TRAIN D3 evaluation of frozen, already selected model outputs."""
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def main():
 reports=[];totals={}
 for key in ['21','BM79A_native_local','10','25','36']:
  f=BASE/'set_competition_support_v4'/('source_'+key);r=read(f/'receipt.json');assert digest(f/'prediction.json')==r['prediction_sha256'];t=ROOT/'server_rebuild/labels_v1'/key/'targets.json';tr=read(t.parent/'receipt.json');assert tr['role']=='TRAIN' and tr['targets_sha256']==digest(t)
  truth=read(t);score=score_native_parents(read(f/'prediction.json'),truth,[False]*len(truth['parents']));b=BASE/'frozen_defaults'/(key+'.json');assert digest(b)==r['baseline_sha256'];old=score_native_parents(read(b),truth,[False]*len(truth['parents']))
  known=lambda s:{(m['original_parent_index'],m['child_index']) for m in s['matches']}
  before=known(old);after=known(score);report=dict(source=key,score=score,retained_original_children=sorted(before&after),lost_original_children=sorted(before-after),new_original_children=sorted(after-before),prediction_sha256=digest(f/'prediction.json'),target_sha256=digest(t),independent_acceptance=False,DEV_reads=0)
  write(f/'evaluation.json',report);reports.append(dict(source=key,categories=score['categories'],old_TP=len(before),retained=len(before&after),lost=len(before-after),new=len(after-before)))
  for c,v in score['categories'].items():
   total=totals.setdefault(c,{k:0 for k in v})
   for k,x in v.items():total[k]+=x
  print(reports[-1],flush=True)
 gate=totals['overall']['TP']>=256 and totals['overall']['FP']<=13 and totals['Hline']['TP']>=128 and totals['Hline']['FP']<=6 and totals['Vline']['TP']>=126 and totals['Vline']['FP']<=6 and totals['HArc']['TP']==3 and totals['HArc']['FP']==0
 write(BASE/'set_competition_support_v4/evaluation_summary.json',dict(cases=reports,categories=totals,five_TRAIN_geometry_gate=gate,formal_contract_pass=False,not_promoted=True,DEV_reads=0))
 print('TOTAL',totals,'FIVE_TRAIN_GATE',gate,flush=True)
if __name__=='__main__':main()
