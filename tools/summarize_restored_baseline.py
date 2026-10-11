"""Aggregate verified predictions under the frozen five-case development gates."""
import argparse,json
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/best_baseline_restore'
KEYS=('21','BM79A_native_local','10','25','36')
GATES={'overall':(269,256,13),'Hline':(134,128,6),'Vline':(132,126,6),'HArc':(3,3,0)}

def summarize(model):
 rows=[];totals={};missing=[]
 for key in KEYS:
  folder=BASE/'legacy_runtime'/('source_'+key)/model/('test_'+key);p=folder/'evaluation.json'
  if not p.exists():missing.append(key);continue
  r=read(p);assert r['model']=='restored_'+model and r['role']=='TRAIN'
  assert digest(folder/'final_decoded.json')==r['prediction_sha256']
  target=ROOT/'server_rebuild/labels_v1'/key/'targets.json';assert digest(target)==r['target_sha256']
  for name,value in r['categories'].items():
   total=totals.setdefault(name,dict(G=0,predicted=0,TP=0,FP=0))
   for k in total:total[k]+=value[k]
  o=r['categories']['overall'];rows.append(dict(key=key,**o,FN=o['G']-o['TP'],max_matched_error_mm=max((m['bidirectional_sampled_maximum_mm'] for m in r['matches']),default=None),evaluation_sha256=digest(p)))
 complete=not missing;gates={}
 for name,(expected,tp,fp) in GATES.items():
  actual=totals.get(name,dict(G=0,predicted=0,TP=0,FP=0));gates[name]=dict(expected_G=expected,minimum_TP=tp,maximum_FP=fp,**actual,passed=complete and actual['G']==expected and actual['TP']>=tp and actual['FP']<=fp)
 passed=complete and all(v['passed'] for v in gates.values())
 report=dict(status='COMPLETE_FIVE_CASE_GEOMETRY_SCORE' if complete else 'PARTIAL_GEOMETRY_SCORE',model=model,cases=rows,missing=missing,totals=totals,gates=gates,geometry_development_gate_passed=passed,dev64_end_test_eligible=passed,historical_parity_proven=False,independent_acceptance=False,attributes_passed=False,timing_passed=False,full_contract_pass=False)
 if complete and model=='run446':
  o=totals['overall'];report['historical_259_TP_5_FP_reproduced']=(o['G'],o['TP'],o['FP'])==(269,259,5)
 write(BASE/(model+'_restored_summary.json'),report)
 print(json.dumps(dict(model=model,complete=complete,cases=rows,gates=gates,full_contract_pass=False)))
 return report
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['run379','run446'],required=True);summarize(ap.parse_args().model)
