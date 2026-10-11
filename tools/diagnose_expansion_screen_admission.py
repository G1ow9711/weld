"""TRAIN-only localization of frozen screen loss on already-generated carriers."""
import argparse,collections,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import tools.rebuild_legacy_downstream as runtime
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_ordered_interval_targets_1128 import decode_count_category
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def run(key):
 assert key in {c['key'] for c in read(BASE/'coverage_cohort_protocol.json')['cases']};start=time.perf_counter();runtime.BASE=BASE/'frozen_features';runtime.CASE=runtime.BASE/'legacy_runtime'/('source_'+key)
 coveragepath=BASE/'native_capacity'/('source_'+key)/'raw_carrier_coverage.json';coverage=read(coveragepath);targetpath=BASE/'labels'/key/'targets.json';assert digest(targetpath)==coverage['target_sha256'];targets=read(targetpath)
 case=runtime.module('sjg_relative_ordinal_runtime_1150').load_case(key);models,binding=runtime.module('sjg_recall_set_worker_1162').load_upstream(runtime.CASE/'run379');obj=models['object'];counts=np.array([decode_count_category(v)[1] for v in obj.classes_]);assert obj.n_features_in_==case['x'].shape[1]
 with np.load(runtime.CASE/'run288'/('source_'+key+'_alias_map.npz')) as z:representative=z['fragment_representative']
 pending=[];all_rows=set()
 for row in coverage['rows']:
  if row['kind']!='line':continue
  c=targets['parents'][row['parent']]['children'][row['child']];ends=np.asarray(c['representation']['endpoints_mm']);candidates=[]
  for near in row['nearest']:
   if near['max_distance_mm']>3.:continue
   pid=near['profile'];curve=case['graph'].curve(pid);q,_=curve.project(ends);lo,hi=sorted(q);a,b=case['graph'].offsets[pid:pid+2];spans=case['graph'].spans[a:b];overlap=np.maximum(0,np.minimum(spans[:,1],hi)-np.maximum(spans[:,0],lo));take=np.flatnonzero(overlap>0);fragments=take+a;reps=representative[fragments];loc=np.searchsorted(case['canonical'],reps);assert np.array_equal(case['canonical'][loc],reps)
   rr=np.column_stack((2*loc,2*loc+1));all_rows.update(rr.ravel().tolist());candidates.append(dict(profile=pid,object_rows=rr.tolist(),overlap_lengths_mm=overlap[take].tolist(),projected_target_length_mm=float(hi-lo)))
  pending.append(dict(parent=row['parent'],child=row['child'],weld_type=row['weld_type'],carrier_candidates=candidates))
 rr=np.array(sorted(all_rows),int);probs=obj.predict_proba(np.asarray(case['x'][rr],np.float32));presence=probs[:,counts>=1].sum(1);lookup=dict(zip(rr,presence));summary=collections.defaultdict(lambda:dict(G=0,covered_carriers=0,any_admitted_fragment=0,full_projected_interval_admitted=0))
 for row in pending:
  for cand in row['carrier_candidates']:
   scores=np.array([[lookup[i] for i in pair] for pair in cand['object_rows']]);w=np.array(cand['overlap_lengths_mm']);p=scores.max(1);cand['presence_probabilities_both_sides']=scores.tolist();cand['admitted_fraction_of_projected_target']=float(w[p>=.5].sum()/cand['projected_target_length_mm']);cand['max_presence_probability']=float(p.max(initial=0))
  vals=row['carrier_candidates'];r=summary[row['weld_type']];r['G']+=1;r['covered_carriers']+=bool(vals);r['any_admitted_fragment']+=any(v['max_presence_probability']>=.5 for v in vals);r['full_projected_interval_admitted']+=any(v['admitted_fraction_of_projected_target']>=1-1e-8 for v in vals)
 case['x']._mmap.close();report=dict(key=key,summary=dict(summary),rows=pending,coverage_sha256=digest(coveragepath),target_sha256=digest(targetpath),original_model_binding=binding,unique_feature_rows_scored=len(rr),scope='At most three nearest finite line carriers per GT child; evaluates fragment presence only, not regressed endpoints, parent assembly, face truth, or final acceptance. Arcs excluded from interval-span diagnostic.',diagnostic_only=True,fit_calls=0,DEV_reads=0,GT_not_used_for_inference_output=True,seconds=time.perf_counter()-start)
 out=coveragepath.parent/'screen_admission.json';assert not out.exists();write(out,report);print(dict(key=key,summary=dict(summary),unique_feature_rows_scored=len(rr)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
 with threadpool_limits(limits=2):run(a.key)
