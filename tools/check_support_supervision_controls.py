"""Validate finite-support supervision against actual line samples and D3."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets,score_native_parents
from weld_pipeline.world_set_evaluation_285 import reference_segment
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def main():
 reports=[]
 for key in ['21','BM79A_native_local','10','25','36']:
  f=BASE/('source_'+key);s=read(f/'d3_supervision_support_v2.json');prior=read(f/'d3_supervision.json');segments=read(f/'child_segments.json');targetpath=ROOT/'server_rebuild/labels_v1'/key/'targets.json';truth=read(targetpath);native=native_targets(truth);worst=0.;checks=0
  for seg,edges in zip(segments,s['child_support']):
   for edge in edges:
    c=native['parents'][edge['target_parent']]['children'][edge['target_child']];e=np.asarray(reference_segment(c)['points'])[[0,-1]];v=e[1]-e[0];length=np.linalg.norm(v);v/=length;pts=np.asarray(seg['points']);q=(pts-e[0])@v;d=np.linalg.norm(pts-e[0]-np.clip(q,0,length)[:,None]*v,axis=1).max();assert d<=3.+1e-8;worst=max(worst,float(d));checks+=1
  parents=read(f/'parents.json')['parents'];classes=set();controls=[]
  for i,(old,row) in enumerate(zip(prior['rows'],s['rows'])):
   assert row['D3_TP']==old['D3_TP'] and row['D3_FP']==old['D3_FP']
   assert row['parent_loss_valid']==(old['parent_loss_valid'] or row['finite_annotated_line_support'])
   if old['parent_loss_valid'] or not row['parent_loss_valid']:continue
   signature=(row['children'],row['D3_TP'],parents[i]['is_closed'])
   if signature in classes:continue
   classes.add(signature);p=parents[i];score=score_native_parents(dict(parents=[p],invalid_predictions=[],total_prediction_children=p['child_count']),truth,[False]*len(truth['parents']))['categories']['overall'];assert score['TP']==row['D3_TP'] and score['FP']==row['D3_FP'];controls.append(dict(parent=i,TP=score['TP'],FP=score['FP']))
  report=dict(source=key,support_links_checked_on_all_actual_samples=checks,max_distance_mm=worst,D3_new_supervision_controls=controls,supervision_sha256=digest(f/'d3_supervision_support_v2.json'),target_sha256=digest(targetpath),PASS=True);reports.append(report);print(key,checks,len(controls),flush=True)
 write(BASE/'support_supervision_controls.json',dict(cases=reports,all_pass=True,scope='Finite annotated support certificates and stratified new D3 supervision controls; not inference or material identity verification'))
if __name__=='__main__':main()
