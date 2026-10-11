"""Diagnostic crossed legality/probability arms with fixed graph inputs."""
import argparse,time
from threadpoolctl import threadpool_limits
import tools.rebuild_legacy_downstream as runtime
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest

def main(arm):
 source=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime/source_10';old=source/'run379/test_10';guard=ROOT/'server_rebuild/roundoff_gap_baseline_probe_v1';new=guard/'legacy_runtime/source_10/run379/test_10';registered=guard/'factorial_followup_protocol.json';assert arm in read(registered)['arms']
 base=ROOT/'server_rebuild/roundoff_factorial_v1'/arm;base.mkdir(parents=True,exist_ok=False);case=base/'legacy_runtime/source_10';case.mkdir(parents=True)
 for p in source.iterdir():
  if p.is_dir() and p.name!='run379':(case/p.name).symlink_to(p,target_is_directory=True)
 dest=case/'run379';dest.mkdir()
 for p in (source/'run379').iterdir():
  if p.is_file():(dest/p.name).symlink_to(p)
 folder=dest/'test_10';folder.mkdir();a=read(old/'edge_manifest.json');b=read(new/'edge_manifest.json');assert a['edge_count']==b['edge_count'];manifest=dict(a);manifest['files']={};manifest.pop('features_sha256',None)
 for name in a['files']:
  origin=old
  if (name=='legal.u8' and arm=='new_legal_old_probability') or (name=='probabilities.f64' and arm=='old_legal_new_probability'):origin=new
  if name not in ('legal.u8','probabilities.f64'):assert digest(old/name)==digest(new/name)
  expected=(b if origin==new else a)['files'][name]['sha256'];assert digest(origin/name)==expected;(folder/name).symlink_to(origin/name);manifest['files'][name]=dict(sha256=expected,bytes=(origin/name).stat().st_size)
 manifest['factorial_diagnostic']=dict(arm=arm,old_manifest_sha256=digest(old/'edge_manifest.json'),guard_manifest_sha256=digest(new/'edge_manifest.json'),not_single_feature_generation=True);write(folder/'edge_manifest.json',manifest)
 write(base/'protocol.json',dict(registered_protocol_sha256=digest(registered),arm=arm,script_sha256=digest(__file__),scope='Crossed diagnostic only; original weights and same node/edge/resource geometry, not production pipeline',protected_DEV_reads=0))
 runtime.CASE=case;runtime.BASE=base
 with threadpool_limits(limits=2):
  for stage in ('solve','decode','select'):
   start=time.perf_counter();runtime.forest_stage('10',stage);write(base/f'{stage}_complete.json',dict(stage=stage,seconds=time.perf_counter()-start));print(arm,'DONE',stage,flush=True)
 from weld_pipeline.sjg_shared_prediction_reader_1236 import read_shared_predictions
 from weld_pipeline.sjg_recall_set_worker_1162 import score_value
 pred=folder/'final_decoded.json';target=ROOT/'server_rebuild/labels_v1/10/targets.json';score=score_value(read_shared_predictions(pred),read(target),'10');write(base/'evaluation.json',score);write(base/'receipt.json',dict(status='COMPLETE_DIAGNOSTIC_ARM',report=score['report'],prediction_sha256=digest(pred),target_sha256=digest(target),promoted=False));print(arm,score['report'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['new_legal_old_probability','old_legal_new_probability']);main(p.parse_args().arm)
