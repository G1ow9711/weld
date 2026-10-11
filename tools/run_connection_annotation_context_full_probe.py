"""Frozen-input whole-parent evaluation of held-family connection adapter."""
import pickle,time
from threadpoolctl import threadpool_limits
import tools.rebuild_legacy_downstream as runtime
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
from weld_pipeline.server_connection_interval_context import context_link_blocks
from weld_pipeline.server_connection_interval_context import ContextConnectionAdapter
import weld_pipeline.sjg_relative_forest_runtime_1139 as relative


def main():
 assert read(ROOT/'server_rebuild/connection_interval_context_v1/runtime_parity.json')['status']=='PASS'
 key='10';gatepath=ROOT/'server_rebuild/connection_annotation_gate_context_v1/leave_10/model.pkl';receipt=read(gatepath.parent/'receipt.json')
 assert key not in receipt['train_keys'] and digest(gatepath)==receipt['model_sha256']
 source=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime/source_10';base=ROOT/'server_rebuild/connection_annotation_context_full_probe_v1';base.mkdir(exist_ok=False)
 case=base/'legacy_runtime/source_10';case.mkdir(parents=True)
 for p in source.iterdir():
  if p.is_dir() and p.name!='run379':(case/p.name).symlink_to(p,target_is_directory=True)
 dest=case/'run379';dest.mkdir()
 for p in (source/'run379').iterdir():
  if p.is_file() and not p.name.startswith('screen_64'):(dest/p.name).symlink_to(p)
 write(base/'protocol.json',dict(gate_model_sha256=digest(gatepath),gate_training_keys=receipt['train_keys'],adapter_sha256=digest(ROOT/'weld_pipeline/server_connection_interval_context.py'),guard_sha256=digest(ROOT/'weld_pipeline/server_roundoff_gap_links.py'),rule='For line-line edges replace total positive mass with binary gate probability; preserve old conditional three-way relation ratios. Arc edges and zero-positive-mass edges unchanged. No threshold or reference lookup.',scope='Held-family downstream diagnostic; original upstream has seen this family. Original baseline 196TP0FP and numerical guard 197TP3FP. Not full XYZ or independent generalization.',protected_DEV_reads=0))
 runtime.CASE=case;runtime.BASE=base;relative.relative_link_blocks=context_link_blocks
 with gatepath.open('rb') as f:gate=pickle.load(f)
 worker=runtime.module('sjg_sparse_recovery_worker_1170');recovery=runtime.module('sjg_recall_recovery_worker_1165');models,_=runtime.module('sjg_recall_set_worker_1162').load_upstream(dest)
 models['connection']=ContextConnectionAdapter(models['connection'],gate);recovery.component=lambda name:models[name]
 with threadpool_limits(limits=2):
  start=time.perf_counter();worker.build_graph(dest,key,True);write(base/'edge_graph_complete.json',dict(seconds=time.perf_counter()-start))
  for stage in ('solve','decode','select'):
   start=time.perf_counter();runtime.forest_stage(key,stage);write(base/(stage+'_complete.json'),dict(seconds=time.perf_counter()-start));print('DONE',stage,flush=True)
 from weld_pipeline.sjg_shared_prediction_reader_1236 import read_shared_predictions
 from weld_pipeline.sjg_recall_set_worker_1162 import score_value
 pred=dest/'test_10/final_decoded.json';target=ROOT/'server_rebuild/labels_v1/10/targets.json';result=score_value(read_shared_predictions(pred),read(target),key)
 write(base/'evaluation.json',result);write(base/'receipt.json',dict(status='FULL_PARENT_DIAGNOSTIC_COMPLETE',report=result['report'],prediction_sha256=digest(pred),target_sha256=digest(target),promoted=False));print(result['report'],flush=True)

if __name__=='__main__':main()
