"""Unchanged original fragment/forest stages on registered expansion TRAIN cases."""
import argparse,time
from pathlib import Path
from threadpoolctl import threadpool_limits
import tools.rebuild_legacy_downstream as original
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1/frozen_features'
def run(key,stage):
 assert key in {c['key'] for c in read(BASE.parent/'additional_batch01_protocol.json')['cases']};original.BASE=BASE;original.CASE=BASE/'legacy_runtime'/('source_'+key);assert (original.CASE/'run185'/('source_'+key)/'receipt.json').exists();marker=original.CASE/(stage+'_reconstruction_receipt.json');assert not marker.exists();start=time.perf_counter()
 with threadpool_limits(limits=2):
  handled=False
  if stage=='select':
   from weld_pipeline.server_empty_parent_selection import select_if_empty
   handled=select_if_empty(original.CASE/'run379',key)
  if not handled:original.run(key,stage)
 write(marker,dict(status='RECONSTRUCTED_STAGE_COMPLETE',stage=stage,source_key=key,seconds=time.perf_counter()-start,source_sha256=digest(__file__),original_router_sha256=digest(original.__file__),GT_reads=0,STEP_reads=0,original_weights_unchanged=True,empty_selection_adapter_used=handled))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);ap.add_argument('--stage',required=True,choices=['graphs','object_features','relative_features','observed_features','bind_models','screen','edge_graph','solve','decode','select']);a=ap.parse_args();run(a.key,a.stage)
