"""Route the frozen original 446 composite to registered expansion TRAIN inputs."""
import argparse,time
from pathlib import Path
from threadpoolctl import threadpool_limits
import tools.rebuild_legacy_downstream as runtime
from tools import rebuild_legacy_incremental_inputs as inputs
from tools import rebuild_legacy_sequences as sequences
from tools import rebuild_legacy_additions as additions
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1/frozen_features'
INPUTS=['whole','local','admission'];ADDITIONS=['occupancy','overlay','pairs','parent_features','child_features','infer446']
def run(key,stage):
 assert key in {c['key'] for c in read(BASE.parent/'coverage_cohort_protocol.json')['cases']}
 runtime.BASE=BASE;runtime.CASE=BASE/'legacy_runtime'/('source_'+key)
 assert (runtime.CASE/'select_reconstruction_receipt.json').exists()
 if stage.startswith('input_'):module=inputs;substage=stage[6:];assert substage in INPUTS
 elif stage.startswith('sequence_'):module=sequences;substage=stage[9:];assert substage in sequences.STAGES
 elif stage.startswith('addition_'):module=additions;substage=stage[9:];assert substage in ADDITIONS
 else:raise ValueError(stage)
 marker=BASE/'expansion_incremental_receipts'/key/(stage+'.json');assert not marker.exists();marker.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 with threadpool_limits(limits=2):module.run(key,substage)
 write(marker,dict(status='COMPLETE',key=key,stage=stage,seconds=time.perf_counter()-start,source_sha256=digest(__file__),original_router_sha256=digest(module.__file__),GT_reads=0,STEP_reads=0,fit_calls=0,original_weights_unchanged=True))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);ap.add_argument('--stage',required=True,choices=['input_'+x for x in INPUTS]+['sequence_'+x for x in sequences.STAGES]+['addition_'+x for x in ADDITIONS]);a=ap.parse_args();run(a.key,a.stage)
