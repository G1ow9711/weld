"""One registered expansion TRAIN case through unchanged original point models."""
import argparse
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def main(key):
 protocol=read(BASE/'coverage_cohort_protocol.json');assert key in {r['key'] for r in protocol['cases']}
 import tools.rebuild_expansion_input_blocks as blocks
 blocks.DEST.mkdir(parents=True,exist_ok=True);blocks.prepare(key)
 import tools.infer_legacy_point_features as original162
 original162.RESTORE=BASE/'frozen_features';original162.BLOCKS=blocks.DEST
 original162.run(key,0)
 import tools.infer_run26_point_features as original26
 original26.RESTORE=BASE/'frozen_features';original26.BLOCKS=blocks.DEST
 original26.run(key,0)
 write(BASE/'frozen_features'/('complete_'+key+'.json'),dict(status='FROZEN_POINT_MODELS_COMPLETE',key=key,coverage_protocol_sha256=digest(BASE/'coverage_cohort_protocol.json'),GT_reads=0,STEP_reads=0,optimizer_updates=0,source_sha256=digest(__file__)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);main(p.parse_args().key)
