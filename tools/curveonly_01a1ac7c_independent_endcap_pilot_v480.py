"""Recompute all-input finite-cap proposals with the controlled query policy."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import types,sys,subprocess,time
from pathlib import Path
from tools import curveonly_01a1ac7c_endcap_connection_pilot_v468 as common_pilot
from tools import curveonly_01a1ac7c_endcap_coverage_pilot_v470 as previous
from weld_pipeline import curveonly_01a1ac7c_independent_endcaps_v476 as rules

e,common=previous.e,previous.common
DEST=e.REC/'independent_endcap_pilot_v480'


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(previous.DEST/'CLOSURE.json')['analysis_closed']
    p=e.read(previous.DEST/'FROZEN_PROTOCOL.json')
    for binding in p['sources']+[p['inputs'],p['model_outputs']]:common.valid(binding)
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(rules.__file__).resolve()]
    for source in sources:assert source.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(source.relative_to(e.ROOT))])
    DEST.mkdir(exist_ok=True);e.write(DEST/'FROZEN_PROTOCOL.json',dict(p,source_commit=commit,
        sources=[e.bind(p) for p in sources]+p['sources'],
        change='Always query the actual finite end-cap witness, independently from generic ray correspondence.',
        no_source_process_or_score_tolerance_change=True,epoch=time.time()))


def observe():
    return types.FunctionType(common_pilot.observe.__code__,dict(common_pilot.__dict__,DEST=DEST,rules=rules))()


if __name__=='__main__':{'prepare':prepare,'observe':observe}[sys.argv[1]]()
