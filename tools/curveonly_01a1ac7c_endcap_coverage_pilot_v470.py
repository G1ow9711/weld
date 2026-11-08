"""Same all-input finite-endcap proposal, refined native-join uncertainty only."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import sys,time,types,subprocess
from pathlib import Path
from tools import curveonly_01a1ac7c_endcap_connection_pilot_v468 as previous
from weld_pipeline import curveonly_01a1ac7c_endcap_model_coverage_v469 as rules

e,common=previous.e,previous.common
DEST=e.REC/'endcap_coverage_pilot_v470'


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(previous.DEST/'CLOSURE.json')['analysis_closed']
    p=e.read(previous.DEST/'FROZEN_PROTOCOL.json')
    for binding in p['sources']+[p['inputs'],p['model_outputs']]:common.valid(binding)
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(rules.__file__).resolve()]
    for source in sources:assert source.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(source.relative_to(e.ROOT))])
    DEST.mkdir(exist_ok=True)
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(p,source_commit=commit,sources=[e.bind(s) for s in sources]+p['sources'],
        change='Only allow existing model endpoint uncertainty at literal zero-gap native joins with actual finite endcap witnesses to the same opposite complete model unit.',
        previous_failed_proposal=e.bind(previous.DEST/'CLOSURE.json'),no_source_or_process_gap_threshold_change=True,epoch=time.time()))


def observe():
    fn=types.FunctionType(previous.observe.__code__,dict(previous.__dict__,DEST=DEST,rules=rules))
    return fn()


if __name__=='__main__':{'prepare':prepare,'observe':observe}[sys.argv[1]]()
