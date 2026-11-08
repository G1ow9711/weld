"""Recheck finite identity with the existing explicit point-fallback schema."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import types,sys,subprocess
from pathlib import Path
from tools import curveonly_01a1ac7c_finite_curve_alias_pilot_v501 as previous
from weld_pipeline import curveonly_01a1ac7c_finite_curve_alias_v502 as rules

e,common=previous.e,previous.common
DEST=e.REC/'finite_curve_alias_pilot_v503';OUTPUT=e.OUT/'finite_curve_alias_pilot_v503'
ns=dict(previous.__dict__,__file__=__file__,DEST=DEST,OUTPUT=OUTPUT,rules=rules)
observe=types.FunctionType(previous.observe.__code__,ns)


def prepare():
    assert e.read(previous.DEST/'CLOSURE.json')['analysis_closed']
    types.FunctionType(previous.prepare.__code__,ns)()
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for path in [Path(previous.__file__).resolve(),Path(rules.previous.__file__).resolve()]:
        assert path.read_bytes()==subprocess.check_output(['git','show',p['source_commit']+':'+str(path.relative_to(e.ROOT))])
        p['sources'].append(e.bind(path))
    p['partial_original_pilot']=e.bind(previous.DEST/'CLOSURE.json')
    p['only_conversion_schema_view_changed']=True
    e.write(DEST/'FROZEN_PROTOCOL.json',p)


if __name__=='__main__':{'prepare':prepare,'observe':observe}[sys.argv[1]]()
