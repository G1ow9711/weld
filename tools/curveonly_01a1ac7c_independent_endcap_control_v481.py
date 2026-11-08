"""Full unchanged-V60 control of independently verified finite endcap recovery."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import types,sys
from tools import curveonly_01a1ac7c_endcap_control_v474 as previous
from tools import curveonly_01a1ac7c_independent_endcap_pilot_v480 as pilot
from weld_pipeline import curveonly_01a1ac7c_independent_endcap_recovery_v477 as rules
from weld_pipeline import curveonly_01a1ac7c_independent_endcap_runtime_v479 as runtime

e,common=previous.e,previous.common
DEST=e.REC/'independent_endcap_control_v481';OUTPUT=e.OUT/'independent_endcap_control_v481'
ns=dict(previous.__dict__,__file__=__file__,DEST=DEST,OUTPUT=OUTPUT,pilot=pilot,rules=rules,runtime=runtime)
for name in ['connection_key','check','prepare','one','predict','score']:
    ns[name]=types.FunctionType(getattr(previous,name).__code__,ns)
connection_key,check,one,predict,score=[ns[name] for name in ['connection_key','check','one','predict','score']]


def prepare():
    # The template checks the original failed controls; replace only that
    # prerequisite read, preserving all immutable first-attempt evidence.
    original_read=e.read
    original_bind=e.bind
    def read(path):
        if str(path).endswith('/endcap_controls_v473/CLOSURE.json'):
            return original_read(e.REC/'independent_endcap_controls_v478/CLOSURE.json')
        return original_read(path)
    def bind(path):
        if str(path).endswith('/endcap_controls_v473/CLOSURE.json'):
            return original_bind(e.REC/'independent_endcap_controls_v478/CLOSURE.json')
        return original_bind(path)
    proxy=types.SimpleNamespace(**dict(e.__dict__,read=read,bind=bind));prepared_ns=dict(ns,e=proxy)
    types.FunctionType(previous.prepare.__code__,prepared_ns)()
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    p['sources'] += [e.bind(previous.__file__),e.bind(rules.previous.__file__)]
    p['controls']=e.bind(e.REC/'independent_endcap_controls_v478/CLOSURE.json')
    p['original_independent_control_failure']=e.bind(e.REC/'endcap_controls_v473/ACTUAL_TERMINAL.json')
    p['finite_endcap_query_policy']='Independent actual finite cap even when generic ray already succeeds'
    e.write(DEST/'FROZEN_PROTOCOL.json',p)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
