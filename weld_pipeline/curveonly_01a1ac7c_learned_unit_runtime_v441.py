"""Current curve model followed by frozen learned native unit arbitration."""
import types
from . import curveonly_01a1ac7c_paired_wall_runtime_v435 as baseline
from . import curveonly_01a1ac7c_direction_spans_v247 as neural
from . import curveonly_01a1ac7c_learned_unit_rules_v440 as rules


def actual_raw(features,packet,model):
    rows=neural.candidate_rows(features,packet)
    states=model.paired.primary.last_states
    assert [s['candidate'] for s in states]==[r['candidate'] for r in rows]
    raw=neural.codec.proposals(packet,rows,[s['codes'] for s in states])
    raw,receipt=neural.normalize(raw,packet)
    assert receipt==model.paired.primary.last_representation_normalization
    return raw


def decode(features,packet,model,solids=None,inventory=None):
    before=baseline.decode(features,packet,model,solids,inventory)
    if model.last_route!='QUALIFIED_FINITE_STEP':return before
    return rules.correct(before,features,packet,actual_raw(features,packet,model),model.point)


def extract(input_record,model):
    root=baseline.baseline.baseline.baseline.root
    return types.FunctionType(root.extract.__code__,dict(root.__dict__,decode=decode))(input_record,model)
