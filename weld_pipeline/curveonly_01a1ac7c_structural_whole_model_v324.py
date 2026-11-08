"""Whole-range validity from structural inputs, without learned-state shortcuts."""
import types
import numpy as np
from . import curveonly_01a1ac7c_whole_endpoint_context_v318 as previous

NAMES, SETTINGS = previous.NAMES, previous.SETTINGS
base, profile, provenance, TOL = previous.base, previous.profile, previous.provenance, previous.TOL
pairs, fields, select = previous.pairs, previous.fields, previous.select
REMOVED = (272,286,288,289,290,291,292,313,314,315,316,317,318,319,320,323,324,325)
assert len(REMOVED) == 18
assert NAMES[272] == 'start_event:prior_boundary' and NAMES[286] == 'end_event:prior_boundary'
assert NAMES[293] == 'native_total_log_length_mm'
ACTIVE = tuple(i for i in range(len(NAMES)) if i not in REMOVED)


def project(features):
    result = np.asarray(features,np.float32).copy(); result[:,REMOVED] = 0.
    return result


def features(*args,**kwargs):
    values,indexes,bounds = previous.features(*args,**kwargs)
    return project(values),indexes,bounds


class WholeIntervalModel(previous.WholeIntervalModel):
    predict = types.FunctionType(previous.previous.WholeIntervalModel.predict.__code__,
        dict(previous.previous.__dict__,features=features))


decode,input_curves = previous.decode,previous.input_curves
