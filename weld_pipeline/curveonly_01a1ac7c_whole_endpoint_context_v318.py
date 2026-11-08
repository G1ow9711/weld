"""Preserve the native terminal occurrence at a closed path's0/L location."""
import inspect
import types
from . import curveonly_01a1ac7c_whole_intervals_v316 as previous

NAMES, SETTINGS, SCALARS = previous.NAMES, previous.SETTINGS, previous.SCALARS
base, profile, provenance, TOL = previous.base, previous.profile, previous.provenance, previous.TOL
pairs, fields, select = previous.pairs, previous.fields, previous.select

_source = inspect.getsource(previous.features)
for _old, _new in [
    ('np.vstack([event_context[:-1], event_context])', 'np.vstack([event_context, event_context[1:]])'),
    ('np.vstack([xyz[:-1], xyz])', 'np.vstack([xyz, xyz[1:]])'),
    ('np.vstack([tangent[:-1], tangent])', 'np.vstack([tangent, tangent[1:]])'),
]:
    assert _source.count(_old) == 1
    _source = _source.replace(_old, _new)
_namespace = dict(previous.__dict__, __file__=__file__, __name__=__name__)
exec(compile(_source, __file__, 'exec'), _namespace)
features = _namespace['features']


class WholeIntervalModel(previous.WholeIntervalModel):
    predict = types.FunctionType(previous.WholeIntervalModel.predict.__code__,
        dict(previous.__dict__, features=features))


decode, input_curves = previous.decode, previous.input_curves
