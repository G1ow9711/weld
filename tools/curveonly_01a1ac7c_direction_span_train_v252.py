"""134-dimensional version of the unchanged TRAIN midrange normalizer."""
import types
from tools import curveonly_01a1ac7c_ordered_span_train_v236 as original
from tools import curveonly_01a1ac7c_direction_span_train_v248 as previous
from weld_pipeline import curveonly_01a1ac7c_direction_spans_v247 as m

SETTINGS = dict(previous.SETTINGS,normalizer_dimensions=134)
configure = types.FunctionType(original.configure.__code__,dict(original.__dict__,SETTINGS=SETTINGS))

# The inherited codec is intentionally shared for exact interval semantics.
# Give only TRAIN assembly a private normalizer namespace; never mutate the
# original111 codec or its module-level numerical constants.
normalizer = types.FunctionType(m.codec.previous.fit_range_normalizer.__code__,
    dict(m.codec.previous.__dict__,OBSERVATIONS=134,PERMUTATION=m.PERMUTATION))
normalizer_codec_previous = types.ModuleType('curveonly_private_134_normalizer')
normalizer_codec_previous.__dict__.update(m.codec.previous.__dict__)
normalizer_codec_previous.fit_range_normalizer = normalizer
normalizer_codec = types.ModuleType('curveonly_private_134_assembly_codec')
normalizer_codec.__dict__.update(m.codec.__dict__)
normalizer_codec.previous = normalizer_codec_previous
assembly_model = types.ModuleType('curveonly_private_134_assembly_model')
assembly_model.__dict__.update(m.__dict__)
assembly_model.codec = normalizer_codec
assemble = types.FunctionType(original.assemble.__code__,dict(original.__dict__,m=assembly_model))

loss_terms,objective = original.loss_terms,original.objective
namespace = dict(original.__dict__,m=m,SETTINGS=SETTINGS,configure=configure,assemble=assemble)
assembly_control = types.FunctionType(original.assembly_control.__code__,namespace)
fit = types.FunctionType(original.fit.__code__,namespace)
