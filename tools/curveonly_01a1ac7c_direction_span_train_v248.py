"""Same original169-instance objective; only134 observed channels are augmented."""
import types
from tools import curveonly_01a1ac7c_ordered_span_train_v236 as previous
from weld_pipeline import curveonly_01a1ac7c_direction_spans_v247 as m

SETTINGS = dict(previous.SETTINGS,observations=134,
    initialization='samefresh111sharedparameters_plus_zero23inputweights',
    original_gradient_objective_and_training_budget_unchanged=True)
configure = types.FunctionType(previous.configure.__code__,dict(previous.__dict__,SETTINGS=SETTINGS))
assemble = types.FunctionType(previous.assemble.__code__,dict(previous.__dict__,m=m))
loss_terms = previous.loss_terms
objective = previous.objective
namespace = dict(previous.__dict__,m=m,SETTINGS=SETTINGS,configure=configure,assemble=assemble)
assembly_control = types.FunctionType(previous.assembly_control.__code__,namespace)
fit = types.FunctionType(previous.fit.__code__,namespace)
