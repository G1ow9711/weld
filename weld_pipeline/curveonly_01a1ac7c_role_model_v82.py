"""Eight finite material-role scalars appended to the exact R27 encoder."""
import types
import numpy as np
from . import curveonly_01a1ac7c_set_v18 as base
from . import curveonly_01a1ac7c_graph_context_v65 as graph
from . import curveonly_01a1ac7c_material_roles_v81 as roles


def candidate_context(packet):
    values=np.asarray(packet['finite_material_roles_v81']['values'],np.float32).reshape(-1,len(roles.NAMES))
    assert values.shape==(len(packet['candidates']),len(roles.NAMES)) and np.isfinite(values).all()
    return np.column_stack([graph.candidate_context(packet),values])


def pooled(values,context):
    return np.r_[base.pooled(values),context].astype(np.float32)


class RoleCurveSet(base.CurveSet):
    def encode(self,features,packet):
        context=candidate_context(packet)
        return np.asarray([pooled(features[a:b],context[ci])
            for ci,(a,b) in enumerate(zip(packet['offsets'][:-1],packet['offsets'][1:]))],np.float32).reshape(-1,197)

    def predict(self,features,packet):
        encoded=self.encode(features,packet);rows=iter(encoded)
        if not len(encoded):
            self.last_timing=dict(pooled_features=197,query_features=200,input_curves=0,
                interval_queries=0,candidate_context_features=12,material_role_features=8,
                new_geometric_queries=0,cpu_seconds=0.)
            return []
        decode=types.FunctionType(base.CurveSet.predict.__code__,dict(base.__dict__,pooled=lambda unused:next(rows)))
        result=decode(self,features,packet);assert next(rows,None) is None
        if self.last_timing is not None:
            self.last_timing.update(pooled_features=197,query_features=200,candidate_context_features=12,
                material_role_features=8,material_role_source='INPUT_ONLY_FINITE_MATERIAL_OBSERVATIONS',new_geometric_queries=0)
        return result
