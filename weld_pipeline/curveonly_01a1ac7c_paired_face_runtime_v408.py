"""Variable-count curve model with a frozen source-side agreement expert."""
import types
from . import curveonly_01a1ac7c_partial_contact_runtime_v385 as baseline
from . import curveonly_01a1ac7c_paired_face_rules_v407 as rules

root=baseline.baseline.baseline.backbone


class PairedCurveModel:
    def __init__(self,primary,source_count,origin):
        self.primary=primary
        self.source_expert=types.SimpleNamespace(count=source_count)
        self.component_origin=origin

    @property
    def last_states(self):return self.primary.last_states

    @property
    def last_timing(self):return self.primary.last_timing

    def predict(self,features,packet):return self.primary.predict(features,packet)


def decode(features,packet,model,solids=None,inventory=None):
    if solids and inventory is None:inventory=root.geometry.build_geometry(packet,solids)
    before=baseline.decode(features,packet,model.primary,solids,inventory)
    return rules.correct(before,features,packet,inventory,solids,model.source_expert)


def extract(input_record,model):
    return types.FunctionType(root.extract.__code__,dict(root.__dict__,decode=decode))(input_record,model)
