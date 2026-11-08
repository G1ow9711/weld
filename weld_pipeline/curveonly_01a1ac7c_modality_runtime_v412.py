"""Choose a frozen curve expert using available, qualified input geometry."""
import types
from . import curveonly_01a1ac7c_paired_face_runtime_v408 as baseline
from . import curveonly_01a1ac7c_runtime_v56 as point_runtime


def qualified_step(packet,solids):
    return bool(solids and packet.get('pose',{}).get('passed') and
        any(c.get('STEP') and c.get('pose') for c in packet['candidates']))


class ModalityCurveModel:
    def __init__(self,paired,point,origin):
        self.paired=paired
        self.point=point
        self.origin=origin
        self.last_route=None

    @property
    def last_states(self):
        return self.paired.last_states if self.last_route=='QUALIFIED_FINITE_STEP' else None

    @property
    def last_timing(self):
        return self.paired.last_timing if self.last_route=='QUALIFIED_FINITE_STEP' else self.point.last_timing


def decode(features,packet,model,solids=None,inventory=None):
    if qualified_step(packet,solids):
        model.last_route='QUALIFIED_FINITE_STEP'
        result=baseline.decode(features,packet,model.paired,solids,inventory)
    else:
        model.last_route='POINT_OBSERVATION_EXPERT'
        width=len(baseline.rules.expert.graph.cad.NAMES)
        result=point_runtime.decode(features[:,:width],packet,model.point)
    result['curve_input_modality_rule']=dict(branch='INPUT_GEOMETRY_AVAILABILITY_V412',
        selected=model.last_route,pose_qualified=bool(packet.get('pose',{}).get('passed')),
        no_qualified_geometry_not_a_negative_weld_target=True,
        no_name_coordinate_size_or_count_lookup=True,no_new_fit_or_VALcalibration=True)
    return result


def extract(input_record,model):
    return types.FunctionType(baseline.root.extract.__code__,dict(baseline.root.__dict__,decode=decode))(input_record,model)
