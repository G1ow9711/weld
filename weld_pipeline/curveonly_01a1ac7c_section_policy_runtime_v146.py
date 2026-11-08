"""Raw XYZ/material input and the independently selected curve-only policy."""
from . import curveonly_01a1ac7c_section_local_v135 as section
from . import curveonly_01a1ac7c_local_runtime_v112 as runtime
from .curveonly_01a1ac7c_feature_policy_v144 import FeaturePolicy

input_curves = section.input_curves
arclength = section.arclength


def decode(features, packet, model, threshold=.5):
    assert isinstance(model, FeaturePolicy)
    assert features.ndim == 2 and features.shape[1] == section.INPUT_COLUMNS
    return runtime.decode(features, packet, model, threshold)
