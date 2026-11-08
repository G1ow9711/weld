"""R35 frozen network and publisher with segmentation-invariant source rules."""
from .curveonly_01a1ac7c_direction_runtime_v261 import (
    NAMES, OBSERVATIONS, OrderedSpanNetwork, OrderedSpanSet, candidate_rows, input_curves,
)
from . import curveonly_01a1ac7c_runtime_v51 as exporter
from . import curveonly_01a1ac7c_corner_publication_v256 as corner
from . import curveonly_01a1ac7c_arc_publication_v6 as publication
from .curveonly_01a1ac7c_native_coverage_selection_v294 import RuleSelection


def decode(features, packet, model, threshold=.5):
    proxy = RuleSelection(model)
    old = publication._fallback
    publication._fallback = corner.fallback
    try:
        result = exporter.decode(features, packet, proxy, threshold)
    finally:
        publication._fallback = old
    result['finite_source_selection'] = proxy.selection
    result['rule_algorithm'] = 'FINITE_SOURCE_NATIVE_COVERAGE_INTEGRAL_V294'
    return result
