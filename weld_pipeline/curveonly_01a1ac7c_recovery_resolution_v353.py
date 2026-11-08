"""Keep the prior model when automatic recovery promotes unresolved curvature.

Only the NEW geometry recovery is guarded. There is no minimum weld length,
no original prediction is deleted, and unknown curvature is not a negative
welding label. The criterion uses the existing native projection precision.
"""
import copy
import types
import numpy as np
from . import curveonly_01a1ac7c_edge_wall_runtime_v350 as recovery
from . import curveonly_01a1ac7c_rule_runtime_v345 as baseline
from . import curveonly_01a1ac7c_structural_event_cells_v298 as numeric


def curvature_observation(candidate):
    p = np.asarray(candidate['points'], float)
    chord = p[-1]-p[0]; length = float(np.linalg.norm(chord))
    if length <= numeric.NUMERIC_MM:
        bend = float(np.linalg.norm(p-p[0], axis=1).max())
    else:
        bend = float(np.linalg.norm(np.cross(p-p[0], chord/length), axis=1).max())
    fit = candidate.get('circle') or {}
    uncertainty = max(numeric.NUMERIC_MM, float(fit.get('max_fit_error_mm', 0.)))
    return dict(observed_bend_mm=bend, native_uncertainty_mm=uncertainty,
        numerically_resolved_curvature=bend > uncertainty,
        no_minimum_weld_length=True)


def resolve(decoded, packet, fallback):
    audit = decoded.get('structural_rule_v338', {})
    observations = [dict(candidate=c['candidate'], **curvature_observation(packet['candidates'][c['candidate']]))
        for c in audit.get('edge_wall_certificates', [])]
    invalid = [q for q in observations if not q['numerically_resolved_curvature']]
    if invalid:
        # The preceding whole output is retained atomically. This withdraws the
        # new recovery permission and preserves all existing short predictions,
        # interval decisions and the already qualified transverse-plate rules.
        result = copy.deepcopy(fallback())
    else:
        result = copy.deepcopy(decoded)
    result['automatic_recovery_resolution_guard'] = dict(observations=observations,
        unresolved_additions=invalid, retained_previous_complete_output=bool(invalid),
        selected_route='V345_PREVIOUS' if invalid else 'V350_EDGE_WALL_RECOVERY',
        reason='UNRESOLVED_CURVATURE_RECOVERY_NOT_CERTIFIED' if invalid else 'RECOVERY_CURVATURE_RESOLVED',
        old_predictions_not_length_filtered=True, projection_numeric_mm=numeric.NUMERIC_MM,
        no_answer_name_or_fixed_coordinate_lookup=True)
    return result


def decode(features, packet, model, solids=None, inventory=None):
    current = recovery.decode(features, packet, model, solids, inventory)
    return resolve(current, packet, lambda:baseline.decode(features, packet, model, solids, inventory))


def extract(input_record, model):
    fn = types.FunctionType(baseline.extract.__code__, dict(baseline.__dict__, decode=decode))
    return fn(input_record, model)
