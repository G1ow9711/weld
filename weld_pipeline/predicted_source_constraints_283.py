"""Pure prediction/source constraint decoding. No labels or teacher sidecars.

The parent axis is redundant with its predicted straight world traversal.
Project it to that traversal's subspace; reject its original wrong orientation.
All endpoints, reference identities, phases, gaps and source permissions remain
unchanged, and the original complete world binder still decides validity.
"""
from copy import deepcopy
import math
from .decoder_graph_structure_254 import slot
from .mixed_parent_graph_260 import rational
from .mixed_parent_source_binding_260 import FrozenMixedGraphSourceBinder
from .mixed_arc_restrictions_260 import RestrictedSourceArc, matching_points
from .raw_triangle_world_charts_258 import dot, sub
from .concrete_curve_certificate_250 import UnprovenCurve
from .typed_prediction_codec_281 import decode_prediction

VERSION = "PREDICTED_PARENT_AXIS_ORTHOGONAL_SOURCE_CONSTRAINT_283_v1"


def exact_record(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


class PredictionSourceConstraints(FrozenMixedGraphSourceBinder):
    def _prediction_coordinates(self, structural):
        """Same original coordinate and permission rules; no proximity joins."""
        coordinates, local_arcs, world_arcs = {}, {}, {}
        for key, prototype in structural["local_arcs"].items():
            domain = structural["span_domains"][key]
            interval = structural["arc_intervals"][key]
            local_arcs[key] = RestrictedSourceArc(prototype, interval)
            world_arcs[key] = RestrictedSourceArc(prototype, interval, context=self._context(domain[1]))
            world_arcs[key].circle_proof()
        for key, master in structural["arc_authorities"].items():
            if world_arcs[key].signature() != world_arcs[master].signature():
                raise UnprovenCurve("Prediction changes its authoritative source circle or phase frame")
        for identity in structural["coordinates"]:
            chain, current = [], identity
            while current not in coordinates:
                chain.append(current)
                row = structural["coordinates"][current]
                if "physical_ref" not in row:
                    break
                current = slot(row["physical_ref"])
            for key in reversed(chain):
                row, domain = structural["coordinates"][key], structural["coordinate_domains"][key]
                endpoint = row.get("curve_endpoint", row.get("local_position", {}).get("curve_endpoint"))
                coordinates[key] = (
                    self._arc_position(key, domain, endpoint, local_arcs, world_arcs) if endpoint is not None
                    else self._position(key, domain, row.get("position", row.get("local_position"))))
                if "physical_ref" in row and not matching_points(coordinates[key], coordinates[slot(row["physical_ref"])]):
                    raise UnprovenCurve("Prediction's physical reference joins different exact source world expressions")
        return coordinates

    def project_predicted_parent_axes(self, graph):
        structural = self.validator.validate(graph)
        coordinates = self._prediction_coordinates(structural)
        projected = deepcopy(graph)
        changes = []
        for parent in projected["parents"]:
            if parent["frame"]["kind"] != "line":
                continue
            first = coordinates[slot(parent["start_boundary"])]["point_exact"]
            last = coordinates[slot(parent["end_boundary"])]["point_exact"]
            if first is None or last is None:
                raise UnprovenCurve("A predicted line frame requires exact source world endpoint expressions")
            delta = sub(last, first)
            norm_squared = dot(delta, delta)
            if norm_squared <= 0:
                raise UnprovenCurve("A predicted line parent has zero world endpoint separation")
            axis = tuple(map(rational, parent["frame"]["axis_world"]))
            oriented_delta = tuple(parent["direction"] * value for value in delta)
            numerator = dot(axis, oriented_delta)
            if numerator <= 0:
                raise UnprovenCurve("Predicted parent axis has wrong orientation; constraint decoding cannot flip it")
            factor = numerator / norm_squared
            constrained = tuple(factor * value for value in oriented_delta)
            difference = tuple(a - b for a, b in zip(axis, constrained))
            assert dot(difference, oriented_delta) == 0
            assert dot(constrained, oriented_delta) > 0
            parent["frame"]["axis_world"] = [exact_record(value) for value in constrained]
            changes.append(dict(parent_slot=deepcopy(parent["slot"]),
                changed=any(difference), axis_projection_change=math.sqrt(float(dot(difference, difference))),
                original_prediction_axis=[exact_record(value) for value in axis],
                constrained_axis=[exact_record(value) for value in constrained],
                original_direction=parent["direction"], direction_changed=False,
                predicted_endpoint_positions_changed=False, original_axis_orientation_preserved=True))
        return projected, dict(version=VERSION, parent_axis_constraints=changes,
            affected_parent_axes=len(changes), changed_parent_axes=sum(v["changed"] for v in changes),
            maximum_axis_projection_change=max((v["axis_projection_change"] for v in changes), default=0.0),
            teacher_or_STEP_reads=0, exact_teacher_sidecar_used=False,
            source_role_capacity_changed=False, endpoints_or_world_kind_changed=False,
            phase_direction_allowed_domains_or_gaps_changed=False,
            geometry_acceptance_relaxed=False, actual_optimizer_updates=0, full_contract_pass=False)

    def bind_prediction(self, prediction, domain_record):
        # Codec rejects extra keys, saved teacher graphs and exact sidecars.
        graph = decode_prediction(prediction, self.validator.anchor_count, domain_record)
        projected, trace = self.project_predicted_parent_axes(graph)
        result = super().bind(projected)
        return dict(predicted_graph=projected, prediction_constraint_trace=trace, source_world_binding=result,
            status="PASS_PREDICTED_ARRAYS_AND_ORIGINAL_SOURCE_CONSTRAINT_BINDING",
            neural_model_predictions_verified=False, actual_optimizer_updates=0, full_contract_pass=False)
