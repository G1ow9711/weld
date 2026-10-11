"""B inference: model predictions alone select parents and ordered children.

No supervision import. Invalid/overflow hypotheses remain explicit false
predictions. Shared cut references identify one stored world coordinate.
"""
import math
import numpy as np

from .contract_curve_geometry import CircularArc, sample_arc, verify_sampled_arc

VERSION = "PREDICTED_WORLD_LINE_CIRCLE_SEQUENCE_RUNTIME_285_v1"
PRESENCE_THRESHOLD = .5


def line_segment(start, end):
    start, end = np.asarray(start, np.float64), np.asarray(end, np.float64)
    length = float(np.linalg.norm(end - start))
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Degenerate predicted line")
    count = int(math.ceil(length)) + 1
    if count > 1_000_001:
        raise ValueError("Predicted line exceeds registered export sample capacity")
    points = np.linspace(start, end, count, dtype=np.float64)
    points[0], points[-1] = start, end
    return dict(kind="line", points=points.tolist(), endpoints_mm=[start.tolist(), end.tolist()],
        length_mm=length, is_closed=False, sampling_audit=dict(max_step_mm=length/(count-1)))


def decode_world_predictions(output):
    # Analytic CPU scoring also imports line_segment from this module.
    # Load neural dependencies only when the neural decoder is actually used.
    import torch
    from .world_set_geometry_285 import parent_samples
    if output["version"] != "PURE_RAW_WORLD_PARENT_QUERY_CHILD_SEQUENCE_285_v1":
        raise ValueError("Registered B model output required")
    if output["input_receipt"]["original_points_consumed"] != output["input_receipt"]["original_points"]:
        raise ValueError("Inference omitted original points")
    q, capacity = output["global_parent_query_capacity"], output["global_child_sequence_capacity"]
    origin = output["origin_mm"].detach().numpy()
    probabilities = output["presence_logits"].sigmoid().detach().numpy()
    if not np.isfinite(probabilities).all() or len(probabilities) != q:
        raise ValueError("Nonfinite or incomplete parent presence outputs")
    parents, invalid = [], []
    for query in np.flatnonzero(probabilities >= PRESENCE_THRESHOLD).tolist():
        kinds_all = output["sequence_kind_logits"][query].argmax(-1).detach().numpy()
        stops = np.flatnonzero(kinds_all == 0)
        overflow = len(stops) == 0 or int(stops[0]) > capacity
        count = capacity + 1 if overflow else int(stops[0])
        mode = int(output["mode_logits"][query].argmax())
        welds = output["weld_type_logits"][query, :max(1, count)].argmax(-1).detach().numpy().tolist()
        predicted_kinds = kinds_all[:max(1, count)].tolist()
        invalid_base = dict(query_index=query, presence_probability=float(probabilities[query]),
            predicted_child_count=count, counted_false_children=max(1, count),
            predicted_weld_types=welds, predicted_kinds=predicted_kinds, capacity_overflow=overflow)
        try:
            if overflow or count == 0:
                raise ValueError("SEQUENCE_CAPACITY_OVERFLOW" if overflow else "PRESENT_PARENT_WITH_ZERO_CHILDREN")
            if not all(kind in (1, 2) for kind in predicted_kinds):
                raise ValueError("Unsupported pre-STOP child kind")
            closed = bool(output["closed_logits"][query].argmax())
            turns = int(output["turns_logits"][query].argmax())
            if turns != int(closed):
                raise ValueError("Predicted closure and winding count disagree")
            direction = int(output["direction_logits"][query].argmax()) * 2 - 1
            gaps = output["gap_logits"][query, :count-1].argmax(-1).bool()
            rows = torch.tensor([query], dtype=torch.long, device=output["presence_logits"].device)
            points, parameters = parent_samples(output, rows, mode, count, gaps[None],
                torch.tensor([direction], dtype=torch.long), torch.tensor([closed]),
                torch.tensor([predicted_kinds], dtype=torch.long))
            values = points[0].detach().numpy() + origin
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite predicted world curve")
            positive = gaps.detach().numpy().tolist()
            boundaries, segments = [], []
            for child in range(count):
                start, end = values[child, 0], values[child, -1]
                if child and not positive[child-1]:
                    start_ref = segments[-1]["boundary_refs"][1]
                    start = np.asarray(boundaries[start_ref], np.float64)
                else:
                    if child and np.array_equal(start, np.asarray(boundaries[segments[-1]["boundary_refs"][1]])):
                        raise ValueError("Positive gap collapsed to a shared cut")
                    start_ref = len(boundaries)
                    boundaries.append(start.tolist())
                if closed and child + 1 == count:
                    end_ref = segments[0]["boundary_refs"][0] if segments else start_ref
                    end = np.asarray(boundaries[end_ref], np.float64)
                else:
                    end_ref = len(boundaries)
                    boundaries.append(end.tolist())
                if predicted_kinds[child] == 1:
                    segment = line_segment(start, end)
                else:
                    if mode == 1:
                        center = parameters["center"][0].detach().numpy() + origin
                        radius = float(parameters["radius"][0])
                        normal = parameters["normal"][0].detach().numpy()
                        left, right = [float(parameters[name][0, child]) for name in ("parameter_left", "parameter_right")]
                        sweep = float(parameters["total_sweep"][0]) * (right-left)
                    elif mode == 2:
                        arc_info = parameters["mixed_arc_parameters"][child]
                        if arc_info is None:
                            raise ValueError("Missing authoritative mixed arc")
                        center = arc_info["center"][0].detach().numpy() + origin
                        radius = float(arc_info["radius"][0])
                        normal = arc_info["normal"][0].detach().numpy()
                        sweep = float(arc_info["sweep"][0])
                    else:
                        raise ValueError("Line parent cannot export an arc")
                    single_closed = closed and count == 1
                    arc = CircularArc(center, radius, normal, (start-center)/radius, sweep, single_closed)
                    segment = sample_arc(arc)
                    # Store each physical boundary once. This only reuses the
                    # same analytic definition, never aligns a prediction to GT.
                    segment["points"][0], segment["points"][-1] = boundaries[start_ref], boundaries[end_ref]
                    verify_sampled_arc(segment)
                segment.update(segment_index=child, segment_count=count,
                    weld_type="horizontal" if welds[child] == 0 else "vertical",
                    boundary_refs=[start_ref, end_ref], attributes=None, formal_E_status="UNKNOWN_UNREVIEWED")
                segments.append(segment)
            parents.append(dict(query_index=query, presence_probability=float(probabilities[query]),
                predicted_mode=mode, is_closed=closed, turns=turns, direction=direction,
                gaps=["POSITIVE_GAP" if value else "SHARED" for value in positive],
                boundaries_mm=boundaries, segments=segments, child_count=count))
        except (ValueError, RuntimeError, OverflowError) as exc:
            invalid.append(dict(**invalid_base, reason=str(exc)))
    return dict(version=VERSION, units="mm", parents=parents, invalid_predictions=invalid,
        present_parent_count=len(parents)+len(invalid), total_prediction_children=sum(p["child_count"] for p in parents)
            +sum(p["counted_false_children"] for p in invalid),
        capacity_overflow=any(p["capacity_overflow"] for p in invalid),
        input_receipt=output["input_receipt"], presence_threshold=PRESENCE_THRESHOLD,
        GT_or_STEP_reads=0, truth_count_used_for_selection=False, NMS_used=False,
        runtime_attributes_trained=False, full_contract_pass=False)
