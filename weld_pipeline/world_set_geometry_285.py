"""Differentiable true world line/circle definitions with shared parameters.

Training chooses supervised branch semantics in the loss. Runtime supplies its
own predicted semantics. Neither path receives a teacher coordinate or domain.
Coordinates returned here are relative to the pure-source origin, in mm.
"""
import math
import torch

TAU = 2 * math.pi


def quaternion_axes(quaternion):
    q = quaternion.to(torch.float64)
    if q.shape[-1] != 4 or not torch.isfinite(q).all():
        raise ValueError("Finite world-circle quaternion required")
    w, x, y, z = q.unbind(-1)
    denominator = (q * q).sum(dim=-1)
    if (denominator <= 0).any():
        raise ValueError("Zero quaternion is invalid; no default orientation")
    u = torch.stack((w*w+x*x-y*y-z*z, 2*(x*y+w*z), 2*(x*z-w*y)), dim=-1) / denominator[..., None]
    v = torch.stack((2*(x*y-w*z), w*w-x*x+y*y-z*z, 2*(y*z+w*x)), dim=-1) / denominator[..., None]
    normal = torch.cross(u, v, dim=-1)
    return u, v, normal


def parameter_intervals(weights, child_count, positive_gaps):
    if (type(child_count) is not int or child_count < 1 or weights.ndim != 3
            or weights.shape[1] < child_count or weights.shape[2] != 2):
        raise ValueError("Explicit finite ordered child intervals required")
    if positive_gaps.shape != (len(weights), child_count - 1):
        raise ValueError("Every predicted/supervised adjacency needs its own gap state")
    values = weights[:, :child_count].to(torch.float64)
    if not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Finite strictly positive interval weights required")
    lengths = values[:, :, 0]
    gaps = values[:, :-1, 1] * positive_gaps.to(torch.float64)
    denominator = lengths.sum(dim=1) + gaps.sum(dim=1)
    lengths, gaps = lengths / denominator[:, None], gaps / denominator[:, None]
    cursor = torch.zeros(len(weights), dtype=torch.float64, device=weights.device)
    left, right = [], []
    for child in range(child_count):
        left.append(cursor)
        end = cursor + lengths[:, child] if child + 1 < child_count else torch.ones_like(cursor)
        right.append(end)
        if child + 1 < child_count:
            cursor = end + gaps[:, child]
    return torch.stack(left, dim=1), torch.stack(right, dim=1)


def coupled_mixed_arc(start, end, raw_normal, sweep_logit, directions, fractions):
    start, end, raw_normal = [value.to(torch.float64) for value in (start, end, raw_normal)]
    chord = end - start
    distance = torch.linalg.vector_norm(chord, dim=-1)
    if (distance <= 0).any():
        raise ValueError("Open endpoint-coupled arc cannot have coincident endpoints")
    axis = chord / distance[:, None]
    normal = raw_normal - (raw_normal * axis).sum(dim=-1)[:, None] * axis
    norm = torch.linalg.vector_norm(normal, dim=-1)
    if (norm <= 0).any() or not torch.isfinite(norm).all():
        raise ValueError("Degenerate predicted chord-plane orientation; no default normal")
    normal = normal / norm[:, None]
    beta = TAU * torch.sigmoid(sweep_logit.to(torch.float64))
    if (beta <= 0).any() or (beta >= TAU).any():
        raise ValueError("Open arc sweep must be strictly between zero and one turn")
    sine = torch.sin(beta / 2)
    center = ((start + end) / 2 + directions[:, None] * (distance / 2 * torch.cos(beta / 2) / sine)[:, None]
              * torch.cross(normal, axis, dim=-1))
    radius = distance / (2 * sine)
    u = (start - center) / radius[:, None]
    v = torch.cross(normal, u, dim=-1)
    angles = directions[:, None] * beta[:, None] * fractions[None]
    points = center[:, None] + radius[:, None, None] * (
        torch.cos(angles)[:, :, None] * u[:, None] + torch.sin(angles)[:, :, None] * v[:, None])
    # Endpoint definitions are the existing shared boundaries. Interior points
    # use the same authoritative circle, rather than a second endpoint head.
    points = torch.cat((start[:, None], points[:, 1:-1], end[:, None]), dim=1)
    return points, dict(center=center, radius=radius, normal=normal, u=u, sweep=directions * beta)


def parent_samples(output, rows, mode, child_count, positive_gaps, directions, closed, child_kinds, sample_count=17):
    """Construct hypothesis geometry. No ground-truth numeric values accepted."""
    if mode not in (0, 1, 2) or rows.dtype != torch.long or rows.ndim != 1:
        raise ValueError("Declared parent mode and query rows required")
    if child_count > output["global_child_sequence_capacity"] or child_count < 1:
        raise ValueError("Global sequence capacity exceeded; cannot truncate")
    if directions.shape != (len(rows),) or closed.shape != (len(rows),) or child_kinds.shape != (len(rows), child_count):
        raise ValueError("Explicit ordered geometry semantics required")
    if not torch.all((directions == -1) | (directions == 1)):
        raise ValueError("Circle traversal direction must be explicit")
    fractions = torch.linspace(0., 1., sample_count, dtype=torch.float64, device=rows.device)
    positive = positive_gaps.to(torch.bool)
    left, right = parameter_intervals(output["interval_weights"][rows], child_count, positive)
    parameter = left[:, :, None] + (right - left)[:, :, None] * fractions[None, None]
    if mode == 0:
        if closed.any() or (child_kinds != 1).any():
            raise ValueError("LINE_CHAIN cannot close or contain an arc")
        a, b = output["line_endpoints_m"][rows].to(torch.float64).unbind(1)
        points = 1000 * (a[:, None, None] + parameter[:, :, :, None] * (b-a)[:, None, None])
        return points, dict(parameter_left=left, parameter_right=right)
    if mode == 1:
        if (child_kinds != 2).any():
            raise ValueError("CIRCLE_CHAIN child must have predicted arc type")
        u, v, normal = quaternion_axes(output["circle_quaternion"][rows])
        center = output["circle_center_m"][rows].to(torch.float64) * 1000
        radius = output["circle_radius_m"][rows].to(torch.float64) * 1000
        if (radius <= 0).any() or not torch.isfinite(radius).all():
            raise ValueError("World circle radius must be positive and finite")
        total = torch.where(closed, torch.full_like(radius, TAU), output["circle_open_sweep_rad"][rows].to(torch.float64))
        phase = output["circle_phase_rad"][rows].to(torch.float64)
        angles = phase[:, None, None] + directions[:, None, None] * total[:, None, None] * parameter
        points = center[:, None, None] + radius[:, None, None, None] * (
            torch.cos(angles)[:, :, :, None] * u[:, None, None] + torch.sin(angles)[:, :, :, None] * v[:, None, None])
        last = torch.where(closed[:, None], points[:, 0, 0], points[:, -1, -1])
        final_child = torch.cat((points[:, -1, :-1], last[:, None]), dim=1)
        points = torch.cat((points[:, :-1], final_child[:, None]), dim=1)
        return points, dict(center=center, radius=radius, normal=normal, u=u, v=v, phase=phase,
                            total_sweep=directions * total, parameter_left=left, parameter_right=right)
    segments, parameters = [], []
    raw = output["mixed_endpoints_m"][rows, :child_count].to(torch.float64) * 1000
    first_boundary = raw[:, 0, 0]
    previous = None
    for child in range(child_count):
        start, end = raw[:, child].unbind(1)
        if child:
            start = torch.where(positive[:, child-1, None], start, previous)
        if child + 1 == child_count:
            end = torch.where(closed[:, None], first_boundary, end)
        points = start[:, None] + fractions[None, :, None] * (end-start)[:, None]
        arc_rows = child_kinds[:, child] == 2
        arc_info = None
        if arc_rows.any():
            raw_arc = output["mixed_arc_parameters"][rows, child]
            arc_directions = output["mixed_direction_logits"][rows, child].argmax(dim=-1) * 2 - 1
            arc_points, arc_info = coupled_mixed_arc(start[arc_rows], end[arc_rows], raw_arc[arc_rows, :3],
                raw_arc[arc_rows, 3], arc_directions[arc_rows], fractions)
            points = points.clone()
            points[arc_rows] = arc_points
        segments.append(points)
        parameters.append(arc_info)
        previous = end
    return torch.stack(segments, dim=1), dict(mixed_arc_parameters=parameters, parameter_left=left, parameter_right=right)
