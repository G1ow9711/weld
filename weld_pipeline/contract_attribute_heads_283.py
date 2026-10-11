"""All14 contract attributes from predicted-child embeddings and query phase.

Plate1/plate2 are fixed named outputs. There is no tangent-based side exchange,
absolute normal dot product, STEP reader, workpiece lookup or GT feature input.
Unconfirmed label values are masked in the loss, not invented as zero.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

VERSION = "FIXED_SIDE_ALL14_CHILD_ATTRIBUTE_NEURAL_INTERFACE_283_v1"
FIELDS = ("thickness_1", "thickness_2", "normal_1", "normal_2", "front_wall", "back_wall",
          "start_height", "end_height", "start_hole_height", "end_hole_height",
          "start_base_edge", "end_base_edge", "has_eave", "side_plate_split")
CONTINUOUS_FIELDS = ("thickness_1", "thickness_2", "front_wall", "back_wall", "start_height", "end_height",
                     "start_hole_height", "end_hole_height", "start_base_edge", "end_base_edge")
NORMAL_FIELDS = ("normal_1", "normal_2")
BINARY_FIELDS = ("has_eave", "side_plate_split")
STATES = ("value", "no_object", "out_of_range", "not_applicable", "unknown")
UNKNOWN = STATES.index("unknown")


class ContractAttributeHeads(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.query = nn.Sequential(nn.Linear(hidden + 10, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.continuous = nn.Linear(hidden, len(CONTINUOUS_FIELDS))
        self.normal = nn.Linear(hidden, 2 * 3)
        self.binary = nn.Linear(hidden, 2 * 2)
        self.state = nn.Linear(hidden, len(FIELDS) * len(STATES))

    def forward(self, predicted_child_embeddings, sample_fractions, predicted_xyz_m, predicted_tangents):
        count = len(predicted_child_embeddings)
        if predicted_child_embeddings.shape != (count, self.hidden) or sample_fractions.shape != (count,):
            raise ValueError("Predicted-child context and every requested curve sample required")
        if (not torch.isfinite(predicted_child_embeddings).all() or not torch.isfinite(sample_fractions).all()
            or ((sample_fractions < 0) | (sample_fractions > 1)).any()):
            raise ValueError("Finite prediction query phase in [0,1] required")
        if predicted_child_embeddings.device != sample_fractions.device or predicted_child_embeddings.dtype != sample_fractions.dtype:
            raise ValueError("Attribute context and query phase must share dtype and device")
        for values in (predicted_xyz_m, predicted_tangents):
            if (values.shape != (count, 3) or values.device != predicted_child_embeddings.device
                or values.dtype != predicted_child_embeddings.dtype or not torch.isfinite(values).all()):
                raise ValueError("Every attribute query requires its actual finite predicted curve point and tangent")
        if (torch.abs(torch.linalg.vector_norm(predicted_tangents, dim=-1) - 1) > 0.001).any():
            raise ValueError("Predicted query tangent must be a defined unit vector, without a fallback direction")
        phase = torch.stack((sample_fractions, 1 - sample_fractions,
                             torch.sin(2 * math.pi * sample_fractions), torch.cos(2 * math.pi * sample_fractions)), dim=-1)
        hidden = self.query(torch.cat((predicted_child_embeddings, phase, predicted_xyz_m, predicted_tangents), dim=-1))
        raw_normals = self.normal(hidden).reshape(count, 2, 3)
        return dict(continuous_m=F.softplus(self.continuous(hidden)), raw_normals=raw_normals,
            normals=F.normalize(raw_normals, dim=-1, eps=1e-12),
            binary_logits=self.binary(hidden).reshape(count, 2, 2),
            state_logits=self.state(hidden).reshape(count, len(FIELDS), len(STATES)))


def attribute_supervised_loss(output, targets, *, material_side_labels_frozen):
    """Status/value masks are label metadata only. All absent GT stays absent."""
    if type(material_side_labels_frozen) is not bool:
        raise ValueError("Side-schema state must be explicit boolean label metadata")
    states = targets["states"]
    known = targets["state_known"]
    count = len(states)
    if states.shape != (count, len(FIELDS)) or known.shape != states.shape or known.dtype != torch.bool or states.dtype != torch.long:
        raise ValueError("All14 original field states and explicit known masks required")
    if ((states[known] < 0) | (states[known] >= len(STATES))).any() or (states[known] == UNKNOWN).any():
        raise ValueError("Unknown truth cannot be relabelled as reviewed field supervision")
    if not material_side_labels_frozen and known[:, :4].any():
        raise ValueError("Unordered/unconfirmed plate sides cannot supervise fixed named material fields")
    zero = sum(value.sum() * 0 for value in output.values())
    components = {"states": zero, "continuous": zero, "directed_normals": zero, "binary": zero}
    if known.any():
        components["states"] = F.cross_entropy(output["state_logits"][known], states[known])
    valid = known & (states == STATES.index("value"))
    columns = [FIELDS.index(field) for field in CONTINUOUS_FIELDS]
    mask = valid[:, columns]
    values = targets["continuous_m"]
    if values.shape != (count, len(CONTINUOUS_FIELDS)):
        raise ValueError("All10 numeric field columns required, with explicit unknown masks")
    if mask.any():
        if not torch.isfinite(values[mask]).all() or (values[mask] < 0).any():
            raise ValueError("Reviewed numeric truth must be finite and nonnegative")
        if (values[:, :2][mask[:, :2]] <= 0).any():
            raise ValueError("Reviewed plate thickness must be positive")
        tolerance = values.new_tensor([0.001 if field.startswith("thickness") else 0.003 for field in CONTINUOUS_FIELDS])
        residual = (output["continuous_m"] - torch.where(mask, values, torch.zeros_like(values))) / tolerance
        components["continuous"] = F.smooth_l1_loss(residual[mask], torch.zeros_like(residual[mask]))
    normal_mask = valid[:, [FIELDS.index(field) for field in NORMAL_FIELDS]]
    normals = targets["normals"]
    if normals.shape != (count, 2, 3):
        raise ValueError("Both directed named material normal vectors required")
    if normal_mask.any():
        truth = normals[normal_mask]
        lengths = torch.linalg.vector_norm(truth, dim=-1)
        if not torch.isfinite(truth).all() or (torch.abs(lengths - 1) > 0.001).any():
            raise ValueError("Reviewed directed normal must meet the frozen unit-length condition")
        truth = truth / lengths[:, None]
        predicted = output["normals"][normal_mask]
        # Same named plate, same sign. Opposite normals incur maximal error.
        signed_dot = torch.sum(predicted * truth, dim=-1).clamp(-1, 1)
        components["directed_normals"] = ((1 - signed_dot) / (1 - math.cos(math.radians(3)))).mean()
        components["directed_normals"] += F.smooth_l1_loss(
            torch.linalg.vector_norm(output["raw_normals"][normal_mask], dim=-1), torch.ones_like(lengths))
    binary_mask = valid[:, [FIELDS.index(field) for field in BINARY_FIELDS]]
    binary = targets["binary"]
    if binary.shape != (count, 2) or binary.dtype != torch.long:
        raise ValueError("Both explicit boolean field targets required")
    if binary_mask.any():
        if ((binary[binary_mask] < 0) | (binary[binary_mask] > 1)).any():
            raise ValueError("Unknown or parent-any boolean is not reviewed child truth")
        components["binary"] = F.cross_entropy(output["binary_logits"][binary_mask], binary[binary_mask])
    total = sum(components.values())
    if not torch.isfinite(total):
        raise ValueError("Nonfinite attribute loss; do not drop original labels or examples")
    return total, dict(components=components, known_state_cells=int(known.sum()),
        known_numeric_cells=int(mask.sum()), known_directed_normal_cells=int(normal_mask.sum()),
        known_boolean_cells=int(binary_mask.sum()), unknown_truth_cells=int((~known).sum()),
        original_field_cell_denominator=count * len(FIELDS), side_exchange_used=False,
        absolute_normal_dot_used=False, unknown_truth_promoted_to_zero=False)
