"""Frozen-role M0 reference and complete-domain M1 streaming start gate.

Only role computation and intermediate storage change. Original blocks, two-row
halos, five-role sigmoid, active-slot ordering and threshold remain authoritative.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .carrier_boundary_runtime_329 import INFERENCE_BLOCK_ROWS, SEQUENCE_HALO_ROWS, PRESENCE_THRESHOLD
from .preallocated_carrier_inference_340 import infer_preallocated_carrier_heads


@dataclass(frozen=True)
class RoleGateReceipt:
    rows: int
    legal_slots: int
    selected_count: int
    selected_slot_ids_sha256: str
    role_block_forwards: int
    whole_carrier_sigmoid_differential_checked: bool = False


def ordered_slot_digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes(order="C")).hexdigest()


def _requirements(model, layout):
    if torch.is_grad_enabled() or model.training:
        raise ValueError("Frozen eval-mode no-grad role inference required")
    if layout.rows <= 0:
        raise ValueError("Complete nonempty carrier layout required")


def _emit(profile, stage, edge, layout, **values):
    if profile is not None:
        profile(stage=stage, edge=edge, carrier_ordinal=int(layout.carrier.ordinal),
            rows=int(layout.rows), legal_slots=int(layout.carrier.slot_count), **values)


def _role_block(model, tokens, provider, layout, first, last, profile):
    lo, hi = max(0, first - SEQUENCE_HALO_ROWS), min(layout.rows, last + SEQUENCE_HALO_ROWS)
    bounds = dict(block_first=first, block_last=last, halo_lo=lo, halo_hi=hi)
    _emit(profile, "ROLE_OBSERVATIONS", "BEGIN", layout, **bounds)
    features, groups = provider.observations(layout, lo, hi)
    _emit(profile, "ROLE_OBSERVATIONS", "END", layout, **bounds, known_objects=dict(features=features, groups=groups))
    _emit(profile, "ROLE_SEQUENCE_ENCODING", "BEGIN", layout, **bounds)
    sequence = model.encode_sequence(torch.from_numpy(features), tokens[torch.from_numpy(groups)], tokens.mean(0))
    _emit(profile, "ROLE_SEQUENCE_ENCODING", "END", layout, **bounds, known_objects=dict(sequence=sequence))
    core = sequence[first - lo:last - lo]
    _emit(profile, "ROLE_ONLY_FIVE_OUTPUT_HEAD", "BEGIN", layout, **bounds)
    embeddings = model.slot_embeddings(core)
    roles = model.roles(embeddings)
    if roles.shape != (last - first, 2, 5) or roles.device.type != "cpu" or roles.dtype != torch.float32:
        raise ValueError("Original CPU float32 two-subslot five-role head required")
    if not bool(torch.isfinite(roles).all()):
        raise ValueError("REGISTERED_NONFINITE_ROLE_DIAGNOSTIC: role gate cannot treat NaN or Inf as empty")
    _emit(profile, "ROLE_ONLY_FIVE_OUTPUT_HEAD", "END", layout, **bounds,
        known_objects=dict(sequence=sequence, embeddings=embeddings, roles=roles))
    return roles


def infer_roles_only_whole_reference(model, tokens, provider, layout, profile=None):
    """M0: role-only computation, original whole-carrier role storage and sigmoid."""
    _requirements(model, layout)
    roles = None
    calls = 0
    for first in range(0, layout.rows, INFERENCE_BLOCK_ROWS):
        last = min(layout.rows, first + INFERENCE_BLOCK_ROWS)
        block = _role_block(model, tokens, provider, layout, first, last, profile)
        if roles is None:
            roles = block.new_empty((layout.rows, 2, 5))
        roles[first:last].copy_(block)
        calls += 1
        del block
    return roles, calls


def whole_role_selected_ids(roles, layout):
    active_np = layout.active_slots
    chosen = roles[torch.from_numpy(active_np)].sigmoid()[:, 1] >= PRESENCE_THRESHOLD
    return np.asarray(layout.slot_ids[active_np][chosen.numpy()], dtype="<i8")


def scan_complete_child_start_gate(model, tokens, provider, layout, profile=None,
                                   compare_original_whole_sigmoid=False):
    """M1 retains scalar identities; optional actual-data differential uses M0 storage.

The registered diagnostic mode uses the same fresh role values as M0, compares
every original legal-slot boolean with its streamed counterpart, then releases
the reference. This mode is recorded explicitly and has M0's role-storage cost.
"""
    _requirements(model, layout)
    written = legal = selected_count = calls = 0
    selected_hash = hashlib.sha256()
    whole_reference = None
    streamed_decisions = np.empty(layout.carrier.slot_count, dtype=np.bool_) if compare_original_whole_sigmoid else None
    for first in range(0, layout.rows, INFERENCE_BLOCK_ROWS):
        last = min(layout.rows, first + INFERENCE_BLOCK_ROWS)
        if first != written:
            raise ValueError("Omitted or repeated original carrier rows")
        roles = _role_block(model, tokens, provider, layout, first, last, profile)
        if compare_original_whole_sigmoid:
            if whole_reference is None:
                whole_reference = roles.new_empty((layout.rows, 2, 5))
            whole_reference[first:last].copy_(roles)
        active_np = layout.active_slots[first:last]
        active = torch.from_numpy(active_np)
        _emit(profile, "ROLE_ACTIVE_SIGMOID_SELECTION", "BEGIN", layout, block_first=first, block_last=last)
        probabilities = roles[active].sigmoid()
        chosen = probabilities[:, 1] >= PRESENCE_THRESHOLD
        block_ids = layout.slot_ids[first:last][active_np]
        chosen_ids = np.asarray(block_ids[chosen.numpy()], dtype="<i8")
        selected_hash.update(chosen_ids.tobytes(order="C"))
        block_legal = int(active_np.sum())
        if compare_original_whole_sigmoid:
            streamed_decisions[legal:legal + block_legal] = chosen.numpy()
        selected_count += len(chosen_ids)
        legal += block_legal
        written = last
        calls += 1
        _emit(profile, "ROLE_ACTIVE_SIGMOID_SELECTION", "END", layout, block_first=first, block_last=last,
            selected_starts=selected_count, legal_slots_scanned=legal,
            known_objects=dict(roles=roles, probabilities=probabilities, reference=whole_reference, decisions=streamed_decisions))
        del chosen_ids, block_ids, chosen, probabilities, active, active_np, roles
    if written != layout.rows or legal != layout.carrier.slot_count:
        raise ValueError("Incomplete original carrier or legal-slot scan")
    if compare_original_whole_sigmoid:
        _emit(profile, "M0_M1_WHOLE_SIGMOID_DIFFERENTIAL", "BEGIN", layout)
        original_decisions = whole_reference[torch.from_numpy(layout.active_slots)].sigmoid()[:, 1] >= PRESENCE_THRESHOLD
        if not np.array_equal(original_decisions.numpy(), streamed_decisions):
            mismatches = np.flatnonzero(original_decisions.numpy() != streamed_decisions)
            raise ValueError("M0_M1_NUMERICAL_SELECTION_MISMATCH: " + str(mismatches[:20].tolist()))
        original_ids = layout.slot_ids[layout.active_slots][original_decisions.numpy()]
        if len(original_ids) != selected_count or ordered_slot_digest(original_ids) != selected_hash.hexdigest():
            raise ValueError("M0_M1_ORDERED_SLOT_IDENTITY_MISMATCH")
        _emit(profile, "M0_M1_WHOLE_SIGMOID_DIFFERENTIAL", "END", layout,
            selected_starts=selected_count, every_legal_slot_boolean_equal=True)
        del original_ids, original_decisions, whole_reference, streamed_decisions
    return RoleGateReceipt(written, legal, selected_count, selected_hash.hexdigest(), calls,
                           bool(compare_original_whole_sigmoid))


def require_recomputed_start_identity(gate, ids, selected, layout):
    if type(gate) is not RoleGateReceipt or gate.rows != layout.rows or gate.legal_slots != layout.carrier.slot_count:
        raise ValueError("Complete original role-gate receipt required")
    if len(selected) != gate.selected_count or ordered_slot_digest(ids[selected]) != gate.selected_slot_ids_sha256:
        raise ValueError("RECOMPUTED_ORIGINAL_START_IDENTITY_MISMATCH")


def materialize_complete_heads_after_gate(model, tokens, provider, layout, gate, profile=None):
    if gate.selected_count <= 0:
        raise ValueError("No full head storage is needed by a genuinely empty start gate")
    _emit(profile, "NONZERO_ORIGINAL_COMPLETE_HEADS", "BEGIN", layout, selected_starts=gate.selected_count)
    heads, calls = infer_preallocated_carrier_heads(model, tokens, provider, layout)
    for key, values in heads.items():
        for first in range(0, len(values), INFERENCE_BLOCK_ROWS):
            if not bool(torch.isfinite(values[first:first + INFERENCE_BLOCK_ROWS]).all()):
                raise ValueError("REGISTERED_NONFINITE_COMPLETE_HEAD_DIAGNOSTIC: " + key)
    _emit(profile, "NONZERO_ORIGINAL_COMPLETE_HEADS", "END", layout, selected_starts=gate.selected_count,
        known_objects=heads)
    return heads, calls


def empty_gate_decoded_result(gate, layout, first_query=0):
    require_recomputed_start_identity(gate, np.empty(0, dtype=np.int64), [], layout)
    if gate.selected_count != 0:
        raise ValueError("Selected starts must retain the full original false-output ledger")
    return dict(parents=[], invalid_predictions=[], total_prediction_children=0,
                selected_start_slots=0, next_query_index=int(first_query))
