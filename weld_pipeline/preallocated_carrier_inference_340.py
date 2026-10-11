"""Preserve the frozen 329 computation while avoiding duplicate head storage.

This module changes CPU inference storage only. It retains every exact carrier,
alias, cell, event, slot and original 8192-row block with its two-row halo.
Canonical carrier methods and the frozen model/decoder remain authoritative.
"""
from dataclasses import dataclass

import torch

from .analytic_carrier_slots_327 import Carrier, exact_geometry_key
from .carrier_boundary_runtime_329 import INFERENCE_BLOCK_ROWS, SEQUENCE_HALO_ROWS


@dataclass(frozen=True, slots=True)
class InferenceCarrier:
    ordinal: int
    weld_type: str
    kind: str
    length_mm: float
    origin_mm: object
    direction: object
    arc: object
    geometry_identity: str
    aliases: tuple
    structural_events_mm: tuple = ()

    from_record = classmethod(Carrier.from_record.__func__)
    cell_count = Carrier.cell_count
    cell_width_mm = Carrier.cell_width_mm
    slot_count = Carrier.slot_count
    slot_spec = Carrier.slot_spec
    addresses_at = Carrier.addresses_at
    point_at = Carrier.point_at
    project = Carrier.project
    summary = Carrier.summary


def build_compact_inference_carriers(records):
    """Same first-record authority and exact alias grouping, in insertion order."""
    by_identity = {}
    carriers = []
    for record in records:
        key = exact_geometry_key(record)
        carrier = by_identity.get(key)
        if carrier is None:
            carrier = InferenceCarrier.from_record(len(carriers), record)
            object.__setattr__(carrier, "geometry_identity", key)
            object.__setattr__(carrier, "aliases", [])
            carriers.append(carrier)
            by_identity[key] = carrier
        carrier.aliases.append(record["id"])
    for carrier in carriers:
        object.__setattr__(carrier, "aliases", tuple(carrier.aliases))
    return carriers


def infer_preallocated_carrier_heads(model, tokens, provider, layout):
    """Identical block computations; copy into one complete buffer per head."""
    if torch.is_grad_enabled():
        raise ValueError("Preallocated buffers are restricted to no-grad inference")
    if layout.rows <= 0:
        raise ValueError("A nonempty complete carrier layout is required")
    buffers = None
    calls = 0
    written = 0
    for first in range(0, layout.rows, INFERENCE_BLOCK_ROWS):
        last = min(layout.rows, first + INFERENCE_BLOCK_ROWS)
        lo = max(0, first - SEQUENCE_HALO_ROWS)
        hi = min(layout.rows, last + SEQUENCE_HALO_ROWS)
        features, groups = provider.observations(layout, lo, hi)
        sequence = model.encode_sequence(torch.from_numpy(features),
            tokens[torch.from_numpy(groups)], tokens.mean(0))
        heads = model.heads(sequence[first - lo:last - lo])
        if buffers is None:
            buffers = {key: value.new_empty((layout.rows, *value.shape[1:]))
                       for key, value in heads.items()}
        if heads.keys() != buffers.keys() or first != written:
            raise ValueError("Head keys or complete row coverage changed")
        for key, value in heads.items():
            expected = buffers[key]
            if (value.shape != (last - first, *expected.shape[1:])
                    or value.dtype != expected.dtype or value.device != expected.device):
                raise ValueError("Head shape, dtype or device changed")
            expected[first:last].copy_(value)
        calls += 1
        written = last
        del heads, sequence, features, groups, value, expected
    if written != layout.rows:
        raise ValueError("Complete carrier rows were not written")
    return buffers, calls
