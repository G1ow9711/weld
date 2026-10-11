"""Pure finite-domain pointer geometry shared by training and free inference."""
import numpy as np
import torch

VERSION = "SOURCE_ONLY_COMPLETE_FINITE_POINTER_DOMAIN_329_v2"


def legal_terminal_mask(carrier, layout, start_slot):
    start_row, _ = layout.location(start_slot)
    # Keep every terminal that can follow the start. Process minimum is
    # enforced jointly AFTER the learned common pose and slot positions are
    # known; a corrected primitive may have a different physical length scale.
    forward = layout.upper_mm[:, None] > layout.lower_mm[start_row]
    legal = forward if carrier.weld_type == "vertical" else (
        forward | (layout.upper_mm[start_row] > layout.lower_mm[:, None]))
    return legal & layout.active_slots & (layout.slot_ids != start_slot)


def learned_parameters(heads, layout):
    lower = torch.from_numpy(layout.lower_mm)[:, None]
    width = torch.from_numpy(layout.upper_mm - layout.lower_mm)[:, None]
    return lower + heads["within_cell_fraction"].to(torch.float64) * width


def relation_features(end_q, next_start_q, shared_slot, same_traversal):
    difference = (next_start_q - end_q).to(torch.float32)
    return torch.stack((difference / 1000., (difference.abs() / 3.).clamp(max=100.),
        difference.new_tensor(float(shared_slot)), difference.new_tensor(float(same_traversal))))
