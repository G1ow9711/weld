"""A trainable admission correction around the frozen legacy query prior.

Zero initialization preserves the actual legacy score exactly. Correction
supervision is assigned annotated-child responsibility, not a physical absence
label for other queries. The correction cannot change geometry through its
gradient; the full geometry/process/frame losses keep their separate path.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .server_legacy_anchored_full_weld_model import LegacyAnchoredFullWeldModel

ARCHITECTURE = 'LEGACY_ANCHORED_REFERENCE_RECALL_V3'


class ReferenceRecallWeldModel(LegacyAnchoredFullWeldModel):
    def __init__(self):
        super().__init__()
        self.reference_delta = nn.Linear(128, 1)
        nn.init.zeros_(self.reference_delta.weight)
        nn.init.zeros_(self.reference_delta.bias)
        self.legacy.requires_grad_(False)
        self.reference.requires_grad_(False)

    def inherit_anchored(self, state):
        missing = self.load_state_dict(state, strict=False)
        if set(missing.missing_keys) != {'reference_delta.weight', 'reference_delta.bias'} or missing.unexpected_keys:
            raise ValueError('Recall initialization must be the actual complete anchored model state')

    def predict(self, hidden, query_ratio):
        result = super().predict(hidden, query_ratio)
        correction = self.reference_delta(hidden[:, :128].detach()).squeeze(1)
        # Slicing the old columns from a concatenated trainable embedding can
        # allocate zero gradients on its other columns. Detach the frozen prior
        # explicitly so admission-only loss has no geometry gradient path.
        result['frozen_reference_score'] = result['reference_score'].detach()
        result['reference_delta_logit'] = correction
        result['reference_score'] = result['frozen_reference_score'] + correction
        return result


def reference_responsibility_loss(prediction):
    """Only known annotated-child assigned queries supply a positive target.

    The squared correction is a prior regularizer. No unassigned query is
    relabeled background. These logits are not calibrated physical presence.
    """
    score = prediction['reference_score']
    if not len(score):
        raise ValueError('Complete reference responsibility requires actual annotated children')
    return dict(reference_assigned_positive=F.softplus(1. - score).mean(),
                reference_prior_correction=prediction['reference_delta_logit'].square().mean())
