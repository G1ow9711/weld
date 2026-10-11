"""Bounded-memory gradients of the same complete nullable set score.

Every affected parent and relation contributes. The active hinge is evaluated
without retaining an autograd graph, then its exact additive score terms are
backpropagated in chunks before one optimizer step. This neither subsamples
the selected set nor replaces full reference matching with local labels.
"""
import numpy as np
import torch


def coefficients(data, predicted, teacher):
    count = len(data['counts'])
    coefficient = np.zeros(count, np.float32)
    masks = []
    for selected, sign in ((predicted, 1.), (teacher, -1.)):
        ids = np.asarray(selected, dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= count):
            raise ValueError('A complete selection must contain unique valid parent indices')
        coefficient[ids] += sign
        mask = np.zeros(count, bool)
        mask[ids] = True
        masks.append(mask)
    a, b = data['pairs'].T
    predicted_mask, teacher_mask = masks
    weights = ((teacher_mask[a] & teacher_mask[b]).astype(np.float32)
               - (predicted_mask[a] & predicted_mask[b]).astype(np.float32))
    return coefficient, weights


def score_terms(model, data, coefficient, weights, pack_batch,
                parent_batch_size=256, relation_batch_size=8192):
    device = next(model.parameters()).device
    nodes = np.flatnonzero(coefficient)
    for start in range(0, len(nodes), parent_batch_size):
        ids = nodes[start:start + parent_batch_size]
        tokens, valid = pack_batch(data, ids, device)
        yield (model.parent_utility(tokens, valid)
               * torch.tensor(coefficient[ids], device=device)).sum()
    edges = np.flatnonzero(weights)
    for start in range(0, len(edges), relation_batch_size):
        ids = edges[start:start + relation_batch_size]
        penalty = model.pair_penalty(torch.tensor(data['pair_features'][ids], device=device),
                      torch.tensor(data['base_penalties'][ids], device=device, dtype=torch.float32),
                      torch.tensor(data['pair_scale'][ids], device=device, dtype=torch.float32))
        yield (penalty * torch.tensor(weights[ids], device=device)).sum()


def backward_complete_hinge(model, data, predicted, teacher, advantage,
                            normalizer, loss_mass, pack_batch,
                            parent_batch_size=256, relation_batch_size=8192):
    """Accumulate the full active hinge gradient; caller steps once afterwards."""
    if min(parent_batch_size, relation_batch_size) < 1 or normalizer <= 0 or loss_mass <= 0:
        raise ValueError('Positive gradient resources and registered objective weights required')
    if not np.isfinite([advantage, normalizer, loss_mass]).all():
        raise ValueError('Full set objective weights must be finite')
    if not model.training:
        raise ValueError('Use a deterministic training mode for both hinge passes')
    if any(isinstance(module, torch.nn.Dropout) and module.p != 0 for module in model.modules()):
        raise ValueError('Two-pass complete gradients require dropout-free set scoring')
    coefficient, weights = coefficients(data, predicted, teacher)
    device = next(model.parameters()).device
    arguments = (model, data, coefficient, weights, pack_batch,
                 parent_batch_size, relation_batch_size)
    with torch.no_grad():
        difference = torch.zeros((), device=device)
        for term in score_terms(*arguments):
            difference = difference + term
        margin = difference * normalizer + advantage
        loss = torch.relu(margin) * loss_mass
        if not torch.isfinite(loss):
            raise ValueError('Complete set hinge became nonfinite')
        active = bool(margin > 0)
    gradient_terms = 0
    if active:
        for term in score_terms(*arguments):
            (term * (normalizer * loss_mass)).backward()
            gradient_terms += 1
        if gradient_terms == 0:
            raise ValueError('Positive reference advantage has no differentiable set difference')
    return dict(loss=float(loss), active=active, gradient_terms=gradient_terms,
                affected_parents=int(np.count_nonzero(coefficient)),
                affected_relations=int(np.count_nonzero(weights)),
                all_complete_set_terms_used=True)
