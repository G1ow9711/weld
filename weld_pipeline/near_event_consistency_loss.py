"""Event-center consistency and hard-background controls, with explicit masks."""
import torch
import torch.nn.functional as F

LAGS = (1, 4, 8)
RADIUS_MM = 16.
STEP_MM = 2.
PHYSICAL_BETA_MM = 3.2


def center_consistency(logits, targets, known):
    offsets = logits[:, 4:7]; labels = targets[:, 4:7]; valid = known[:, 4:7]
    terms = []
    for lag in LAGS:
        if offsets.shape[-1] <= lag:
            continue
        expected = -lag*STEP_MM/RADIUS_MM
        target_difference = labels[:, :, lag:]-labels[:, :, :-lag]
        mask = ((valid[:, :, lag:] > .5) & (valid[:, :, :-lag] > .5)
                & ((target_difference-expected).abs() < 2e-5)).float()
        center_difference_mm = (offsets[:, :, lag:]-offsets[:, :, :-lag])*RADIUS_MM+lag*STEP_MM
        loss = F.smooth_l1_loss(center_difference_mm, torch.zeros_like(center_difference_mm),
                              beta=PHYSICAL_BETA_MM, reduction='none')
        terms.append(((loss*mask).sum((0, 2))/mask.sum((0, 2)).clamp_min(1.)).mean())
    return torch.stack(terms).mean() if terms else offsets.sum()*0.


def hard_background_means(bce, positive, negative):
    # All known negatives retain half their original contribution.
    original = (bce*negative).sum((0, 2))/negative.sum((0, 2)).clamp_min(1.)
    hard = []
    for c in range(3):
        values = bce[:, c][negative[:, c] > .5]
        if not len(values):
            hard.append(bce[:, c].sum()*0.)
            continue
        count = min(len(values), max(16, int(positive[:, c].sum().detach())))
        hard.append(torch.topk(values, count, sorted=False).values.mean())
    return .5*(original+torch.stack(hard))


def loss_fn(logits, targets, known, consistency, hard_negatives):
    bce = F.binary_cross_entropy_with_logits(logits[:, :4], targets[:, :4], reduction='none')
    fg = (bce[:, :1]*known[:, :1]).sum()/known[:, :1].sum().clamp_min(1.)
    positive = targets[:, 1:4]*known[:, 1:4]
    negative = (1.-targets[:, 1:4])*known[:, 1:4]
    pos = (bce[:, 1:4]*positive).sum((0, 2))/positive.sum((0, 2)).clamp_min(1.)
    if hard_negatives:
        neg = hard_background_means(bce[:, 1:4], positive, negative)
    else:
        neg = (bce[:, 1:4]*negative).sum((0, 2))/negative.sum((0, 2)).clamp_min(1.)
    offset = F.smooth_l1_loss(logits[:, 4:7], targets[:, 4:7], beta=.2, reduction='none')
    offset = (offset*known[:, 4:7]).sum((0, 2))/known[:, 4:7].sum((0, 2)).clamp_min(1.)
    total = fg+.5*(pos+neg).mean()+.25*offset.mean()
    if consistency:
        total = total+center_consistency(logits, targets, known)
    return total
