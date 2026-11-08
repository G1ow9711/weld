"""Loss-only endpoint-mass consistency; no inference count head or constraint."""
import types
import torch
from torch.nn import functional as F
from tools import curveonly_01a1ac7c_instance_order_train_v198 as previous

SETTINGS = dict(previous.SETTINGS, endpoint_mass_loss_weight=.5,
               endpoint_mass_smooth_beta=.25,
               endpoint_mass_target='SUM_OF_EXISTING_ONEHOT_ENDPOINT_LABELS')
configure = previous.configure
assemble = previous.assemble


class MassTerms(tuple):
    def __new__(cls, values, penalty):
        result = super().__new__(cls, values)
        result.mass_penalty = penalty
        return result


def event_mass_penalty(point, batch):
    width = point.shape[-1]
    mask = torch.arange(width, device=point.device)[None, :] < batch['size'][:, None]
    mass = (torch.sigmoid(point[:, 1:3])*mask[:, None, :]).sum(2)
    target = batch['targets'][:, 1:3].sum(2)
    weight = batch['class_weight'][:, 1:3].sum(2).mean(1)
    error = F.smooth_l1_loss(torch.log1p(mass), torch.log1p(target), reduction='none',
                           beta=SETTINGS['endpoint_mass_smooth_beta']).mean(1)
    return (error*weight).sum()


def loss_terms(point, full, batch):
    values = previous.previous.loss_terms(point, full, batch)
    return MassTerms(values, event_mass_penalty(point, batch))


def objective(terms, masses):
    return (previous.previous.objective(terms, masses)+SETTINGS['endpoint_mass_loss_weight']*
            terms.mass_penalty/masses['classification'])


namespace = dict(previous.namespace, SETTINGS=SETTINGS, loss_terms=loss_terms,
                 objective=objective, assemble=assemble)
assembly_control = types.FunctionType(previous.assembly_control.__code__, namespace)
fit = types.FunctionType(previous.fit.__code__, namespace)


def mathematical_controls():
    point = torch.full((2, 6, 16), -30., dtype=torch.float64)
    target = torch.zeros_like(point)
    target[0, 1, 2] = target[0, 2, 5] = 1
    target[1, 1, :7] = target[1, 2, :7] = 1
    for b, role, ids in ((0, 1, [2]), (0, 2, [5]),
                         (1, 1, list(range(7))), (1, 2, list(range(7)))):
        point[b, role, ids] = 30.
    weight = torch.zeros((2, 3, 16), dtype=torch.float64)
    weight[0, :, :8] = 1/8
    weight[1, :, :12] = 1/12
    batch = dict(size=torch.tensor([8, 12]), targets=target, class_weight=weight)
    perfect = event_mass_penalty(point, batch)
    assert float(perfect.detach()) < 1e-9
    extra = point.clone()
    extra[0, 1:3, 7] = torch.logit(torch.tensor(.9, dtype=torch.float64))
    extra.requires_grad_()
    penalty = event_mass_penalty(extra, batch)
    penalty.backward()
    assert float(penalty.detach()) > float(perfect.detach()) and bool((extra.grad[0, 1:3, 7] > 0).all())
    missing = point.clone()
    missing[0, 1, 2] = missing[0, 2, 5] = torch.logit(torch.tensor(.1, dtype=torch.float64))
    missing.requires_grad_()
    value = event_mass_penalty(missing, batch)
    value.backward()
    assert float(value.detach()) > float(perfect.detach()) and float(missing.grad[0, 1, 2]) < 0
    padded = F.pad(extra.detach(), (0, 11), value=0.)
    padded_batch = dict(batch, targets=F.pad(target, (0, 11)),
                        class_weight=F.pad(weight, (0, 11)))
    assert torch.allclose(event_mass_penalty(padded, padded_batch), penalty.detach(), atol=1e-12, rtol=0)
    return dict(passed=True, exact_event_mass_has_zero_penalty=True,
                extra_event_gradient_suppresses=True,
                missing_event_gradient_increases=True, padded_cells_do_not_change_loss=True,
                target7_is_continuous_not_a_count_class=True,
                no_dataset_or_model_weights_read=True)
