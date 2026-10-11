"""Zero-initialized trainable residuals for complete-parent set scores."""
import numpy as np
import torch
from torch import nn


class StructuredParentResidual(nn.Module):
    def __init__(self, parent_mean, parent_std, pair_mean, pair_std):
        super().__init__()
        for name, value in [('parent_mean', parent_mean), ('parent_std', parent_std),
                            ('pair_mean', pair_mean), ('pair_std', pair_std)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.parent = nn.Sequential(nn.Linear(1458, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.pair = nn.Sequential(nn.Linear(29, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        for head in [self.parent, self.pair]:
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def parent_residual(self, features):
        return self.parent(torch.asinh((features-self.parent_mean)/self.parent_std)).squeeze(-1)

    def pair_residual(self, features):
        return torch.tanh(self.pair(torch.asinh((features-self.pair_mean)/self.pair_std)).squeeze(-1))


def predict_scores(model, x, pair_x, counts, pairs, base_unary, base_penalty, batch_size=8192):
    device = model.parent_mean.device
    with torch.inference_mode():
        def residual(data, function):
            pieces = [function(torch.as_tensor(np.asarray(data[i:i+batch_size]), device=device)).cpu().numpy()
                      for i in range(0, len(data), batch_size)]
            return np.concatenate(pieces).astype(np.float64) if pieces else np.empty(0, np.float64)
        u = residual(x, model.parent_residual)
        p = residual(pair_x, model.pair_residual)
    scale = 2*np.minimum(counts[pairs[:, 0]], counts[pairs[:, 1]])
    return np.asarray(base_unary, np.float64) + u*counts, np.clip(np.asarray(base_penalty, np.float64)+p*scale, 0., scale)


def score_difference(model, data, predicted, teacher):
    """Differentiable score(predicted)-score(teacher), with detached choices."""
    n = len(data['counts']); coefficients = np.zeros(n)
    coefficients[np.asarray(predicted, int)] += 1
    coefficients[np.asarray(teacher, int)] -= 1
    nodes = np.flatnonzero(coefficients)
    pm = np.zeros(n, bool); tm = np.zeros(n, bool)
    pm[np.asarray(predicted, int)] = True; tm[np.asarray(teacher, int)] = True
    pairs = data['pairs']
    pc = pm[pairs[:, 0]] & pm[pairs[:, 1]]
    tc = tm[pairs[:, 0]] & tm[pairs[:, 1]]
    edge_coefficients = tc.astype(float) - pc.astype(float)
    edges = np.flatnonzero(edge_coefficients)
    device = model.parent_mean.device
    result = torch.zeros((), device=device, dtype=torch.float64)
    if len(nodes):
        residual = model.parent_residual(torch.as_tensor(data['x'][nodes], device=device)).double()
        unary = torch.as_tensor(data['base_unary'][nodes], device=device) + residual*torch.as_tensor(data['counts'][nodes], device=device)
        result = result+(unary*torch.as_tensor(coefficients[nodes], device=device)).sum()
    if len(edges):
        residual = model.pair_residual(torch.as_tensor(data['px'][edges], device=device)).double()
        scale = torch.as_tensor(data['pair_scale'][edges], device=device)
        penalty = torch.clamp(torch.as_tensor(data['base_penalty'][edges], device=device)+residual*scale, min=0.)
        penalty = torch.minimum(penalty, scale)
        result = result+(penalty*torch.as_tensor(edge_coefficients[edges], device=device)).sum()
    return result
