"""Trainable nullable set utility on actual predicted ordered child tokens."""
import numpy as np
import torch
from torch import nn

from .server_full_prediction_runtime import proposal_token


class OrderedParentSetModel(nn.Module):
    def __init__(self, warm_parent_encoder=None):
        super().__init__()
        self.child = nn.Linear(151, 128)
        with torch.no_grad():
            self.child.weight.zero_()
            self.child.weight[:, :128] = torch.eye(128)
            self.child.bias.zero_()
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0, batch_first=True)
        self.sequence = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        if warm_parent_encoder is not None:
            self.sequence.load_state_dict(warm_parent_encoder)
        self.unary = nn.Linear(128, 1)
        self.pair = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))
        for head in (self.unary, self.pair[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def parent_utility(self, tokens, valid):
        hidden = self.child(tokens)
        positions = torch.arange(hidden.shape[1], device=hidden.device, dtype=hidden.dtype)[:, None]
        scale = torch.exp(torch.arange(0, 128, 2, device=hidden.device, dtype=hidden.dtype)
                          * (-9.210340371976184 / 128))
        ordered = torch.zeros_like(hidden[0])
        ordered[:, 0::2] = torch.sin(positions * scale)
        ordered[:, 1::2] = torch.cos(positions * scale)
        encoded = self.sequence(hidden + ordered[None], src_key_padding_mask=~valid)
        count = valid.sum(1)
        pooled = (encoded * valid[:, :, None]).sum(1) / count[:, None]
        return torch.tanh(self.unary(pooled).squeeze(1)) * count

    def pair_penalty(self, features, base, scale):
        residual = torch.tanh(self.pair(torch.asinh(features)).squeeze(1)) * scale
        return torch.minimum(torch.clamp(base + residual, min=0), scale)


def parent_batch(pool, proposals, indices, device):
    parents = [pool['parents'][int(i)] for i in indices]
    maximum = max((p['child_count'] for p in parents), default=0)
    tokens = np.zeros((len(parents), maximum, 151), np.float32)
    valid = np.zeros((len(parents), maximum), bool)
    for row, parent in enumerate(parents):
        for position, (child, original) in enumerate(zip(parent['segments'], parent['input_children'])):
            segment = dict(child, points=child['endpoints_mm'])
            child_token = proposal_token(segment, proposals['child_tokens'][original, :128],
                                      proposals['query_xyz'][original], proposals['query_bases'][original], child['reference_score'])
            previous_gap = parent['gaps'][position - 1] if position else 'START'
            tokens[row, position] = np.r_[child_token, float(previous_gap == 'SHARED'),
                                          float(previous_gap == 'POSITIVE_GAP'), float(parent['is_closed'])]
            valid[row, position] = True
    return torch.tensor(tokens, device=device), torch.tensor(valid, device=device)


@torch.inference_mode()
def predict_set_scores(model, pool, proposals, pairs, pair_features, base_penalty, batch_size=256):
    device = next(model.parameters()).device
    unary = []
    for start in range(0, len(pool['parents']), batch_size):
        indices = np.arange(start, min(start + batch_size, len(pool['parents'])))
        tokens, valid = parent_batch(pool, proposals, indices, device)
        unary.append(model.parent_utility(tokens, valid).cpu().numpy())
    counts = np.array([p['child_count'] for p in pool['parents']])
    scale = 2 * np.minimum(counts[pairs[:, 0]], counts[pairs[:, 1]])
    penalties = []
    for start in range(0, len(pairs), batch_size):
        penalties.append(model.pair_penalty(torch.tensor(pair_features[start:start + batch_size], device=device),
                           torch.tensor(base_penalty[start:start + batch_size], device=device, dtype=torch.float32),
                           torch.tensor(scale[start:start + batch_size], device=device, dtype=torch.float32)).cpu().numpy())
    return (np.concatenate(unary).astype(float) if unary else np.empty(0),
            np.concatenate(penalties).astype(float) if penalties else np.empty(0))
