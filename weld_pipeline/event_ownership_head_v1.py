"""Conditional O/B/I ownership engineering; no project data or training adapter.

The input feature recipe remains caller supplied. This module does not load or
change frozen 175 weights. Decoded blocks contain query identities only; centre
fusion and event confidence are deliberately separate, unfinished interfaces.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


VERSION = 'conditional_event_ownership_obi_crf_prepare_v1'
STATUS = 'STANDBY_NO_TRAINING'
O, B, I, UNKNOWN = 0, 1, 2, -1
ROLE_NAMES = ('start', 'end', 'cut')
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _lengths(lengths, batch_size, time_size, device):
    values = torch.as_tensor(lengths, device=device)
    if batch_size == 0 and values.shape == (0,):
        return values.to(dtype=torch.long)
    if values.shape != (batch_size,) or values.dtype not in _INTEGER_DTYPES:
        raise ValueError('One integer valid length per sequence is required')
    if bool(torch.any(values < 0)) or bool(torch.any(values > time_size)):
        raise ValueError('Valid lengths must be between zero and padded time size')
    return values.to(dtype=torch.long)


def _safe_logsumexp(values, dim):
    """An unreachable state has -inf value and zero, finite derivatives.

    Direct logsumexp([-inf, ...]) gives NaN backward derivatives. Replace only
    wholly unreachable reductions before computing, then restore their -inf.
    """
    reachable = torch.isfinite(values).any(dim=dim, keepdim=True)
    safe = torch.where(reachable, values, torch.zeros_like(values))
    result = torch.logsumexp(safe, dim=dim)
    return torch.where(reachable.squeeze(dim), result,
                       torch.full_like(result, float('-inf')))


class PartialLabelOBICRF(nn.Module):
    """Independent role chains with a fixed legal O/B/I transition graph.

    Scores are [batch,time,role,tag]. Unknown targets constrain no states.
    log_partition can return -inf for an impossible constraint set; loss rejects
    that condition explicitly. decode takes no GT or known-label arguments.
    """
    def __init__(self, num_roles=3):
        super().__init__()
        if isinstance(num_roles, bool) or not isinstance(num_roles, Integral) or num_roles < 1:
            raise ValueError('A positive integer role count is required')
        self.num_roles = int(num_roles)
        self.start_transitions = nn.Parameter(torch.zeros(self.num_roles, 3))
        self.transitions = nn.Parameter(torch.zeros(self.num_roles, 3, 3))
        self.end_transitions = nn.Parameter(torch.zeros(self.num_roles, 3))
        self.register_buffer('legal_start', torch.tensor([True, True, False]))
        self.register_buffer('legal_transitions', torch.tensor([
            [True, True, False], [True, True, True], [True, True, True]]))

    def _prepare(self, emissions, lengths):
        if (not isinstance(emissions, torch.Tensor) or not emissions.is_floating_point()
                or emissions.ndim != 4 or emissions.shape[2:] != (self.num_roles, 3)):
            raise ValueError('Floating [batch,time,role,3] emissions are required')
        if emissions.device != self.start_transitions.device:
            raise ValueError('CRF parameters and emissions must share a device')
        batch_size, time_size = emissions.shape[:2]
        lengths = _lengths(lengths, batch_size, time_size, emissions.device)
        valid = torch.arange(time_size, device=emissions.device)[None, :] < lengths[:, None]
        if not bool(torch.isfinite(emissions[valid]).all()):
            raise ValueError('Valid-time emissions must be finite')
        for parameter in (self.start_transitions, self.transitions, self.end_transitions):
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError('CRF transition parameters must be finite')
        # Pad values are never evidence, even if a caller fills them with NaN.
        work = torch.where(valid[:, :, None, None], emissions, torch.zeros_like(emissions))
        if work.dtype in (torch.float16, torch.bfloat16):
            work = work.float()
        return work, lengths, valid

    def _scores(self, dtype):
        start = self.start_transitions.to(dtype=dtype).masked_fill(~self.legal_start, float('-inf'))
        transition = self.transitions.to(dtype=dtype).masked_fill(~self.legal_transitions, float('-inf'))
        return start, transition, self.end_transitions.to(dtype=dtype)

    def log_partition(self, emissions, lengths, allowed_tags=None):
        work, lengths, valid = self._prepare(emissions, lengths)
        if allowed_tags is not None:
            if (not isinstance(allowed_tags, torch.Tensor) or allowed_tags.shape != work.shape
                    or allowed_tags.dtype != torch.bool or allowed_tags.device != work.device):
                raise ValueError('Allowed states must be a same-device boolean emission-shaped mask')
            # Padded constraints never contribute to a legal path.
            allowed = torch.where(valid[:, :, None, None], allowed_tags,
                                  torch.ones_like(allowed_tags))
        else:
            allowed = torch.ones_like(work, dtype=torch.bool)
        start, transition, end = self._scores(work.dtype)
        if work.shape[1] == 0:
            anchor = (work.sum() + start[:, :2].sum() + self.transitions.sum() + end.sum()) * 0.
            return work.new_zeros((work.shape[0], self.num_roles)) + anchor
        alpha = (work[:, 0] + start).masked_fill(~allowed[:, 0], float('-inf'))
        for time in range(1, work.shape[1]):
            scores = alpha[:, :, :, None] + transition[None, :, :, :]
            following = (_safe_logsumexp(scores, dim=2) + work[:, time]).masked_fill(
                ~allowed[:, time], float('-inf'))
            alpha = torch.where((lengths > time)[:, None, None], following, alpha)
        result = _safe_logsumexp(alpha + end, dim=2)
        return torch.where((lengths > 0)[:, None], result, torch.zeros_like(result))

    def loss(self, emissions, lengths, tags, known_mask=None, reduction='none'):
        work, lengths, valid = self._prepare(emissions, lengths)
        if (not isinstance(tags, torch.Tensor) or tags.shape != work.shape[:3]
                or tags.dtype not in _INTEGER_DTYPES or tags.device != work.device):
            raise ValueError('Same-device integer [batch,time,role] target tags are required')
        if known_mask is None:
            known = tags != UNKNOWN
        else:
            if (not isinstance(known_mask, torch.Tensor) or known_mask.shape != tags.shape
                    or known_mask.dtype != torch.bool or known_mask.device != work.device):
                raise ValueError('Known mask must be same-device boolean and target-shaped')
            known = known_mask
        known = known & valid[:, :, None]
        if bool(torch.any((tags[known] < O) | (tags[known] > I))):
            raise ValueError('Known labels must be O, B or I; unknown is not a negative label')
        safe_tags = torch.where(known, tags, torch.zeros_like(tags)).long()
        allowed = torch.where(known[:, :, :, None], F.one_hot(safe_tags, num_classes=3).bool(),
                              torch.ones_like(work, dtype=torch.bool))
        full = self.log_partition(work, lengths)
        constrained = self.log_partition(work, lengths, allowed)
        if not bool(torch.isfinite(constrained).all()):
            raise ValueError('Known O/B/I labels have no legal completion')
        result = full - constrained
        if reduction == 'none':
            return result
        if reduction == 'sum':
            return result.sum()
        if reduction == 'mean':
            # Do not dilute supervised losses with empty or wholly unknown roles.
            denominator = known.any(dim=1).sum().clamp(min=1).to(dtype=result.dtype)
            return result.sum() / denominator
        raise ValueError('Reduction must be none, sum or mean')

    @torch.no_grad()
    def decode(self, emissions, lengths):
        """Legal Viterbi tags; UNKNOWN appears only outside valid lengths."""
        work, lengths, _ = self._prepare(emissions, lengths)
        output = torch.full(work.shape[:3], UNKNOWN, dtype=torch.long, device=work.device)
        if work.shape[1] == 0 or work.shape[0] == 0:
            return output
        start, transition, end = self._scores(work.dtype)
        alpha = work[:, 0] + start
        history = []
        for time in range(1, work.shape[1]):
            following, previous = (alpha[:, :, :, None] + transition[None, :, :, :]).max(dim=2)
            following = following + work[:, time]
            alpha = torch.where((lengths > time)[:, None, None], following, alpha)
            history.append(previous)
        last = (alpha + end).argmax(dim=2)
        for batch, length in enumerate(lengths.cpu().tolist()):
            for role in range(self.num_roles):
                if length == 0:
                    continue
                state = int(last[batch, role])
                output[batch, length - 1, role] = state
                for time in range(length - 1, 0, -1):
                    state = int(history[time - 1][batch, role, state])
                    output[batch, time - 1, role] = state
        return output


class EventOwnershipHead(nn.Module):
    """One small shared sequence head; actual feature composition is undefined."""
    def __init__(self, input_dim):
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, Integral) or input_dim < 1:
            raise ValueError('A caller-supplied positive integer input dimension is required')
        self.input_dim = int(input_dim)
        self.gru = nn.GRU(self.input_dim, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.heads = nn.ModuleList([nn.Linear(128, 3) for _ in ROLE_NAMES])
        self.crf = PartialLabelOBICRF(len(ROLE_NAMES))

    def forward(self, features, lengths):
        if (not isinstance(features, torch.Tensor) or not features.is_floating_point()
                or features.ndim != 3 or features.shape[2] != self.input_dim):
            raise ValueError('Floating [batch,time,input_dim] features are required')
        if features.device != self.gru.weight_ih_l0.device:
            raise ValueError('Ownership head and features must share a device')
        lengths = _lengths(lengths, features.shape[0], features.shape[1], features.device)
        valid = torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
        if not bool(torch.isfinite(features[valid]).all()):
            raise ValueError('Valid-time input features must be finite')
        safe = torch.where(valid[:, :, None], features, torch.zeros_like(features))
        nonempty = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if features.shape[1] == 0 or not len(nonempty):
            anchor = safe.sum() * 0.
            for parameter in self.parameters():
                anchor = anchor + parameter.sum() * 0.
            return features.new_zeros((*features.shape[:2], len(ROLE_NAMES), 3)) + anchor
        packed = pack_padded_sequence(safe.index_select(0, nonempty),
            lengths.index_select(0, nonempty).cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=features.shape[1])
        hidden = features.new_zeros((*features.shape[:2], 128)).index_copy(0, nonempty, encoded)
        logits = torch.stack([head(hidden) for head in self.heads], dim=2)
        return logits.masked_fill(~valid[:, :, None, None], 0.)


@dataclass(frozen=True)
class EventBlock:
    role: int
    block_id: int
    query_indices: tuple


def _integer_matrix(values, name):
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] < 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(name + ' must be a [time,role] integer matrix')
    return array


def teacher_owners_to_obi(owners):
    """Complete synthetic ownership protocol, not a project target adapter.

    Owners use nonnegative identities and background -1. Missing/unknown owner
    labels are not guessed. A repeated noncontiguous identity is unrepresentable
    by a single B I* block and is rejected rather than silently duplicated.
    """
    owners = _integer_matrix(owners, 'Complete owners')
    if np.any(owners < -1):
        raise ValueError('Complete ownership is required; unknown owner adapter is NOT_IMPLEMENTED')
    tags = np.empty(owners.shape, dtype=np.int64)
    for role in range(owners.shape[1]):
        active = -1
        seen = set()
        for index, identity in enumerate(owners[:, role]):
            identity = int(identity)
            if identity == -1:
                tags[index, role] = O
                active = -1
            elif identity == active:
                tags[index, role] = I
            else:
                if identity in seen:
                    raise ValueError('An event identity reappears noncontiguously; O/B/I prerequisite failed')
                tags[index, role] = B
                seen.add(identity)
                active = identity
    return tags


def obi_to_blocks(tags):
    """Complete decoded tags to role-local runtime IDs, with no centre merge."""
    tags = _integer_matrix(tags, 'Complete O/B/I tags')
    if np.any((tags < O) | (tags > I)):
        raise ValueError('Valid decoded times must be complete O/B/I tags')
    membership = np.full(tags.shape, -1, dtype=np.int64)
    result = []
    for role in range(tags.shape[1]):
        role_indices = []
        active = None
        for query, tag in enumerate(tags[:, role]):
            tag = int(tag)
            if tag == O:
                active = None
            elif tag == B:
                role_indices.append([])
                active = len(role_indices) - 1
                role_indices[active].append(query)
                membership[query, role] = active
            else:
                if active is None:
                    raise ValueError('I cannot start a sequence or follow O')
                role_indices[active].append(query)
                membership[query, role] = active
        result.append([EventBlock(role, identity, tuple(indices))
            for identity, indices in enumerate(role_indices)])
    return membership, result


def reverse_obi(tags, swap_endpoint_roles=True):
    """Reverse query order and rebuild block starts, rather than reversing tags."""
    membership, _ = obi_to_blocks(tags)
    reversed_owners = membership[::-1].copy()
    if swap_endpoint_roles:
        if membership.shape[1] < 2:
            raise ValueError('Endpoint role swap requires at least start and end roles')
        order = [1, 0] + list(range(2, membership.shape[1]))
        reversed_owners = reversed_owners[:, order]
    return teacher_owners_to_obi(reversed_owners)


def fuse_frozen_centres(*args, **kwargs):
    raise NotImplementedError('NOT_IMPLEMENTED: frozen centre fusion needs an approved association/ownership interface')


def build_ownership_features(*args, **kwargs):
    raise NotImplementedError('NOT_IMPLEMENTED: no feature recipe or normalizer has been approved')


def instance_confidences(*args, **kwargs):
    raise NotImplementedError('NOT_IMPLEMENTED: exact instance posterior and training-only calibration are not defined')
