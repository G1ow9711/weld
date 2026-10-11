"""Fixed-budget endpoint proposals prioritized by distinct score peaks.

This operates on model scores and input event order only. It introduces no
spatial threshold, target, physical-contact assertion, or new endpoint. Flat
maxima use their first input ID; finite-domain endpoints can be maxima. If fewer
than k peaks exist, remaining slots use the original stable score ranking.
"""
import numpy as np
import torch
from .server_grid_endpoint_selector import StreamingEndpointTopK


def peak_first_indices(scores, legal, k=8):
    scores = np.asarray(scores)
    legal = np.asarray(legal)
    if scores.ndim != 1 or legal.shape != scores.shape or legal.dtype != np.bool_:
        raise ValueError('One ordered score vector and boolean legal mask required')
    if k < 1 or not np.isfinite(scores).all():
        raise ValueError('Finite scores and positive budget required')
    if not len(scores) or not legal.any():
        return np.empty(0, np.int64)
    values = np.where(legal, scores, -np.inf)
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    levels = values[starts]
    previous = np.r_[-np.inf, levels[:-1]]
    following = np.r_[levels[1:], -np.inf]
    peaks = starts[np.isfinite(levels) & (levels > previous) & (levels > following)]
    selected = peaks[np.argsort(-values[peaks], kind='stable')[:k]].tolist()
    if len(selected) < k:
        used = set(selected)
        for index in np.argsort(-values, kind='stable'):
            if not legal[index]:
                break
            if int(index) not in used:
                selected.append(int(index))
                if len(selected) == k:
                    break
    # Keep the original score as the pair score; peak status is only proposal
    # allocation, never an added score or a prediction of contact.
    selected = np.asarray(selected, np.int64)
    return selected[np.lexsort((selected, -values[selected]))]


class PeakEndpointCandidates(StreamingEndpointTopK):
    """Reuse bounded pair construction after peak-first endpoint selection.

    Features remain streamed. Only scalar scores, positions and IDs are retained
    per profile, with an explicit event cap matching the finite-grid generator.
    """
    def __init__(self, k=8, max_events=1_000_001):
        super().__init__(k)
        self.max_events = max_events
        self.count = 0
        self.chunks = []
        self.device = None
        self.dtype = None
        self.batch = None
        self.last_position = None

    @torch.no_grad()
    def add(self, logits, legal, positions_mm, event_ids):
        if logits.ndim != 3 or logits.shape[-1] != 2 or legal.shape != logits.shape or legal.dtype != torch.bool:
            raise ValueError('Expected B x events x 2 scores and legal mask')
        if positions_mm.shape != logits.shape[:2] or event_ids.shape != (logits.shape[1],) or event_ids.dtype != torch.long:
            raise ValueError('Invalid input position or ID shape')
        if not torch.isfinite(logits).all() or not torch.isfinite(positions_mm).all():
            raise ValueError('Nonfinite input')
        if not len(event_ids):
            return
        if int(event_ids[0]) <= self.last_id or not torch.all(event_ids[1:] > event_ids[:-1]):
            raise ValueError('Strictly increasing input IDs required')
        if not torch.all(positions_mm[:,1:] > positions_mm[:,:-1]):
            raise ValueError('Strictly increasing event coordinates required')
        if self.last_position is not None and not torch.all(positions_mm[:,0] > self.last_position):
            raise ValueError('Chunks must continue the same ordered event domain')
        if self.batch is not None and (self.batch != len(logits) or self.device != logits.device or self.dtype != logits.dtype):
            raise ValueError('Chunk batch/device/dtype mismatch')
        if self.count + len(event_ids) > self.max_events:
            raise ValueError('Fixed event memory budget exceeded')
        self.last_id = int(event_ids[-1])
        self.last_position = positions_mm[:,-1].clone()
        self.count += len(event_ids)
        self.batch, self.device, self.dtype = len(logits), logits.device, logits.dtype
        self.chunks.append((logits.double().cpu().numpy().copy(), legal.cpu().numpy().copy(),
                            positions_mm.double().cpu().numpy().copy(), event_ids.cpu().numpy().copy()))

    def pairs(self):
        if not self.chunks:
            return None
        scores = np.concatenate([v[0] for v in self.chunks], axis=1)
        legal = np.concatenate([v[1] for v in self.chunks], axis=1)
        positions = np.concatenate([v[2] for v in self.chunks], axis=1)
        ids = np.concatenate([v[3] for v in self.chunks])
        n = min(self.k, len(ids))
        chosen_scores = np.full((self.batch,n,2), -np.inf, scores.dtype)
        chosen_ids = np.zeros((self.batch,n,2), np.int64)
        chosen_positions = np.zeros((self.batch,n,2), np.float64)
        for batch in range(self.batch):
            for side in (0,1):
                chosen = peak_first_indices(scores[batch,:,side], legal[batch,:,side], self.k)
                size = len(chosen)
                chosen_scores[batch,:size,side] = scores[batch,chosen,side]
                chosen_ids[batch,:size,side] = ids[chosen]
                chosen_positions[batch,:size,side] = positions[batch,chosen]
        self.scores = torch.as_tensor(chosen_scores, device=self.device, dtype=self.dtype)
        self.ids = torch.as_tensor(chosen_ids, device=self.device)
        self.positions = torch.as_tensor(chosen_positions, device=self.device)
        return super().pairs()
