"""Shared local point profiles for horizontal heatmaps, with no world origin."""
import numpy as np

from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.sequence_augmentation import change_frame

VERSION = 'horizontal_local_patch_heatmaps_v1'
LOCAL = np.r_[0:49, 70:84]
OFFSETS_MM = np.array([-32., -8., -2., 0., 2., 8., 32.])
FEATURE_COUNT = 521


def equivalent_frame(sequence, target, reverse=False, flip=False):
    feature = np.array(sequence['features'], dtype=np.float32).T.copy()
    labels = np.array(target, dtype=np.float32).T.copy()
    valid = np.ones((1, feature.shape[1]), np.float32)
    change_frame(feature, labels, valid, reverse=reverse, flip_normal=flip)
    xyz = np.asarray(sequence['xyz'])[::-1] if reverse else sequence['xyz']
    direction = -np.asarray(sequence['direction']) if reverse else sequence['direction']
    return {**sequence, 'features': feature.T, 'xyz': xyz, 'direction': direction}, labels.T


def patch_features(sequence, indices, structural, mean, covariance):
    f = np.asarray(sequence['features'], dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(f)):
        raise ValueError('Patch centers outside the candidate')
    step = float(sequence['sample_step_mm'])
    positions = indices[:, None] + OFFSETS_MM[None, :] / step
    covered = ((positions >= 0) & (positions <= len(f)-1)).astype(np.float32)
    clipped = np.clip(positions, 0., len(f)-1)
    lower = clipped.astype(np.int64)
    upper = np.minimum(lower+1, len(f)-1)
    fraction = (clipped-lower)[..., None]
    local = f[:, LOCAL]
    profiles = local[lower] * (1.-fraction) + local[upper] * fraction
    basis = f[indices, 57:66].astype(float).reshape(-1, 3, 3)
    centered = np.asarray(sequence['xyz'])[indices].astype(float)-np.asarray(mean)
    relative = np.einsum('ni,nji->nj', centered, basis) / 1000.
    projected = np.einsum('nai,ij,nbj->nab', basis, np.asarray(covariance), basis) / 1e6
    cov = projected[:, np.triu_indices(3)[0], np.triu_indices(3)[1]]
    context = np.broadcast_to(np.asarray(structural, np.float32), (len(indices), 50))
    result = np.column_stack([profiles.reshape(len(indices), -1), covered, f[indices, 55:57],
        f[indices, 57:66], np.sort(f[indices, 52:55], axis=1), relative, cov, context]).astype(np.float32)
    if result.shape != (len(indices), FEATURE_COUNT) or not np.isfinite(result).all():
        raise ValueError('Invalid local patch descriptor')
    return result


def training_indices(target, rng):
    y = np.asarray(target)
    known = np.all(y != -1., axis=1)
    candidates = np.flatnonzero(known)
    if not len(candidates):
        return np.empty(0, dtype=np.int64)
    picks = [rng.choice(candidates, min(40, len(candidates)), replace=False)]
    foreground = np.flatnonzero(known & (y[:, 0] > .5))
    if len(foreground):
        picks.append(rng.choice(foreground, min(24, len(foreground)), replace=False))
    event = np.flatnonzero(known & (y[:, 1:4].max(axis=1) > .05))
    if len(event):
        picks.append(rng.choice(event, min(48, len(event)), replace=False))
    return np.unique(np.concatenate(picks))


def infer_heatmaps(bundle, sequences, structural, mean, covariance):
    probabilities = []
    for i, seq in enumerate(sequences):
        p = np.zeros((len(seq['features']), 7), np.float32)
        if seq['type'] == 'horizontal' and seq.get('curve_kind') != 'arc':
            for start in range(0, len(p), 4096):
                indices = np.arange(start, min(start+4096, len(p)))
                x = patch_features(seq, indices, structural[i], mean, covariance)
                p[indices, :4] = np.clip(bundle['heatmaps'].predict(x), 0., 1.)
        probabilities.append(p)
    return probabilities


def predict_straight(bundle, sequences, structural, mean, covariance):
    p = infer_heatmaps(bundle, sequences, structural, mean, covariance)
    return decode_intervals(sequences, p, threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
