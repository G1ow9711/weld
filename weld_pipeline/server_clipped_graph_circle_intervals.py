"""Angular projection of observed graph-edge portions inside a circle annulus."""
import numpy as np


def clipped_supported_circle_intervals(vertices, edges, center, radius, band_mm=3.):
    """Clip existing segments, never add edges or fill positive angular gaps.

    The annulus expresses candidate support tolerance, not weld/contact truth.
    """
    if not np.isfinite(radius) or not np.isfinite(band_mm) or band_mm <= 0:
        raise ValueError('Finite radius and positive finite band required')
    if radius <= band_mm or not len(edges):
        return []
    relative = np.asarray(vertices, float) - np.asarray(center, float)
    edges = np.asarray(edges, int)
    a = relative[edges[:, 0]]
    d = relative[edges[:, 1]] - a
    aa = np.sum(d*d, axis=1)
    valid = aa > 0
    # Parameter of closest point on the infinite edge line.
    middle = -np.sum(a*d, axis=1) / np.where(valid, aa, 1.)
    closest = a + middle[:, None]*d
    closest2 = np.sum(closest*closest, axis=1)
    outer2 = (radius+band_mm)**2 - closest2
    inner2 = (radius-band_mm)**2 - closest2
    outer_half = np.sqrt(np.maximum(outer2, 0) / np.where(valid, aa, 1.))
    inner_half = np.sqrt(np.maximum(inner2, 0) / np.where(valid, aa, 1.))
    lo = np.maximum(0., middle-outer_half)
    hi = np.minimum(1., middle+outer_half)
    valid &= (outer2 > 0) & (hi > lo)
    tau = 2*np.pi
    vertex_angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), tau)
    intervals = []
    for i in np.flatnonzero(valid):
        pieces = [(lo[i], hi[i])]
        if inner2[i] > 0:
            pieces = [(lo[i], min(hi[i], middle[i]-inner_half[i])),
                      (max(lo[i], middle[i]+inner_half[i]), hi[i])]
        for start, end in pieces:
            if end <= start:
                continue
            angles = []
            for t in (start, end):
                if t == 0:
                    angle = vertex_angles[edges[i, 0]]
                elif t == 1:
                    angle = vertex_angles[edges[i, 1]]
                else:
                    p = a[i]+t*d[i]
                    angle = np.mod(np.arctan2(p[1], p[0]), tau)
                angles.append(float(angle))
            left, right = sorted(angles)
            if left == right:
                continue
            if right-left <= np.pi:
                intervals.append((left, right))
            else:
                intervals.extend(((right, tau), (0., left)))
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    if len(merged)>1 and merged[0][0] == 0. and merged[-1][1] == tau:
        merged = merged[1:-1] + [[merged[-1][0], merged[0][1]+tau]]
    return merged
