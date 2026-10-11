"""Conservative angular support union of observed finite graph edges."""
import numpy as np


def supported_circle_intervals(vertices, edges, center, radius, band_mm=3.):
    """Use only edges wholly inside an annulus; never bridge positive gaps.

    Band bounds geometric candidate support, not physical weld/contact accuracy.
    Returns unwrapped [lo, hi] radians; a wrap interval may end above 2*pi.
    Radii <= band are excluded because radial projection can be singular.
    """
    if not np.isfinite(radius) or not np.isfinite(band_mm) or band_mm <= 0:
        raise ValueError('Finite radius and positive finite band required')
    if radius <= band_mm or not len(edges):
        return []
    relative = np.asarray(vertices, float) - np.asarray(center, float)
    a = relative[edges[:, 0]]; b = relative[edges[:, 1]]; delta = b-a
    vv = np.sum(delta*delta, axis=1)
    t = np.clip(-np.sum(a*delta, axis=1)/np.where(vv > 0, vv, 1), 0, 1)
    minimum = np.linalg.norm(a+t[:, None]*delta, axis=1)
    maximum = np.maximum(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    good = (vv > 0) & (minimum >= radius-band_mm) & (maximum <= radius+band_mm)
    # Every retained straight segment stays outside the center and in the band.
    tau = 2*np.pi
    # Compute each vertex angle once: shared endpoints have identical numbers,
    # avoiding direction-dependent roundoff gaps from start + delta arithmetic.
    angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), tau)
    intervals=[]
    for first, second in angles[edges[good]]:
        lo,hi = sorted((float(first),float(second)))
        if hi == lo:
            continue
        if hi-lo <= np.pi:
            intervals.append((lo,hi))
        else:
            intervals.extend(((hi,tau),(0.,lo)))
    merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo,hi])
    if len(merged)>1 and merged[0][0]==0. and merged[-1][1]==tau:
        wrap=[merged[-1][0], merged[0][1]+tau]
        merged=merged[1:-1]+[wrap]
    return merged
