"""Plate support around candidates, computed from XYZ without CAD or labels."""
import numpy as np

VERSION = 'point_plate_context_v1'


class PlateContext:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, float)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or not np.isfinite(self.xyz).all():
            raise ValueError('Expected finite XYZ in millimeters')
        self.projections = {}
        self.planes = {}

    def plane(self, normal, point):
        n = np.asarray(normal, float)
        n /= np.linalg.norm(n)
        if n[np.argmax(np.abs(n))] < 0:
            n = -n
        axis = tuple(n.round(7))
        offset = float(point @ n)
        key = (*axis, round(offset, 3))
        if axis not in self.projections:
            self.projections[axis] = self.xyz @ n
        if key not in self.planes:
            self.planes[key] = self.xyz[np.abs(self.projections[axis] - offset) < .15]
        return self.planes[key], n, self.projections[axis] - offset

    def describe(self, sequence):
        q = np.asarray(sequence['xyz'], float)
        mid = (q[0] + q[-1]) / 2.
        u = np.asarray(sequence['direction'], float)
        feature = np.asarray(sequence['features'])
        axes = feature[len(feature)//2, 60:66].astype(float).reshape(2, 3)
        rows, diagnostics = [], []
        for normal in axes:
            points, n, raw_offset = self.plane(normal, mid)
            cross = np.cross(u, n)
            cross /= np.linalg.norm(cross)
            p = points[np.linalg.norm(points[:, :2] - mid[:2], axis=1) <= 200.]
            if len(p):
                z = p[:, 2] - mid[2]
                zstats = [np.ptp(z), -z.min(), z.max(), *np.quantile(z, [.05, .25, .75, .95])]
                delta = p - mid
                along, side = delta @ u, delta @ cross
                # A center cross-section avoids the candidate's endpoint and
                # process-window differences dominating joint recognition.
                slab = np.abs(along) <= 20.
                rays = []
                for sign in [-1., 1.]:
                    t = sign * side[slab]
                    t = t[(t > .5) & (t <= 100.)]
                    occupied = np.unique(np.floor(t / 5.)).astype(int)
                    rays.append([float(t.max()) if len(t) else 0., len(occupied),
                                 *[int(np.any(np.abs(t - r) <= 5.)) for r in [6., 12., 25., 50., 80.]]])
                rays.sort()
            else:
                zstats, rays = [0.] * 7, [[0.] * 7, [0.] * 7]
            # Nearby parallel faces supply candidate thickness distances.
            local = self.xyz[(np.linalg.norm(self.xyz[:, :2] - mid[:2], axis=1) <= 100.) &
                             (np.abs((self.xyz - mid) @ u) <= 20.) & (np.abs(raw_offset) <= 40.)]
            offsets = local @ n - mid @ n
            rounded, counts = np.unique(np.round(offsets / .05) * .05, return_counts=True)
            peaks = rounded[counts >= max(4, len(local) * .025)]
            separation = np.sort(np.abs(peaks[np.abs(peaks) > .5]))
            separation = np.r_[separation[:4], np.full(max(0, 4-len(separation)), 40.)]
            row = np.r_[zstats, np.ravel(rays), separation]
            rows.append(row.tolist())
            diagnostics.append({'local_plane_points': len(p), 'local_vertical_span_mm': zstats[0],
                                'parallel_face_separations_mm': separation.tolist()})
        rows.sort()
        extents = np.ptp(self.xyz, axis=0)
        # Dimensions and gravity relation are physical context, never a name
        # lookup. They are supplied only to the supervised process split head.
        process = np.r_[np.sort(extents[:2]), extents[2], abs(u[2]),
                        sorted(d['local_vertical_span_mm'] for d in diagnostics)]
        return np.array(rows, np.float32).ravel() / 1000., process.astype(np.float32) / 1000., diagnostics


def contact_descriptor(sequence):
    f = np.asarray(sequence['features'], float)
    center = f[max(0, int(len(f)*.35)):max(1, int(len(f)*.65)), np.r_[0:49, 70:84]]
    return np.r_[np.median(center, axis=0), np.std(center, axis=0)].astype(np.float32)


def endpoint_descriptor(sequence):
    f = np.asarray(sequence['features'], float)[:, np.r_[0:49, 70:84]]
    length = sequence['sample_step_mm'] * (len(f)-1)
    x = np.arange(len(f)) * sequence['sample_step_mm']
    positions = np.r_[np.arange(0., min(80., length)+.01, 5.),
                      length - np.arange(0., min(80., length)+.01, 5.)]
    # Fixed 17 samples at each end; shorter candidates clamp at the opposite
    # end rather than changing dimensionality or using reference endpoints.
    positions = np.r_[np.minimum(np.arange(17)*5., length), np.maximum(length-np.arange(17)*5., 0.)]
    values = np.column_stack([np.interp(positions, x, f[:, j]) for j in range(f.shape[1])])
    return np.r_[length/1000., values.ravel()].astype(np.float32)
