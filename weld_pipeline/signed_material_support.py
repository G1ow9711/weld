"""PCD-only coupled signed surface support. Material identity remains a hypothesis."""
import numpy as np
from scipy.spatial import cKDTree

VERSION = 'signed_material_support_v1'
FEATURE_COUNT = 166
PARAMETERS = dict(radius_mm=160., plane_slab_mm=.15, along_windows_mm=[8., 24., 80.],
                  transverse_bins_mm=[.5, 2., 4., 8., 16., 32., 64., 128.],
                  parallel_peak_range_mm=40., parallel_peak_bin_mm=.05,
                  parallel_peak_min_points=4, parallel_peak_min_fraction=.025,
                  parallel_peaks_per_signed_side=4, projection_round_decimals=5)


def unit(value):
    value = np.asarray(value, float)
    length = np.linalg.norm(value)
    if not np.isfinite(value).all() or length < 1e-8:
        raise ValueError('Finite nonzero contact axes required')
    return value / length


def frame_views(features):
    """Eight exact equivalent frames: unordered planes and independent normal signs."""
    x = np.asarray(features, np.float32)
    if x.shape[-1] != FEATURE_COUNT:
        raise ValueError('Invalid material support feature width')
    rays = x[..., :84].reshape(*x.shape[:-1], 2, 3, 2, 7)
    extents = x[..., 84:108].reshape(*x.shape[:-1], 2, 3, 2, 2)
    peaks = x[..., 108:156].reshape(*x.shape[:-1], 2, 2, 4, 3)
    views = []
    for exchange in (False, True):
        order = [1, 0] if exchange else [0, 1]
        r, e, p = rays[..., order, :, :, :], extents[..., order, :, :, :], peaks[..., order, :, :, :]
        for flip_a, flip_b in ((False, False), (True, False), (False, True), (True, True)):
            rr, ee, pp = r.copy(), e.copy(), p.copy()
            # The sign on plane A is measured against normal B, and conversely.
            for plane, flip in ((0, flip_b), (1, flip_a)):
                if flip:
                    rr[..., plane, :, :, :] = rr[..., plane, :, ::-1, :]
                    ee[..., plane, :, :, :] = ee[..., plane, :, ::-1, :]
            for axis, flip in ((0, flip_a), (1, flip_b)):
                if flip:
                    pp[..., axis, :, :, :] = pp[..., axis, ::-1, :, :]
            views.append(np.concatenate((rr.reshape(*x.shape[:-1], 84),
                                         ee.reshape(*x.shape[:-1], 24),
                                         pp.reshape(*x.shape[:-1], 48), x[..., 156:]), axis=-1))
    return views


class SignedMaterialSupport:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, float)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or not np.isfinite(self.xyz).all():
            raise ValueError('Finite PCD XYZ required')
        self.tree = cKDTree(self.xyz)

    def describe(self, query, normals, tangent):
        query = np.asarray(query, float)
        axes = np.stack([unit(axis) for axis in normals])
        along = unit(tangent)
        rows = self.tree.query_ball_point(query, PARAMETERS['radius_mm'] + 1e-7)
        delta = self.xyz[rows] - query
        projections = np.round(delta @ axes.T, 5)
        t = np.round(delta @ along, 5)
        distances = np.round(np.linalg.norm(delta, axis=1), 5)
        ray_features, extent_features, peak_features = [], [], []
        bins = np.asarray(PARAMETERS['transverse_bins_mm'])
        for plane in range(2):
            plane_distance = projections[:, plane]
            transverse = projections[:, 1-plane]
            for window in PARAMETERS['along_windows_mm']:
                mask = (np.abs(plane_distance) <= .15) & (np.abs(t) <= window)
                for sign in (-1., 1.):
                    ray = sign * transverse[mask]
                    ray = ray[(ray >= .5) & (ray <= 128.)]
                    counts = np.histogram(ray, bins=bins)[0]
                    ray_features.extend(np.log1p(counts) / 10.)
                    extent_features.extend([float(ray.max()/128.) if len(ray) else 0., np.log1p(len(ray))/10.])
        for axis in range(2):
            supported = (np.abs(t) <= 24.) & (np.abs(projections[:, 1-axis]) <= 64.)
            values = projections[supported, axis]
            for sign in (-1., 1.):
                signed = sign * values
                signed = signed[(signed > .2) & (signed <= 40.)]
                centers, counts = np.unique(np.round(signed/.05).astype(np.int64), return_counts=True)
                keep = counts >= max(4., .025*int(supported.sum()))
                centers, counts = centers[keep], counts[keep]
                order = np.argsort(centers, kind='stable')[:4]
                selected = [(float(centers[i]*.05/40.), float(np.log1p(counts[i])/10.), 1.) for i in order]
                selected += [(0., 0., 0.)] * (4-len(selected))
                peak_features.extend(np.asarray(selected).ravel())
        if len(delta) >= 3:
            covariance = np.cov(np.round(delta, 5).T)
            eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0., None)
            eigenvalues /= max(float(eigenvalues.sum()), 1e-12)
        else:
            eigenvalues = np.zeros(3)
        nearest = float(self.tree.query(query, k=1)[0])
        basic = [abs(float(axes[0] @ axes[1])), min(nearest, 160.)/160., np.log1p(len(delta))/10.,
                 *[np.log1p(int((distances <= radius).sum()))/10. for radius in (8., 24., 80.)],
                 *eigenvalues, float(np.abs(axes @ along).sum()/2.)]
        features = np.asarray([*ray_features, *extent_features, *peak_features, *basic], np.float32)
        if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
            raise ValueError('Invalid signed material descriptor')
        return features

    def describe_sequence(self, sequence, indices):
        normals = np.asarray(sequence['features'], float)[:, 60:66].reshape(-1, 2, 3)
        return np.stack([self.describe(sequence['xyz'][i], normals[i], sequence['direction']) for i in indices])


def predict_equivalent_views(model, features):
    """The averaged classifier is exactly invariant to material-normal sign and order."""
    return np.mean([model.predict_proba(view) for view in frame_views(features)], axis=0)
