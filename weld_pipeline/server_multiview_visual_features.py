"""Input-only orthographic density/depth context, mapped to every original XYZ.

These image features propose weld regions; they do not define weld curves.
Min/max depth retain two visible layers, not a complete occlusion model.
No labels, candidate carriers, source identities or fitted state enter here.
"""
import numpy as np
from scipy.ndimage import uniform_filter

VIEWS = (("XY", (0, 1, 2)), ("XZ", (0, 2, 1)), ("YZ", (1, 2, 0)))
RADII = (3, 9, 25)


def visual_features(xyz, pixel_mm=2., max_pixels=32_000_000):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Finite nonempty complete XYZ is required")
    if not np.isfinite(pixel_mm) or pixel_mm <= 0:
        raise ValueError("Positive pixel size required")
    result = np.empty((len(xyz), 3 * (5 + 5 * len(RADII))), np.float32)
    columns, audit = [], []
    col = 0
    for name, (a, b, d) in VIEWS:
        origin = np.floor(xyz[:, [a, b]].min(0) / pixel_mm) * pixel_mm
        ij = np.floor((xyz[:, [a, b]] - origin) / pixel_mm).astype(np.int64)
        shape = (int(ij[:, 1].max()) + 1, int(ij[:, 0].max()) + 1)
        if shape[0] * shape[1] > max_pixels:
            raise ValueError("Declared image allocation exceeded; input was not truncated")
        idx = ij[:, 1] * shape[1] + ij[:, 0]
        count = np.bincount(idx, minlength=shape[0] * shape[1]).astype(np.float32)
        if int(count.sum(dtype=np.float64)) != len(xyz):
            raise ValueError("Point-to-image map lost input points")
        depth = ((xyz[:, d] - xyz[:, d].min()) / 1000.).astype(np.float32)
        near = np.full(len(count), np.inf, np.float32)
        far = np.full(len(count), -np.inf, np.float32)
        np.minimum.at(near, idx, depth)
        np.maximum.at(far, idx, depth)
        occupied = count > 0
        near[~occupied] = far[~occupied] = 0
        span = far - near

        def emit(label, values):
            nonlocal col
            result[:, col] = values
            columns.append(name + "/" + label)
            col += 1

        emit("log_point_count", np.log1p(count[idx]))
        emit("depth_from_near_m", depth - near[idx])
        emit("depth_from_far_m", far[idx] - depth)
        emit("pixel_depth_span_m", span[idx])
        emit("depth_from_input_min_m", depth)
        mask = occupied.astype(np.float32).reshape(shape)
        mid = ((near + far) * .5).reshape(shape)
        for size in RADII:
            smooth = lambda x: uniform_filter(x, size=size, mode="constant", cval=0)
            mass = smooth(mask)
            denom = np.maximum(mass, 1e-10)
            mean = smooth(mid) / denom
            variance = np.maximum(smooth(mid * mid) / denom - mean * mean, 0)
            emit(f"window_{size}_occupied", mass.ravel()[idx])
            emit(f"window_{size}_log_count", np.log1p(smooth(count.reshape(shape)).ravel()[idx]))
            emit(f"window_{size}_depth_difference_m", mean.ravel()[idx] - depth)
            emit(f"window_{size}_depth_std_m", np.sqrt(variance).ravel()[idx])
            emit(f"window_{size}_depth_span_m", (smooth(span.reshape(shape)) / denom).ravel()[idx])
        audit.append(dict(view=name, axes=[a, b, d], shape=list(shape),
                          origin_mm=origin.tolist(), pixel_mm=pixel_mm,
                          input_points=len(xyz), mapped_points=int(count.sum(dtype=np.float64))))
    if col != result.shape[1] or not np.isfinite(result).all():
        raise ValueError("Nonfinite or incomplete visual features")
    return result, dict(columns=columns, views=audit, label_reads=0,
                        original_point_order_preserved=True,
                        rotation_invariance_claimed=False)
