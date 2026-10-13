"""Input-only image features with complete grid context for arbitrarily large views.

Use the established dense producer within its budget. Larger views retain the
same grid and filter halo; tiling never drops points or increases pixel size.
"""
import numpy as np
from scipy.ndimage import uniform_filter
from .server_multiview_visual_features import visual_features as dense_features, VIEWS, RADII

SCHEMA = 'ORTHOGRAPHIC_RAW_POINT_DEPTH_DENSITY_60_V1'


def _tile_values(ij, depth, shape):
    idx = ij[:, 1] * shape[1] + ij[:, 0]
    count = np.bincount(idx, minlength=shape[0] * shape[1]).astype(np.float32)
    if int(count.sum(dtype=np.float64)) != len(ij):
        raise ValueError('Point-to-image map lost input points')
    near = np.full(len(count), np.inf, np.float32)
    far = np.full(len(count), -np.inf, np.float32)
    np.minimum.at(near, idx, depth); np.maximum.at(far, idx, depth)
    occupied = count > 0
    near[~occupied] = far[~occupied] = 0
    span = far - near
    values = [np.log1p(count[idx]), depth - near[idx], far[idx] - depth, span[idx], depth]
    mask = occupied.astype(np.float32).reshape(shape)
    mid = ((near + far) * .5).reshape(shape)
    for size in RADII:
        smooth = lambda x: uniform_filter(x, size=size, mode='constant', cval=0)
        mass = smooth(mask); denom = np.maximum(mass, 1e-10)
        mean = smooth(mid) / denom
        variance = np.maximum(smooth(mid * mid) / denom - mean * mean, 0)
        values.extend((mass.ravel()[idx], np.log1p(smooth(count.reshape(shape)).ravel()[idx]),
                       mean.ravel()[idx] - depth, np.sqrt(variance).ravel()[idx],
                       (smooth(span.reshape(shape)) / denom).ravel()[idx]))
    return np.column_stack(values)


def visual_features(xyz, pixel_mm=2., max_pixels=32_000_000, tile_size=1024):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Finite nonempty complete XYZ is required')
    if not np.isfinite(pixel_mm) or pixel_mm <= 0 or max_pixels < 1 or tile_size < 1:
        raise ValueError('Positive pixel size and image allocation required')
    metadata = []
    for name, (a, b, d) in VIEWS:
        origin = np.floor(xyz[:, [a, b]].min(0) / pixel_mm) * pixel_mm
        ij = np.floor((xyz[:, [a, b]] - origin) / pixel_mm).astype(np.int64)
        shape = (int(ij[:, 1].max()) + 1, int(ij[:, 0].max()) + 1)
        metadata.append((name, (a, b, d), origin, ij, shape))
    if all(shape[0] * shape[1] <= max_pixels for _, _, _, _, shape in metadata):
        return dense_features(xyz, pixel_mm=pixel_mm, max_pixels=max_pixels)
    result = np.empty((len(xyz), 60), np.float32); audit = []
    for view, (name, axes, origin, ij, shape) in enumerate(metadata):
        depth = ((xyz[:, axes[2]] - xyz[:, axes[2]].min()) / 1000.).astype(np.float32)
        view_result = result[:, view * 20:(view + 1) * 20]
        tiles = 1
        if shape[0] * shape[1] <= max_pixels:
            view_result[:] = _tile_values(ij, depth, shape)
        else:
            halo = max(RADII) // 2
            if (tile_size + 2 * halo) ** 2 > max_pixels:
                raise ValueError('Declared tile allocation exceeded; input was not truncated')
            tile_xy = ij // tile_size
            unique_tiles = np.unique(tile_xy, axis=0); tiles = len(unique_tiles)
            assigned = np.zeros(len(xyz), bool)
            for tx, ty in unique_tiles:
                core = (tile_xy[:, 0] == tx) & (tile_xy[:, 1] == ty)
                x0 = max(0, int(tx) * tile_size - halo); y0 = max(0, int(ty) * tile_size - halo)
                x1 = min(shape[1], (int(tx) + 1) * tile_size + halo)
                y1 = min(shape[0], (int(ty) + 1) * tile_size + halo)
                context = (ij[:, 0] >= x0) & (ij[:, 0] < x1) & (ij[:, 1] >= y0) & (ij[:, 1] < y1)
                values = _tile_values(ij[context] - [x0, y0], depth[context], (y1 - y0, x1 - x0))
                if assigned[core].any():
                    raise ValueError('Point written by more than one tile core')
                view_result[core] = values[core[context]]; assigned[core] = True
            if not assigned.all():
                raise ValueError('Tiled visual input lost raw points')
        audit.append(dict(view=name, axes=list(axes), shape=list(shape), origin_mm=origin.tolist(),
            pixel_mm=pixel_mm, input_points=len(xyz), mapped_points=len(xyz), tiles=tiles))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite visual features')
    labels = ['log_point_count','depth_from_near_m','depth_from_far_m','pixel_depth_span_m','depth_from_input_min_m']
    labels += [f'window_{size}_{name}' for size in RADII for name in
               ['occupied','log_count','depth_difference_m','depth_std_m','depth_span_m']]
    return result, dict(schema=SCHEMA, columns=[name+'/'+label for name,_ in VIEWS for label in labels],
        views=audit, label_reads=0, original_point_order_preserved=True, rotation_invariance_claimed=False)
