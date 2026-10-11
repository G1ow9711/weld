"""Pure source coordinate inputs; physical geometry stays float64.

This is the XYZ component of a future anchor/domain/context encoder. It is
not a substitute for complete source interfaces, geometry targets or heads.
"""
import math
import numpy as np

VERSION = 'PURE_PCD_INITIAL_XYZ_COMPONENT_270_v1'


def fit_training_normalizer(statistics):
    if len(statistics) != 25 or len({row['pcd']['sha256'] for row in statistics}) != 25:
        raise ValueError('Complete25 distinct original source statistics required')
    if {row['role'] for row in statistics} != {'training', 'development'}:
        raise ValueError('Frozen training/development source roles required')
    for row in statistics:
        if (row['pcd']['processing_units'] != 'mm' or row['pcd']['input_units'] != 'm' or
            not row['pcd']['path'].endswith('_GW1_AllFaces.pcd') or
            row['points'] != row['pcd']['points'] or not row['all_original_rows_and_order_preserved']):
            raise ValueError('Original AllFaces millimetre source statistics required')
    train = [row for row in statistics if row['role'] == 'training']
    if len(train) != 21 or sum(row['role'] == 'development' for row in statistics) != 4:
        raise ValueError('Complete21 training and4 development sources required')
    mean, scatter, count = np.zeros(3, np.float64), np.zeros((3, 3), np.float64), 0
    for row in train:
        n = int(row['points'])
        source_mean = np.asarray(row['mean_mm'], dtype=np.float64)
        covariance = np.asarray(row['population_covariance_mm2'], dtype=np.float64)
        if (n < 1 or source_mean.shape != (3,) or covariance.shape != (3, 3) or
            not np.all(np.isfinite(source_mean)) or not np.all(np.isfinite(covariance))):
            raise ValueError('Finite complete source moments required')
        delta, total = source_mean - mean, count + n
        scatter += covariance * n + np.outer(delta, delta) * (count * n / total)
        mean += delta * (n / total)
        count = total
    covariance = scatter / count
    scale = math.sqrt(float(np.trace(covariance)) / 3.0)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite training-only isotropic scale required')
    return dict(version=VERSION, center_mm=mean.tolist(), isotropic_scale_mm=scale,
        population_covariance_mm2=covariance.tolist(), training_source_count=21,
        training_point_count=count, development_source_count=4,
        training_pcd_sha256=[row['pcd']['sha256'] for row in train],
        feature_channels=['centered_isotropic_x', 'centered_isotropic_y', 'centered_isotropic_z'],
        feature_dtype='float32', authoritative_world_geometry_dtype='float64',
        source_metadata_or_ids_are_neural_channels=False, GT_or_STEP_used=False,
        complete_stage1_anchor_domain_context_features=False, actual_optimizer_updates=0)


def initial_xyz_features(raw_xyz_mm, normalizer):
    points = np.asarray(raw_xyz_mm)
    if points.dtype != np.float64 or points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError('Original finite float64 millimetre coordinates required')
    if normalizer['version'] != VERSION or normalizer['GT_or_STEP_used']:
        raise ValueError('Frozen pure-source training normalizer required')
    centered = (points - np.asarray(normalizer['center_mm'], dtype=np.float64)) / normalizer['isotropic_scale_mm']
    return centered.astype(np.float32)
