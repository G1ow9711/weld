"""Point-only finite-contact span candidates, with unresolved datum explicit."""
import numpy as np

from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates

VERSION = 'point_contact_plate_extent_residual_v1'


def physical_span_baseline(features):
    x = np.asarray(features)
    # The 8 mm local window: extent along projected gravity, then convert to Z.
    return x[..., 13] * 1000. * np.sqrt(np.maximum(0., 1. - x[..., 1] ** 2))


def predict_contact_extent(bundle, profile, point, normal):
    if not isinstance(profile, FiniteContactPlates):
        raise TypeError('Raw XYZ-derived finite contact profile required')
    features, observation = profile.observe(point, normal)
    if features[0] == 0:
        return {'state': 'unknown', 'value': None, 'reason': 'NO_OBSERVABLE_CONTACT_FACE'}
    residual = float(bundle['span_residual'].predict(features[None, :])[0])
    span = float(physical_span_baseline(features) + residual)
    if span <= 0 or not np.isfinite(span):
        return {'state': 'unknown', 'value': None, 'reason': 'INVALID_LEARNED_CONTACT_EXTENT'}
    return {'state': 'candidate_not_reviewed', 'contact_local_vertical_span_mm': span,
        'physical_baseline_mm': float(physical_span_baseline(features)),
        'learned_residual_mm': residual, 'local_base_candidates': observation['horizontal_base_candidates'],
        'reasons': observation['reasons'], 'source': 'PCD_physics_with_FIT_offline_CAD_supervision',
        'contract_height_datum_resolved': False, 'full_contract_pass': False}
