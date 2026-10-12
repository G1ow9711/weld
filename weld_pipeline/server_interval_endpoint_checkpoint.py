"""Strict complete training admission for the independent endpoint model."""
from pathlib import Path
from types import FunctionType

import numpy as np

from .server_interval_endpoint_residual_model import IntervalEndpointResidualModel, ARCHITECTURE
from .server_local_query_checkpoint import verify_complete_local_query_fit as original_verify
from .sjg_local_runtime_904 import digest, read


COMPLETE_STATUS = 'ALL_TRAIN_LOCAL_XYZ_ENDPOINT_RESIDUAL_FITTED_NOT_PROMOTED_NOT_COMPLETE_INFERENCE'


def architecture(state):
    if (state.get('architecture') != ARCHITECTURE
            or set(state['state_dict']) != set(IntervalEndpointResidualModel().state_dict())):
        raise ValueError('All original and endpoint correction parameters are required')
    return ARCHITECTURE


def load_interval_endpoint_model(state, device='cpu'):
    architecture(state)
    mean, std = np.asarray(state['mean']), np.asarray(state['std'])
    if (mean.shape != (457,) or std.shape != (457,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or not (std > 0).all()
            or state['optimizer_steps'] < 1):
        raise ValueError('Actual training and finite warm normalization are required')
    model = IntervalEndpointResidualModel().to(device)
    model.load_state_dict(state['state_dict'], strict=True); model.eval()
    return model, ARCHITECTURE


def verify_complete_interval_endpoint_fit(root, base, state, checkpoint_sha256, receipt_path):
    receipt_path = Path(receipt_path).resolve()
    receipt_path.relative_to(Path(base).resolve())
    receipt = read(receipt_path)
    if receipt.get('status') != COMPLETE_STATUS:
        raise ValueError('An incomplete or diagnostic endpoint model cannot enter full training admission')
    protocol = read(receipt_path.parent / 'protocol.json')
    if not isinstance(protocol.get('settings'), dict):
        raise ValueError('Complete endpoint fitting settings are required')
    # Preserve every original view, role, source, update and XYZ check. Only the
    # declared model class, parameter set and complete status change.
    namespace = dict(original_verify.__globals__, ARCHITECTURE=ARCHITECTURE,
        COMPLETE_STATUS=COMPLETE_STATUS, architecture=architecture)
    verifier = FunctionType(original_verify.__code__, namespace, original_verify.__name__,
                            original_verify.__defaults__, original_verify.__closure__)
    result = verifier(root, base, state, checkpoint_sha256, receipt_path)
    settings = protocol['settings']
    if (settings['endpoint_offset_unit'] != 'metres'
            or settings['endpoint_projection_backward'] != 'IDENTITY_ON_INPUT_QUERY_BOUND_PROJECTION'
            or settings['endpoint_residual_source_sha256'] != digest(
                Path(root) / 'weld_pipeline/server_interval_endpoint_residual_model.py')
            or len(settings['endpoint_control_receipt_sha256']) < 2):
        raise ValueError('The actual endpoint parameterization or its controls changed')
    return dict(result, endpoint_parameterization_and_sources_verified=True)
