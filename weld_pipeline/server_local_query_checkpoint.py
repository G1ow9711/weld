"""Independent strict loading and full TRAIN verification for local XYZ queries."""
from pathlib import Path
from types import FunctionType

import numpy as np

from .server_local_query_weld_model import LocalQueryWeldModel, ARCHITECTURE
from .server_partitioned_full_dataset import PartitionedFullShardDataset
from .server_query_checkpoint_model import verify_complete_query_fit as original
from .sjg_local_runtime_904 import digest, read


COMPLETE_STATUS = 'ALL_TRAIN_LOCAL_XYZ_QUERY_FITTED_NOT_PROMOTED_NOT_COMPLETE_INFERENCE'


def architecture(state):
    if state.get('architecture') != ARCHITECTURE:
        raise ValueError('The declared architecture must include the actual raw XYZ context branch')
    expected = set(LocalQueryWeldModel().state_dict())
    if set(state['state_dict']) != expected:
        raise ValueError('A complete original model and all local branch parameters are required')
    return ARCHITECTURE


def load_local_query_model(state, device='cpu'):
    architecture(state)
    mean, std = np.asarray(state['mean']), np.asarray(state['std'])
    if (mean.shape != (457,) or std.shape != (457,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or not (std > 0).all() or state['optimizer_steps'] < 1):
        raise ValueError('Actual learning and finite original normalization are required')
    model = LocalQueryWeldModel().to(device)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    return model, ARCHITECTURE


def verify_complete_local_query_fit(root, base, state, checkpoint_sha256, receipt_path):
    root, base = Path(root).resolve(), Path(base).resolve()
    base.relative_to(root)
    dataset = PartitionedFullShardDataset(root, base)
    receipt_path = Path(receipt_path).resolve(); receipt_path.relative_to(base)
    fixed = root / 'server_rebuild/full_dataset_v1/training_registry.json'
    actual = base / 'training_registry.json'
    def redirected(path):
        path = Path(path).resolve()
        return actual if path == fixed else path
    namespace = dict(original.__globals__, query_architecture=architecture,
        COMPLETE_STATUSES={ARCHITECTURE:COMPLETE_STATUS},
        read=lambda path:read(redirected(path)), digest=lambda path:digest(redirected(path)))
    verifier = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    result = verifier(root, state, checkpoint_sha256, receipt_path)
    settings = read(receipt_path.parent / 'protocol.json')['settings']
    if (settings['architecture'] != ARCHITECTURE
            or not settings['complete_raw_xyz_required'] or settings['query_local_feature_columns'] != 72
            or settings['query_assignment'] != 'EXISTING_DEPLOYED_DENSE_INPUT_GRID'
            or settings['validation_or_test_label_reads'] != 0
            or state['sources'] != {name:digest(root / name) for name in state['sources']}):
        raise ValueError('The actual local-query sources, full XYZ input or assignment recipe changed')
    for identity, view in state['view_evidence'].items():
        key = identity.split(':', 1)[1]
        job_key, _ = dataset.versions[key]
        job = dataset.jobs[job_key]
        if (view['raw_xyz_sha256'] != job['workpiece_sha256']
                or view['complete_raw_xyz_points'] != job['original_points']
                or len(view['local_context_reports']) != settings['steps_per_view']
                or any(report['input_neighborhood_rows'] != view['reference_children']
                       or report['local_feature_columns'] != 72
                       or report['raw_point_tree_size'] != job['original_points']
                       or not report['queries_belong_to_deployed_input_grid']
                       for report in view['local_context_reports'])):
            raise ValueError('A complete TRAIN view lacks actual full raw XYZ or per-query local context')
    return dict(result, training_registry_sha256=dataset.registry_sha256,
        all_actual_raw_xyz_sources_and_local_context_rows_verified=True,
        held_or_protected_fit_calls=0)
