"""Strict factories and complete TRAIN evidence for anchored query checkpoints.

Loading never opens a label. Historical anchored checkpoints lack an explicit
architecture tag; a recall checkpoint must carry its tag and both new head
parameters. Whole-model strict loading rejects omitted or unexpected weights.
"""
from pathlib import Path

import numpy as np

from .server_legacy_anchored_full_weld_model import LegacyAnchoredFullWeldModel
from .server_reference_recall_weld_model import ReferenceRecallWeldModel, ARCHITECTURE
from .sjg_local_runtime_904 import read, digest

ANCHORED_ARCHITECTURE = 'LEGACY_ANCHORED_FULL_WELD_V2'
COMPLETE_STATUSES = {
    ANCHORED_ARCHITECTURE: 'ALL_TRAIN_WARMSTART_FITTED_NOT_PROMOTED_NOT_COMPLETE_INFERENCE',
    ARCHITECTURE: 'ALL_TRAIN_REFERENCE_RECALL_FITTED_NOT_PROMOTED_NOT_COMPLETE_INFERENCE',
}


def query_architecture(state):
    weights = state['state_dict']
    if 'legacy.trunk.0.weight' not in weights:
        raise ValueError('A complete anchored query checkpoint is required')
    recall_keys = {name for name in weights if name.startswith('reference_delta.')}
    if recall_keys:
        if (recall_keys != {'reference_delta.weight', 'reference_delta.bias'}
                or state.get('architecture') != ARCHITECTURE):
            raise ValueError('Recall admission weights require their complete explicit architecture')
        return ARCHITECTURE
    if state.get('architecture', ANCHORED_ARCHITECTURE) != ANCHORED_ARCHITECTURE:
        raise ValueError('Declared query architecture differs from actual weights')
    return ANCHORED_ARCHITECTURE


def load_query_model(state, device='cpu'):
    architecture = query_architecture(state)
    mean, std = np.asarray(state['mean']), np.asarray(state['std'])
    if (mean.shape != (457,) or std.shape != (457,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or not (std > 0).all()
            or state['optimizer_steps'] < 1):
        raise ValueError('Actual fitting and finite original TRAIN normalization are required')
    model = (ReferenceRecallWeldModel() if architecture == ARCHITECTURE
             else LegacyAnchoredFullWeldModel()).to(device)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    return model, architecture


def verify_complete_query_fit(root, state, checkpoint_sha256, receipt_path):
    """Require every registered view and its actual per-step fitting evidence.

Only TRAIN registration, receipts and checkpoint bookkeeping are inspected.
No training/held reference geometry or process label is opened here.
"""
    root = Path(root).resolve()
    receipt_path = Path(receipt_path).resolve()
    receipt_path.relative_to(root)
    receipt = read(receipt_path)
    protocol_path = receipt_path.parent / 'protocol.json'
    protocol = read(protocol_path)
    settings = protocol['settings']
    registry_path = root / 'server_rebuild/full_dataset_v1/training_registry.json'
    registration = read(registry_path)
    versions = {v['key']: v for job in registration['cases'] for v in job['versions']}
    architecture = query_architecture(state)
    if (receipt['status'] != COMPLETE_STATUSES[architecture]
            or receipt['checkpoint_sha256'] != checkpoint_sha256
            or receipt['protocol_sha256'] != state['protocol_sha256']
            or digest(protocol_path) != state['protocol_sha256']
            or settings['registry_sha256'] != digest(registry_path)
            or state['sources'] != settings['sources']
            or receipt['training_versions'] != len(versions)
            or receipt['validation_or_test_label_reads'] != 0
            or settings['validation_or_test_label_reads'] != 0
            or receipt['epochs'] != settings['epochs'] or receipt['epochs'] < 1
            or receipt['optimizer_steps'] != state['optimizer_steps']):
        raise ValueError('Complete query fitting receipt differs from the actual checkpoint and TRAIN protocol')
    if architecture == ARCHITECTURE and (receipt.get('architecture') != architecture
            or settings.get('architecture') != architecture):
        raise ValueError('Complete recall query evidence lacks its architecture binding')
    expected = {str(e) + ':' + key for e in range(receipt['epochs']) for key in versions}
    if (set(state['completed_views']) != expected or len(state['completed_views']) != len(expected)
            or receipt['completed_views'] != len(expected)
            or set(state['view_evidence']) != expected):
        raise ValueError('Query has not actually fitted every registered TRAIN view in every epoch')
    steps = settings['steps_per_view']
    if steps < 1:
        raise ValueError('Positive complete per-view fitting steps are required')
    updates = 0
    for identity in expected:
        view = state['view_evidence'][identity]
        key = identity.split(':', 1)[1]
        if (view['label_receipt_sha256'] != versions[key]['label_receipt_sha256']
                or view['reference_children'] != versions[key]['children']
                or len(view['fit_results']) != steps or len(view['reports']) != steps
                or any(not np.isfinite(result['loss']) for result in view['fit_results'])):
            raise ValueError('A complete registered view lacks actual full-child fitting evidence')
        updates += len(view['fit_results'])
    if updates != state['optimizer_steps']:
        raise ValueError('Actual per-view updates differ from the query optimizer counter')
    return dict(query_architecture=architecture, query_receipt_sha256=digest(receipt_path),
                query_protocol_sha256=digest(protocol_path), query_epochs=receipt['epochs'],
                actual_optimizer_steps=updates, all_registered_train_views_fitted=True,
                validation_or_test_or_protected_fit_calls=0)
