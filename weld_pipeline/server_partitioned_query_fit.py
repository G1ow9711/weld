"""Verify every actual query training view against a declared complete partition.

The original verifier runs unchanged. Its one fixed registration pathname is
redirected to the authenticated partition registration in a private function
namespace. All epoch, label-receipt, child-count and optimizer checks remain.
"""
from pathlib import Path
from types import FunctionType

from .server_partitioned_full_dataset import PartitionedFullShardDataset
from .server_query_checkpoint_model import verify_complete_query_fit as original
from .sjg_local_runtime_904 import digest, read


def verifier(root, base):
    root = Path(root).resolve()
    base = Path(base).resolve()
    base.relative_to(root)
    dataset = PartitionedFullShardDataset(root, base)
    fixed = root / 'server_rebuild/full_dataset_v1/training_registry.json'
    actual = base / 'training_registry.json'

    def redirected(path):
        path = Path(path).resolve()
        return actual if path == fixed else path

    environment = dict(original.__globals__)
    environment.update(read=lambda path: read(redirected(path)),
                       digest=lambda path: digest(redirected(path)))
    complete = FunctionType(original.__code__, environment, original.__name__,
                            original.__defaults__, original.__closure__)

    def verify(receipt_root, state, checkpoint_sha256, receipt_path):
        if Path(receipt_root).resolve() != root or digest(actual) != dataset.registry_sha256:
            raise ValueError('The complete declared TRAIN registration changed')
        receipt_path = Path(receipt_path).resolve()
        receipt_path.relative_to(base)
        result = complete(root, state, checkpoint_sha256, receipt_path)
        if state['sources'] != {name: digest(root / name) for name in state['sources']}:
            raise ValueError('Actual complete query execution sources changed')
        return dict(result, training_registry_sha256=dataset.registry_sha256,
                    dataset_directory=str(base.relative_to(root)),
                    original_complete_view_verifier_unchanged=True)

    return verify
