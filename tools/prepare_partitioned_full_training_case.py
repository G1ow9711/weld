"""Run the unchanged complete XYZ feature pipeline in a declared partition.

This isolated worker changes only registration and owned storage locations.
Original point weights, all geometry stages, candidates and input values remain
unchanged. The original producer hash and this execution context are recorded.
"""
import argparse
from pathlib import Path

from tools import prepare_full_training_case as original
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = original.ROOT
SOURCES = ['tools/prepare_partitioned_full_training_case.py',
    'tools/prepare_full_training_case.py', 'weld_pipeline/server_partitioned_full_dataset.py',
    'tools/rebuild_expansion_input_blocks.py', 'tools/infer_legacy_point_features.py',
    'tools/infer_run26_point_features.py', 'tools/rebuild_legacy_rank2_candidates.py',
    'tools/rebuild_legacy_downstream.py', 'weld_pipeline/server_fixed_fragment_queries.py',
    *['tools/' + module + '.py' for module, _ in original.GEOMETRY.values()]]


def configure(dataset_name):
    if Path(dataset_name).name != dataset_name or dataset_name in ('', '.', '..', 'full_dataset_v1'):
        raise ValueError('A separate complete dataset partition is required')
    base = ROOT / 'server_rebuild' / dataset_name
    proof = read(base / 'registration_repair/receipt.json')
    if proof['bindings'] != {name: digest(base / name) for name in proof['bindings']}:
        raise ValueError('Corrected registration changed')
    dataset = PartitionedFullShardDataset(ROOT, base)
    original.BASE = base
    original.WORKSPACE = base / 'training_workspace'
    original.SHARDS = dataset.shards
    return dataset


def main(args):
    dataset = configure(args.dataset_name)
    if args.job not in dataset.jobs:
        raise ValueError('Only a registered TRAIN input may enter this worker')
    job = original.registered_job(args.job)
    if job != dataset.jobs[args.job]:
        raise ValueError('Preserved producer and corrected reader registration differ')
    workspace = original.WORKSPACE
    target = workspace.resolve(strict=True)
    if not workspace.is_symlink() or target.parent != Path('/tmp') or not target.name.startswith('weld-full-training-'):
        raise ValueError('An authenticated task-owned temporary workspace is required')
    sources = {name: digest(ROOT / name) for name in SOURCES}
    out = dataset.base / 'preparation' / args.job
    out.mkdir(parents=True, exist_ok=True)
    binding = dict(dataset_directory=str(dataset.base.relative_to(ROOT)), job_key=args.job,
        registry_sha256=dataset.registry_sha256, registration_sha256=digest(
            dataset.base / 'registration_repair/receipt.json'), sources=sources,
        original_feature_producer_sha256=digest(original.__file__),
        input_algorithm_or_frozen_point_weights_modified=False,
        complete_original_stages=list(original.STAGES), GT_geometry_as_input=False,
        held_labels_admitted=False, optimizer_updates=0)
    path = out / 'partitioned_worker_binding.json'
    if path.exists() and read(path) != binding:
        raise ValueError('Immutable worker execution context changed')
    if not path.exists():
        write(path, binding)
    original.main(args.job, args.stage)
    if sources != {name: digest(ROOT / name) for name in sources}:
        raise ValueError('Complete input worker source changed during execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--stage', required=True, choices=original.STAGES)
    main(parser.parse_args())
