"""Resume held feature preparation without importing training model modules.

The original CPU fragment producer requires a process without torch imports.
Only the held controller and input worker are loaded here. Their functions,
frozen point weights, registered inputs and feature calculations stay intact.
Completed stages are reused with their original bindings, and this distinct
execution context is recorded separately from the failed launcher.
"""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys

from tools import prepare_full_evaluation_case as worker
from tools import prepare_full_evaluation_inputs as controller
from weld_pipeline.server_partitioned_full_dataset import PartitionedEvaluationShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = worker.ROOT
SOURCE_NAMES = [
    'tools/prepare_lightweight_partitioned_held_inputs.py',
    'weld_pipeline/server_partitioned_full_dataset.py',
    'weld_pipeline/server_full_shard_dataset.py',
]


def local_name(value):
    if Path(value).name != value or value in ('', '.', '..'):
        raise ValueError('A local dataset or preparation directory name is required')
    return value


def write_once(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable held execution binding changed')
    else:
        write(path, value)


class HeldSubprocess:
    def __init__(self, args):
        self.args = args

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def run(self, argv, *args, **kwargs):
        argv = list(argv)
        if '-m' in argv:
            index = argv.index('-m')
            if argv[index + 1] == 'tools.prepare_full_evaluation_case':
                argv = [*argv[:index + 1], 'tools.prepare_lightweight_partitioned_held_inputs',
                        '--dataset-name', self.args.dataset_name,
                        '--preparation-name', self.args.preparation_name,
                        '--previous-preparation-name', self.args.previous_preparation_name,
                        *argv[index + 2:]]
        return subprocess.run(argv, *args, **kwargs)


def configure(args):
    base = ROOT / 'server_rebuild' / local_name(args.dataset_name)
    if args.dataset_name == 'full_dataset_v1':
        raise ValueError('A separately registered corrected partition is required')
    proof = read(base / 'registration_repair/receipt.json')
    if proof['bindings'] != {name: digest(base / name) for name in proof['bindings']}:
        raise ValueError('Corrected dataset registration changed')
    dataset = PartitionedEvaluationShardDataset(ROOT, base)
    output = base / local_name(args.preparation_name)
    previous = base / local_name(args.previous_preparation_name)
    if output == previous:
        raise ValueError('Keep the prior failed preparation context intact')
    for module in (worker, controller):
        module.BASE = base
        module.WORKSPACE = base / 'evaluation_workspace'
        module.SHARDS = base / 'evaluation_input_shards'
        module.OUT = output
        module.EvaluationShardDataset = lambda root: PartitionedEvaluationShardDataset(root, base)
    worker.core.BASE = base
    controller.SOURCE_NAMES = list(dict.fromkeys([*controller.SOURCE_NAMES, *SOURCE_NAMES]))
    controller.subprocess = HeldSubprocess(args)
    output.mkdir(exist_ok=True)
    return base, dataset, output, previous


def main(args):
    if 'torch' in sys.modules:
        raise RuntimeError('Held input launcher must not import training model modules')
    base, dataset, output, previous = configure(args)
    if args.job is not None:
        if args.job not in dataset.jobs or args.stage is None:
            raise ValueError('A registered held job and original input stage are required')
        sources = {name: digest(ROOT / name) for name in controller.SOURCE_NAMES}
        binding = dict(dataset_directory=str(base.relative_to(ROOT)), job_key=args.job,
            stage=args.stage, registry_sha256=dataset.registry_sha256,
            launcher_sources=sources, original_entry_sha256=digest(worker.__file__),
            original_input_functions_unchanged=True, torch_imported_before_stage=False,
            fit_allowed=False, optimizer_updates=0, GT_geometry_as_input=False)
        folder = output / 'execution_bindings'
        folder.mkdir(exist_ok=True)
        write_once(folder / (args.job + '_' + args.stage + '.json'), binding)
        worker.main(args.job, args.stage)
        if sources != {name: digest(ROOT / name) for name in sources}:
            raise ValueError('Held input source changed during execution')
        return
    if args.stage is not None:
        raise ValueError('An input stage requires a held job')
    # Use the old controller lock as well so two preparation contexts cannot
    # write the same owned workspace or shard directory concurrently.
    with (previous / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read(previous / 'protocol.json')
        if old['registry_sha256'] != dataset.registry_sha256 or old['fit_allowed']:
            raise ValueError('Prior held registration changed or authorized fitting')
        for name, sha in old['source_bindings'].items():
            if digest(ROOT / name) != sha:
                raise ValueError('Prior held input computation changed')
        stages = {}
        for key in dataset.jobs:
            prior = previous / key
            if not prior.is_dir():
                continue
            for marker in (prior / 'stages').glob('*.json'):
                value = read(marker)
                if (value['registry_sha256'] != dataset.registry_sha256
                        or value['source_sha256'] != digest(worker.__file__)
                        or value['fit_allowed'] or value['optimizer_updates'] != 0):
                    raise ValueError('A completed held stage changed its execution binding')
                stages[str(marker.relative_to(ROOT))] = digest(marker)
            link = output / key
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve(strict=True) != prior.resolve(strict=True):
                    raise ValueError('Unexpected held stage reuse destination')
            else:
                link.symlink_to(prior, target_is_directory=True)
        binding = dict(dataset_directory=str(base.relative_to(ROOT)),
            registry_sha256=dataset.registry_sha256,
            previous_protocol_sha256=digest(previous / 'protocol.json'),
            previous_failure_sha256=digest(previous / 'failure.json'),
            reused_stage_receipts=stages,
            launcher_sources={name: digest(ROOT / name) for name in controller.SOURCE_NAMES},
            original_input_functions_unchanged=True, original_cpu_import_check_preserved=True,
            fit_allowed=False, optimizer_updates=0, GT_geometry_as_input=False)
        resume_file = output / 'lightweight_resume_binding.json'
        if resume_file.exists():
            initial_stages = read(resume_file)['reused_stage_receipts']
            if any(stages.get(name) != sha for name, sha in initial_stages.items()):
                raise ValueError('An initially reused held stage was removed or replaced')
            binding['reused_stage_receipts'] = initial_stages
        write_once(resume_file, binding)
        controller.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--preparation-name', required=True)
    parser.add_argument('--previous-preparation-name', default='evaluation_preparation_v1')
    parser.add_argument('--job')
    parser.add_argument('--stage', choices=worker.STAGES)
    parsed = parser.parse_args()
    try:
        main(parsed)
    except BaseException:
        # Keep failures in this execution context; the original failed receipt
        # remains an immutable explanation of why a new launcher was needed.
        import time
        import traceback
        destination = ROOT / 'server_rebuild' / local_name(parsed.dataset_name) / local_name(parsed.preparation_name)
        destination.mkdir(exist_ok=True)
        write(destination / 'failure.json', dict(error=traceback.format_exc(), unix_time=time.time()))
        raise
