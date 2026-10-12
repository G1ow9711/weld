"""Run preserved complete parent/held stages with explicit local-XYZ query context.

All original code files stay unchanged. An isolated process namespace replaces
only the query factory/scorer and adds their source bindings. Original complete
reference matching, set optimization, role boundaries and export gates remain.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from types import FunctionType

from tools import prepare_full_predicted_parent_pool as pool_module
from tools.qualify_full_dataset_labels import ply_points
from tools.run_partitioned_full_dataset_stage import (
    ALLOWED_ENTRY, configure as original_configure, entry_body)
from weld_pipeline.server_local_query_checkpoint import load_local_query_model, verify_complete_local_query_fit
from weld_pipeline.server_local_query_runtime import LocalCompleteQueryProposals
from weld_pipeline.server_local_query_weld_model import ARCHITECTURE
from weld_pipeline.sjg_local_runtime_904 import digest, read, write


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/run_local_full_dataset_stage.py',
    'weld_pipeline/server_local_query_checkpoint.py', 'weld_pipeline/server_local_query_runtime.py',
    'weld_pipeline/server_local_query_weld_model.py', 'weld_pipeline/server_query_point_neighborhood.py',
    'weld_pipeline/sjg_global_neighborhood_sequence_1239.py', 'tools/qualify_full_dataset_labels.py']


def child_arguments(argv, dataset_name):
    argv = list(argv)
    if '-m' not in argv:
        return argv
    position = argv.index('-m')
    if position + 1 >= len(argv) or argv[position + 1] not in ALLOWED_ENTRY:
        return argv
    return [*argv[:position+1], 'tools.run_local_full_dataset_stage',
            '--dataset-name', dataset_name, '--module', argv[position+1], '--', *argv[position+2:]]


class LocalSubprocess:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, argv, *args, **kwargs):
        return subprocess.Popen(child_arguments(argv, self.dataset_name), *args, **kwargs)

    def run(self, argv, *args, **kwargs):
        return subprocess.run(child_arguments(argv, self.dataset_name), *args, **kwargs)


def configure(dataset_name):
    base, train, held, modules = original_configure(dataset_name)
    original_prepare = pool_module.prepare

    def verify_query(root, state, sha256, receipt):
        return verify_complete_local_query_fit(root, base, state, sha256, receipt)

    def prepare(dataset, job_key, model, state, priors, output, query_mode,
                fixed_query_fractions=(.125, .375, .625, .875), competition_mode='eager', resource_mode='fixed_v1'):
        # The same ordinary workpiece XYZ declared in the role registration is
        # the only extra input. This path never opens a reference or label.
        job = dataset.jobs[job_key]
        path = ROOT.parent / 'data/G_Data' / job['workpiece_source']
        if digest(path) != job['workpiece_sha256']:
            raise ValueError('Actual complete XYZ input changed before local inference')
        xyz = ply_points(path)
        if len(xyz) != job['original_points']:
            raise ValueError('Complete raw point count changed before local inference')
        factory = lambda m,mean,std,**kwargs:LocalCompleteQueryProposals(m,mean,std,xyz,**kwargs)
        namespace = dict(original_prepare.__globals__, CompleteQueryProposals=factory)
        actual = FunctionType(original_prepare.__code__, namespace, original_prepare.__name__,
                              original_prepare.__defaults__, original_prepare.__closure__)
        receipt = actual(dataset, job_key, model, state, priors, output, query_mode,
                         fixed_query_fractions, competition_mode, resource_mode)
        binding = Path(output) / 'pools' / job_key / 'local_raw_xyz_binding.json'
        value = dict(raw_xyz_sha256=job['workpiece_sha256'], complete_raw_points=len(xyz),
            raw_xyz_context_columns=72, architecture=ARCHITECTURE,
            source_bindings={name:digest(ROOT / name) for name in SOURCES},
            reference_geometry_reads=0, source_identity_feature_columns=0,
            original_pool_receipt_sha256=digest(binding.parent / 'receipt.json'))
        if binding.exists() and read(binding) != value:
            raise ValueError('Frozen raw query context binding differs')
        if not binding.exists():
            write(binding, value)
        return receipt

    bridge = LocalSubprocess(dataset_name)
    for module in modules.values():
        if hasattr(module, 'load_query_model'):
            module.load_query_model = load_local_query_model
        if hasattr(module, 'verify_complete_query_fit'):
            module.verify_complete_query_fit = verify_query
        if hasattr(module, 'ARCHITECTURE'):
            module.ARCHITECTURE = ARCHITECTURE
        if getattr(module, 'prepare', None) is original_prepare:
            module.prepare = prepare
        if hasattr(module, 'subprocess'):
            module.subprocess = bridge
        for field in ('SOURCES', 'SOURCE_NAMES'):
            if hasattr(module, field):
                setattr(module, field, list(dict.fromkeys([*getattr(module, field), *SOURCES])))
    return base, train, held, modules


def main(args):
    if args.module not in ALLOWED_ENTRY:
        raise ValueError('Only preserved complete pipeline stages may use the explicit local query context')
    base, train, held, modules = configure(args.dataset_name)
    module = modules[args.module]
    source = Path(module.__file__)
    tree, redirected = entry_body(source.read_text(), args.dataset_name)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if '--allow-partial-query' in arguments or '--allow-partial-model' in arguments:
        raise ValueError('Partial local-query controls cannot enter complete model evaluation')
    identity = hashlib.sha256((args.module + '\0' + '\0'.join(arguments)).encode()).hexdigest()[:20]
    binding = base / 'local_stage_execution_bindings' / (identity + '.json')
    binding.parent.mkdir(exist_ok=True)
    value = dict(module=args.module, arguments=arguments, dataset_name=args.dataset_name,
        original_entry_sha256=digest(source), local_sources={name:digest(ROOT / name) for name in SOURCES},
        redirected_output_literals=redirected, architecture=ARCHITECTURE,
        original_scoring_training_and_role_gates_preserved=True)
    if binding.exists() and read(binding) != value:
        raise ValueError('Explicit local execution binding changed')
    if not binding.exists():
        write(binding, value)
    namespace = module.__dict__
    original_argv = sys.argv
    try:
        sys.argv = [str(source), *arguments]
        exec(compile(tree, str(source), 'exec'), namespace, namespace)
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    main(parser.parse_args())
