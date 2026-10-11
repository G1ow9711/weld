"""Execute preserved complete training/evaluation stages in a declared partition.

This is an isolated process context, not a replacement loss or model. Original
functions run unchanged with partition readers and storage globals. The original
CLI body is executed with its sole fixed output-directory literal redirected;
child modules inherit this explicit context through the same launcher.
"""
import argparse
import ast
import importlib
from pathlib import Path
import subprocess
import sys

from weld_pipeline.server_partitioned_full_dataset import (
    PartitionedFullShardDataset, PartitionedEvaluationShardDataset)
from weld_pipeline.server_partitioned_query_fit import verifier
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_SOURCES = ['tools/run_partitioned_full_dataset_stage.py',
    'weld_pipeline/server_partitioned_query_fit.py',
    'weld_pipeline/server_partitioned_full_dataset.py']
MODULES = ['tools.prepare_full_training_case', 'tools.prepare_full_evaluation_case',
    'tools.prepare_full_evaluation_inputs', 'tools.prepare_full_predicted_parent_pool',
    'tools.prepare_full_recall_parent_pool', 'tools.train_full_predicted_parent_selector',
    'tools.train_full_recall_parent_dataset', 'tools.continue_full_recall_training',
    'tools.continue_reciprocal_full_recall_training', 'tools.train_reciprocal_full_recall_parent_dataset',
    'tools.evaluate_full_weld_predictions', 'tools.assess_full_weld_output_fields',
    'tools.continue_validation_fixed_test', 'tools.export_full_weld_predictions',
    'tools.continue_full_output_export']
ALLOWED_ENTRY = set(MODULES) - {'tools.prepare_full_training_case',
    'tools.prepare_full_predicted_parent_pool', 'tools.prepare_full_recall_parent_pool',
    'tools.train_full_predicted_parent_selector', 'tools.train_full_recall_parent_dataset'}


def child_arguments(argv, dataset_name):
    argv = list(argv)
    if '-m' not in argv:
        return argv
    position = argv.index('-m')
    if position + 1 >= len(argv) or argv[position + 1] not in ALLOWED_ENTRY:
        return argv
    return [*argv[:position + 1], 'tools.run_partitioned_full_dataset_stage',
            '--dataset-name', dataset_name, '--module', argv[position + 1], '--', *argv[position + 2:]]


class PartitionSubprocess:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, argv, *args, **kwargs):
        return subprocess.Popen(child_arguments(argv, self.dataset_name), *args, **kwargs)

    def run(self, argv, *args, **kwargs):
        return subprocess.run(child_arguments(argv, self.dataset_name), *args, **kwargs)


def configure(dataset_name):
    if Path(dataset_name).name != dataset_name or dataset_name in ('', '.', '..', 'full_dataset_v1'):
        raise ValueError('A separate authenticated complete partition is required')
    base = ROOT / 'server_rebuild' / dataset_name
    proof = read(base / 'registration_repair/receipt.json')
    if proof['bindings'] != {name: digest(base / name) for name in proof['bindings']}:
        raise ValueError('Corrected complete registration changed')
    train = PartitionedFullShardDataset(ROOT, base)
    held = PartitionedEvaluationShardDataset(ROOT, base)
    verify = verifier(ROOT, base)
    modules = {name: importlib.import_module(name) for name in MODULES}
    bridge = PartitionSubprocess(dataset_name)
    for module in modules.values():
        if hasattr(module, 'BASE'):
            module.BASE = base
        if hasattr(module, 'FullShardDataset'):
            module.FullShardDataset = lambda value: PartitionedFullShardDataset(value, base)
        if hasattr(module, 'EvaluationShardDataset'):
            module.EvaluationShardDataset = lambda value: PartitionedEvaluationShardDataset(value, base)
        if hasattr(module, 'verify_complete_query_fit'):
            module.verify_complete_query_fit = verify
        if hasattr(module, 'subprocess'):
            module.subprocess = bridge
        for field in ('SOURCES', 'SOURCE_NAMES'):
            if hasattr(module, field):
                setattr(module, field, list(dict.fromkeys([*getattr(module, field), *CONTEXT_SOURCES])))
    core = modules['tools.prepare_full_training_case']
    core.WORKSPACE, core.SHARDS = base / 'training_workspace', base / 'input_shards'
    for name in ('tools.prepare_full_evaluation_case', 'tools.prepare_full_evaluation_inputs'):
        module = modules[name]
        module.WORKSPACE = base / 'evaluation_workspace'
        module.SHARDS = base / 'evaluation_input_shards'
        module.OUT = base / 'evaluation_preparation_v1'
    return base, train, held, modules


class EntryPartition(ast.NodeTransformer):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.redirected_output_literals = 0

    def visit_Constant(self, node):
        if node.value == 'server_rebuild/full_dataset_v1':
            self.redirected_output_literals += 1
            return ast.copy_location(ast.Constant(value='server_rebuild/' + self.dataset_name), node)
        return node


def entry_body(source, dataset_name):
    tree = ast.parse(source)
    expected = ast.dump(ast.parse("__name__ == '__main__'", mode='eval').body)
    blocks = [node for node in tree.body if isinstance(node, ast.If) and ast.dump(node.test) == expected]
    if len(blocks) != 1 or blocks[0].orelse:
        raise ValueError('The preserved module needs exactly one original CLI entry block')
    entry = ast.Module(body=blocks[0].body, type_ignores=[])
    redirect = EntryPartition(dataset_name)
    result = ast.fix_missing_locations(redirect.visit(entry))
    return result, redirect.redirected_output_literals


def main(args):
    if args.module not in ALLOWED_ENTRY:
        raise ValueError('Only declared complete pipeline entry points can be executed')
    base, train, held, modules = configure(args.dataset_name)
    module = modules[args.module]
    source = Path(module.__file__)
    tree, changed = entry_body(source.read_text(), args.dataset_name)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if '--allow-partial-query' in arguments or '--allow-partial-model' in arguments:
        raise ValueError('Partial controls cannot enter the complete repaired pipeline')
    identity = args.module.rsplit('.', 1)[1]
    for flag in ('--experiment-name', '--job', '--stage'):
        if flag in arguments:
            value = arguments[arguments.index(flag) + 1]
            if Path(value).name != value or value in ('', '.', '..'):
                raise ValueError('Execution binding identities must be local names')
            identity += '_' + value
    folder = base / 'stage_execution_bindings'
    folder.mkdir(exist_ok=True)
    path = folder / (identity + '.json')
    binding = dict(module=args.module, arguments=arguments, dataset_directory=str(base.relative_to(ROOT)),
        registration_proof_sha256=digest(base / 'registration_repair/receipt.json'),
        training_registry_sha256=train.registry_sha256, evaluation_registry_sha256=held.registry_sha256,
        original_entry_sha256=digest(source), sources={name: digest(ROOT / name) for name in CONTEXT_SOURCES},
        original_cli_entry_ast=ast.dump(tree, include_attributes=False),
        only_fixed_cli_output_directory_literals_redirected=changed,
        original_complete_functions_losses_optimizer_and_selection_rules_unchanged=True,
        child_processes_receive_explicit_partition=True, partial_fit_authorization=False)
    if path.exists() and read(path) != binding:
        raise ValueError('Immutable stage execution context changed')
    if not path.exists():
        write(path, binding)
    sys.argv = [str(source), *arguments]
    exec(compile(tree, str(source), 'exec'), module.__dict__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    main(parser.parse_args())
