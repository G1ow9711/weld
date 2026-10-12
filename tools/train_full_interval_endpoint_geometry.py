"""Extend the unchanged complete metric trainer with the tested endpoint head.

The source-bound AST edits are deliberately limited and counted: declare the
new model, inherit the actual old warm model, bind the new controls, and name
the new parameterization. Every complete-data cache, update, optimizer, resume,
mask, source, role and finish check executes from the original trainer.
"""
import argparse
import ast
import inspect
from pathlib import Path

from tools import train_full_local_query_geometry_metric as original
from weld_pipeline.server_interval_endpoint_checkpoint import (
    COMPLETE_STATUS, verify_complete_interval_endpoint_fit)
from weld_pipeline.server_interval_endpoint_residual_model import IntervalEndpointResidualModel, ARCHITECTURE
from weld_pipeline.server_local_query_weld_model import ARCHITECTURE as WARM_ARCHITECTURE
from weld_pipeline.server_local_query_checkpoint import verify_complete_local_query_fit
from weld_pipeline.sjg_local_runtime_904 import digest, read


ROOT = Path(__file__).resolve().parents[1]
SOURCES = list(dict.fromkeys([*original.SOURCES,
    'tools/train_full_interval_endpoint_geometry.py',
    'weld_pipeline/server_interval_endpoint_checkpoint.py',
    'weld_pipeline/server_interval_endpoint_residual_model.py',
    'tools/compare_interval_endpoint_residual.py']))


def require_endpoint_controls(base, args):
    jobs = set(); versions = set()
    for value in args.endpoint_control:
        path = (ROOT / value).resolve(strict=True); path.relative_to(base)
        proof = read(path); protocol = read(path.parent / 'protocol.json')
        if (proof['status'] != 'ACTUAL_COMPLETE_TRAIN_VIEW_ENDPOINT_RESIDUAL_COMPARISON_NOT_PROMOTED'
                or proof['protocol_sha256'] != digest(path.parent / 'protocol.json')
                or protocol['sources'] != {n:digest(ROOT/n) for n in protocol['sources']}
                or protocol['warm_checkpoint_sha256'] != args.warm_sha256
                or not proof['all_initial_named_outputs_and_child_losses_bitwise_equal']
                or not proof['all_frozen_original_parameters_bitwise_preserved']
                or not proof['old_model_loader_rejected_partial_architecture']
                or not proof['endpoint_head_initial_gradient_sum'] > 0
                or proof['held_reference_reads'] != 0 or proof['production_optimizer_updates'] != 0
                or proof['actual_updates_per_arm'] < 64):
            raise ValueError('Current actual complete-view endpoint controls are required')
        jobs.add(protocol['job_key']); versions.add(protocol['version'])
        for label in ('original', 'endpoint_residual'):
            result = read(path.parent / (label + '_result.json'))
            model = (ROOT / result['checkpoint_path']).resolve(strict=True)
            model.relative_to(path.parent)
            if (digest(model) != result['checkpoint_sha256']
                    or result['after']['totals'] != proof['after'][label]
                    or len(result['trace']) != proof['actual_updates_per_arm']):
                raise ValueError('The real fitted control result or saved model changed')
    if len(jobs) < 2 or len(versions) < 2:
        raise ValueError('Controls must cover at least two separate actual TRAIN inputs')


def verified_query(root, base, state, sha256, receipt):
    if state.get('architecture') == WARM_ARCHITECTURE:
        return verify_complete_local_query_fit(root, base, state, sha256, receipt)
    return verify_complete_interval_endpoint_fit(root, base, state, sha256, receipt)


class EndpointTrainerChanges(ast.NodeTransformer):
    def __init__(self):
        self.counts = dict(warm_architecture=0, initialization=0, settings=0)

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.left, ast.Subscript)
                and isinstance(node.left.value, ast.Name) and node.left.value.id == 'initial'
                and isinstance(node.left.slice, ast.Constant) and node.left.slice.value == 'architecture'
                and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Name)
                and node.comparators[0].id == 'ARCHITECTURE'):
            node.comparators[0] = ast.Constant(WARM_ARCHITECTURE)
            self.counts['warm_architecture'] += 1
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'model' and node.func.attr == 'load_state_dict'
                and node.args and isinstance(node.args[0], ast.Subscript)
                and isinstance(node.args[0].value, ast.Name) and node.args[0].value.id == 'initial'):
            node.func.attr = 'inherit_complete_local'; node.keywords = []
            self.counts['initialization'] += 1
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'settings' and node.func.attr == 'update'):
            for keyword in node.keywords:
                if keyword.arg == 'stage':
                    keyword.value = ast.Constant('COMPLETE_TRAIN_LOCAL_XYZ_METRE_ENDPOINT_RESIDUAL')
                if keyword.arg == 'metric_warmstart_control_receipt_sha256':
                    keyword.arg = 'endpoint_control_receipt_sha256'
                    keyword.value = ast.parse('[digest(ROOT / p) for p in args.endpoint_control]', mode='eval').body
            node.keywords.extend(ast.parse(
                "f(architecture=ARCHITECTURE, endpoint_offset_unit='metres', "
                "endpoint_projection_backward='IDENTITY_ON_INPUT_QUERY_BOUND_PROJECTION', "
                "endpoint_residual_source_sha256=digest(ROOT / 'weld_pipeline/server_interval_endpoint_residual_model.py'))",
                mode='eval').body.keywords)
            self.counts['settings'] += 1
        return node


def configured_main():
    tree = ast.parse(inspect.getsource(original.main))
    function = tree.body[0]
    start = next(i for i,node in enumerate(function.body) if isinstance(node, ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == 'metric_control_folder' for t in node.targets))
    end = next(i for i,node in enumerate(function.body[start:], start) if isinstance(node, ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == 'out' for t in node.targets))
    if end-start != 4:
        raise ValueError('Original warm-control structure changed; inspect before extending')
    function.body[start:end] = ast.parse('require_endpoint_controls(base, args)').body
    changes = EndpointTrainerChanges(); tree = changes.visit(tree)
    if changes.counts != dict(warm_architecture=1, initialization=1, settings=1):
        raise ValueError('Original complete trainer structure changed unexpectedly')
    ast.fix_missing_locations(tree)
    environment = dict(original.main.__globals__, SOURCES=SOURCES,
        ARCHITECTURE=ARCHITECTURE, COMPLETE_STATUS=COMPLETE_STATUS,
        LocalQueryWeldModel=IntervalEndpointResidualModel,
        require_endpoint_controls=require_endpoint_controls,
        verify_complete_local_query_fit=verified_query)
    exec(compile(tree, __file__, 'exec'), environment)
    return environment['main'], changes.counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--warm-checkpoint', required=True); parser.add_argument('--warm-sha256', required=True)
    parser.add_argument('--endpoint-control', required=True, action='append')
    parser.add_argument('--epochs', type=int, default=3); parser.add_argument('--steps-per-view', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1031); parser.add_argument('--max-cache-gib', type=float, default=16.)
    parser.add_argument('--control-run', action='store_true')
    main, _ = configured_main()
    main(parser.parse_args())
