"""Run the preserved complete local-XYZ pipeline with verified TRAIN disk reuse.

Only the TRAIN loader's unchanged original relation values are reused. Query
fitting, reference utility, optimizer, all held roles, final model freezes and
exports retain their original implementations and complete requirements.
"""
import argparse
from pathlib import Path
import subprocess
from types import FunctionType

from tools import run_local_full_dataset_stage as original
from tools import reciprocal_relation_training_backend as backend
from weld_pipeline.server_disk_cached_parent_competition import DiskCachedParentCompetition
from weld_pipeline.sjg_local_runtime_904 import digest, read

ROOT = Path(__file__).resolve().parents[1]
NUMERICAL = ROOT / 'server_rebuild/full_dataset_v2/disk_parent_relation_numerical_control_20261012_v2/receipt.json'
FIT = ROOT / 'server_rebuild/full_dataset_v1/disk_relation_actual_fit_controls_20261012_v1/receipt.json'
SOURCES = list(dict.fromkeys([*original.SOURCES,
    'tools/run_disk_cached_local_full_dataset_stage.py',
    'weld_pipeline/server_disk_cached_parent_competition.py',
    'tools/verify_disk_cached_parent_competition.py',
    'tools/verify_disk_relation_training_backend.py']))


def controls():
    numerical, fit = read(NUMERICAL), read(FIT)
    protocol_path = NUMERICAL.parent / 'protocol.json'
    numerical_protocol = read(protocol_path)
    required = ['all_original_graph_pairs_and_active_subgraphs_identical',
        'all_32_original_features_and_prior_penalties_bitwise_identical',
        'all_reversed_replay_rows_identical_without_recomputation',
        'returned_array_mutation_does_not_change_cached_values', 'changed_frozen_input_rejected',
        'invalid_pairs_rejected', 'bounded_storage_overflow_retains_every_requested_edge',
        'private_storage_removed_on_close']
    actual = ['all_input_tokens_and_costs_and_views_bitwise_equal', 'all_eight_histories_equal',
        'final_parameter_and_optimizer_tensors_bitwise_equal',
        'final_unary_penalties_and_selection_bitwise_equal', 'actual_disk_relations_reused']
    if (numerical['status'] != 'PASS_ALL_ORIGINAL_FROZEN_RELATIONS_AND_COMPLETE_DISK_REPLAY'
            or not all(numerical[k] for k in required) or numerical['optimizer_updates'] != 0
            or numerical['reference_geometry_reads'] != 0
            or numerical_protocol['sources'] != {n:digest(ROOT / n) for n in numerical_protocol['sources']}
            or fit['status'] != 'PASS_DISK_REUSE_COMPLETE_JOB_AND_ACTUAL_EIGHT_STEP_FIT'
            or not all(fit[k] for k in actual) or fit['actual_optimizer_updates_per_model'] != 8
            or fit['held_reference_reads'] != 0 or fit['disk_cache_metrics']['disk_hits'] < 1
            or fit['sources'] != {n:digest(ROOT / n) for n in fit['sources']}):
        raise ValueError('Current complete original-value and actual optimizer disk-reuse proofs are required')
    bound = list(dict.fromkeys([*SOURCES, *numerical_protocol['sources'], *fit['sources'],
        str(NUMERICAL.relative_to(ROOT)), str(protocol_path.relative_to(ROOT)), str(FIT.relative_to(ROOT))]))
    return fit, bound


class DiskSubprocess:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def arguments(self, argv):
        result = original.child_arguments(argv, self.dataset_name)
        if '-m' in result:
            index = result.index('-m') + 1
            if result[index] == 'tools.run_local_full_dataset_stage':
                result[index] = 'tools.run_disk_cached_local_full_dataset_stage'
        return result

    def Popen(self, argv, *args, **kwargs):
        return subprocess.Popen(self.arguments(argv), *args, **kwargs)

    def run(self, argv, *args, **kwargs):
        return subprocess.run(self.arguments(argv), *args, **kwargs)


def configure(dataset_name):
    fit, bound = controls()
    base, train, held, modules = original.configure(dataset_name)
    admitted = backend.configure

    def disk_configure(*args, **kwargs):
        loader, guards, metadata = admitted(*args, **kwargs)
        reference = dict(metadata)
        for key in ('reciprocal_fit_control_path', 'reciprocal_fit_control_sha256',
                    'reciprocal_optimizer_parity_updates'):
            reference.pop(key, None)
        if reference != fit['backend']:
            raise ValueError('Disk optimizer controls belong to a different complete TRAIN recipe')
        environment = dict(loader.__globals__, LazyParentCompetition=DiskCachedParentCompetition)
        loader = FunctionType(loader.__code__, environment, loader.__name__,
                              loader.__defaults__, loader.__closure__)
        guards = list(dict.fromkeys([*guards, *bound]))
        metadata = dict(metadata, disk_relation_cache_scope='ONE_FROZEN_TRAIN_INPUT_ONLY',
            disk_relation_control_sha256=digest(NUMERICAL), disk_actual_fit_control_sha256=digest(FIT),
            max_disk_relation_cache_bytes=2 * 1024**3,
            disk_cache_overflow_removes_candidates=False, model_quality_and_timing_not_claimed=True)
        return loader, guards, metadata

    bridge = DiskSubprocess(dataset_name)
    for module in modules.values():
        if getattr(module, 'configure', None) is admitted:
            module.configure = disk_configure
        if hasattr(module, 'subprocess'):
            module.subprocess = bridge
        for field in ('SOURCES', 'SOURCE_NAMES'):
            if hasattr(module, field):
                setattr(module, field, list(dict.fromkeys([*getattr(module, field), *bound])))
    return base, train, held, modules


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    namespace = dict(original.main.__globals__, configure=configure, SOURCES=SOURCES)
    execute = FunctionType(original.main.__code__, namespace, original.main.__name__,
                           original.main.__defaults__, original.main.__closure__)
    execute(args)
