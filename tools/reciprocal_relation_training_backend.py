"""Admit exact reciprocal distance reuse to the complete TRAIN backend."""
from pathlib import Path
from types import FunctionType

from tools.bounded_relation_training_admission import configure as bounded_configure
from tools.bounded_relation_training_backend import configured_main as bounded_main
from weld_pipeline.server_reciprocal_cached_relation_prior import ReciprocalCachedRun446RelationPrior
from weld_pipeline.sjg_local_runtime_904 import read, digest

ROOT = Path(__file__).resolve().parents[1]


def configure_numerical(reference_controls, synthetic_control, geometry_control,
                        competition_control, bounded_fit_control, reciprocal_control, output):
    path = Path(reciprocal_control).resolve()
    path.relative_to(ROOT)
    proof = read(path)
    required = ['all_original_frozen_relations_checked',
                'all_32_features_and_prior_penalties_bitwise_equal',
                'original_directed_arithmetic_and_later_mutated_context_checked',
                'reversed_pair_replay_bitwise_equal', 'exception_scope_reset_checked',
                'cache_does_not_outlive_one_pair',
                'original_candidates_graph_samples_and_features_unchanged']
    if (proof['status'] != 'PASS_RECIPROCAL_ORIGINAL_ALL_RELATIONS'
            or not all(proof[name] for name in required)
            or proof['checked_relations'] < 1 or proof['reference_geometry_reads'] != 0
            or proof['optimizer_updates'] != 0
            or proof['primitive_distance_computations'] != proof['reused_primitive_distances']
            or proof['sources'] != {name: digest(ROOT / name) for name in proof['sources']}):
        raise ValueError('Current complete original-relation reciprocal proof required')
    loader, guards, metadata = bounded_configure(reference_controls, synthetic_control,
        geometry_control, competition_control, bounded_fit_control, output)
    if proof['prior_run446_sha256'] != read(Path(geometry_control))['prior_run446_sha256']:
        raise ValueError('Reciprocal proof uses a different original relation prior')
    guards = list(dict.fromkeys([*guards, *proof['sources'], str(path.relative_to(ROOT)),
        'tools/reciprocal_relation_training_backend.py']))
    metadata = dict(metadata, reciprocal_distance_reuse='ONE_IMMUTABLE_ORIGINAL_PAIR_CONTEXT',
        reciprocal_control_path=str(path.relative_to(ROOT)),
        reciprocal_control_sha256=digest(path), original_directed_pair_arithmetic_retained=True,
        reciprocal_cache_eviction_removes_candidates=False)
    return loader, guards, metadata


def configure(reference_controls, synthetic_control, geometry_control, competition_control,
              bounded_fit_control, reciprocal_control, reciprocal_fit_control, output):
    path = Path(reciprocal_fit_control).resolve()
    path.relative_to(ROOT)
    proof = read(path)
    required = ['all_input_tokens_and_costs_and_views_bitwise_equal',
                'all_eight_histories_equal', 'final_parameter_and_optimizer_tensors_bitwise_equal',
                'final_unary_penalties_and_selection_bitwise_equal',
                'forged_reciprocal_source_binding_rejected', 'actual_reciprocal_distances_reused']
    if (proof['status'] != 'PASS_RECIPROCAL_COMPLETE_JOB_AND_ACTUAL_EIGHT_STEP_FIT'
            or not all(proof[name] for name in required)
            or proof['actual_optimizer_updates_per_model'] != 8
            or proof['held_reference_reads'] != 0
            or proof['sources'] != {name: digest(ROOT / name) for name in proof['sources']}):
        raise ValueError('Current complete-job reciprocal optimizer proof required')
    loader, guards, metadata = configure_numerical(reference_controls, synthetic_control,
        geometry_control, competition_control, bounded_fit_control, reciprocal_control, output)
    if proof['backend'] != metadata:
        raise ValueError('Reciprocal optimizer proof belongs to another backend')
    guards = list(dict.fromkeys([*guards, *proof['sources'], str(path.relative_to(ROOT))]))
    metadata = dict(metadata, reciprocal_fit_control_path=str(path.relative_to(ROOT)),
        reciprocal_fit_control_sha256=digest(path), reciprocal_optimizer_parity_updates=8)
    return loader, guards, metadata


def configured_main(original, loader, guards, metadata, entry_file, output):
    prepared = bounded_main(original, loader, guards, metadata, entry_file, output)
    environment = dict(prepared.__globals__)
    environment['CachedRun446RelationPrior'] = ReciprocalCachedRun446RelationPrior
    return FunctionType(prepared.__code__, environment, prepared.__name__,
                        prepared.__defaults__, prepared.__closure__)


def reciprocal_arguments(parser, require_fit=True):
    parser.add_argument('--reciprocal-control', required=True)
    if require_fit:
        parser.add_argument('--reciprocal-training-control', required=True)
