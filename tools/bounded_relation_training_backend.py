"""Bind exact numerical proofs to bounded reuse of original relation inputs."""
from pathlib import Path
from types import FunctionType

from tools.compact_endpoint_training_backend import configure as compact_configure
from tools.compact_endpoint_training_backend import configured_main as compact_main
from weld_pipeline.server_bounded_cached_relation_prior import BoundedCachedRun446RelationPrior
from weld_pipeline.server_bounded_lazy_parent_competition import BoundedLazyParentCompetition
from weld_pipeline.sjg_local_runtime_904 import read,digest

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['tools/bounded_relation_training_backend.py',
    'weld_pipeline/server_bounded_cached_relation_prior.py',
    'weld_pipeline/server_bounded_lazy_parent_competition.py',
    'tools/verify_bounded_cached_relation_prior.py','tools/verify_bounded_lazy_parent_competition.py']


def configure(reference_controls,synthetic_control,geometry_control,competition_control,output):
    proofs=[]
    statuses=['PASS_BOUNDED_ORIGINAL_RELATION_FEATURES_AND_RECOMPUTATION',
              'PASS_COMPLETE_ORIGINAL_GRAPH_AND_BOUNDED_FEATURE_RECOMPUTATION']
    for name,status in zip([geometry_control,competition_control],statuses):
        path=Path(name).resolve();path.relative_to(ROOT)
        value=read(path)
        if (value['status']!=status or value['reference_geometry_reads']!=0
                or value['parameter_updates']!=0 or value['checked_relations']<1
                or value['sources']!={n:digest(ROOT/n) for n in value['sources']}):
            raise ValueError('Current actual original-relation eviction proofs required')
        proofs.append((path,value))
    geometry,competition=[value for _,value in proofs]
    if (not geometry['all_original_frozen_relations_checked']
            or not geometry['all_32_features_and_prior_penalties_bitwise_equal']
            or not geometry['evicted_geometry_recomputed_bitwise_equal']
            or not geometry['new_workpiece_resets_cache']
            or not competition['complete_graph_all_pairs_bitwise_equal']
            or not competition['active_subset_full_input_graph_bitwise_equal']
            or not competition['all_32_features_and_prior_penalties_and_scales_bitwise_equal']
            or not competition['evicted_reversed_order_replay_bitwise_equal']
            or not competition['invalid_pairs_rejected']
            or geometry['pool_receipt_sha256']!=competition['pool_receipt_sha256']
            or geometry['prior_run446_sha256']!=competition['prior_run446_sha256']):
        raise ValueError('Complete matching original relation controls are required')
    loader,guards,metadata=compact_configure(reference_controls,synthetic_control,output)
    environment=dict(loader.__globals__)
    environment['LazyParentCompetition']=BoundedLazyParentCompetition
    loader=FunctionType(loader.__code__,environment,loader.__name__,loader.__defaults__,loader.__closure__)
    guards=list(dict.fromkeys([*guards,*SOURCES,
        *[n for _,value in proofs for n in value['sources']],
        *[str(path.relative_to(ROOT)) for path,_ in proofs]]))
    metadata=dict(metadata,relation_reuse_backend='BOUNDED_ORIGINAL_INPUT_GEOMETRY_AND_PAIR_FEATURES',
        relation_controls=[dict(path=str(path.relative_to(ROOT)),sha256=digest(path)) for path,_ in proofs],
        max_cached_parents=4096,max_cached_pairs=8192,
        cached_parent_eviction_removes_candidates=False,cached_pair_eviction_removes_output_edges=False,
        original_neighbor_graph_and_ties_and_penalties_retained=True)
    return loader,guards,metadata


def configured_main(original,loader,guards,metadata,entry_file,output):
    prepared=compact_main(original,loader,guards,metadata,entry_file,output)
    environment=dict(prepared.__globals__)
    environment['CachedRun446RelationPrior']=BoundedCachedRun446RelationPrior
    return FunctionType(prepared.__code__,environment,prepared.__name__,prepared.__defaults__,prepared.__closure__)


def relation_arguments(parser):
    parser.add_argument('--bounded-geometry-control',required=True)
    parser.add_argument('--bounded-competition-control',required=True)
