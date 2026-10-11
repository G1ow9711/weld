"""Complete, alias-balanced supervision of the frozen197 structural scores.

Training labels are separate from runtime inputs. Importing this module never
imports Torch; the differentiable consumer imports it only after release checks.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .structural_score_targets_232 import (
    LOSS_KIND, VERSION as TARGET_VERSION, TrainingRelationLayout, array_sha, immutable,
)

VERSION = 'COMPLETE_ALIAS_BALANCED_STRUCTURAL_LOSS_232_v1'


def file_sha(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class SourceTargets:
    source: str
    runtime_ids: np.ndarray
    nodes: np.ndarray
    children: np.ndarray
    duplicates: np.ndarray
    manifest: dict
    manifest_sha256: str
    target_archive_sha256: str


def load_source_targets(spec, evidence):
    """Verify immutable manifests, original files and every complete target mask."""
    if evidence['original_spec'] != spec or evidence['source'] != spec['source']:
        raise ValueError('Frozen original source and target evidence differ')
    manifest_artifact, target_artifact = evidence['manifest_artifact'], evidence['target_array_artifact']
    for artifact in (manifest_artifact, target_artifact):
        if file_sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Changed immutable target artifact')
    manifest = json.loads(Path(manifest_artifact['path']).read_text('utf-8-sig'))
    if (manifest['version'] != TARGET_VERSION or manifest['loss_kind'] != LOSS_KIND
            or manifest['source'] != spec['source'] or manifest['target_artifact'] != target_artifact
            or manifest['full_ordered_paths'] != evidence['full_ordered_path_ids']
            or len(manifest['per_path']) != spec['path_count']
            or manifest['source_path_denominator'] != spec['path_count']
            or manifest['label_binding']['source_payloads_sha256'] != spec['expected_sha256']):
        raise ValueError('Original complete source manifest/mask binding changed')
    if tuple(p['path_id'] for p in manifest['per_path']) != tuple(manifest['full_ordered_paths']):
        raise ValueError('Complete ordered path manifest required')
    for name, expected in spec['expected_sha256'].items():
        if file_sha(Path(spec['artifact_directory']) / name) != expected:
            raise ValueError('Changed original source artifact: ' + name)
    with np.load(Path(spec['artifact_directory']) / (spec['source'] + '_all_runtime_nodes.npz'),
                 allow_pickle=False) as archive:
        ids = archive['runtime_ids']
    with np.load(target_artifact['path'], allow_pickle=False) as archive:
        values = tuple(archive[key] for key in ('node_targets', 'child_targets', 'duplicate_targets'))
    layout = TrainingRelationLayout(ids, manifest['full_ordered_paths'])
    sizes = (len(ids), layout.pair_count('child'), layout.pair_count('duplicate'))
    for kind, value, size in zip(('nodes', 'children', 'duplicates'), values, sizes):
        if (value.dtype != np.int64 or value.shape != (size,) or np.any(value < -2)
                or array_sha(value) != manifest['masks_sha256'][kind]):
            raise ValueError('Complete original target vector differs: ' + kind)
    if manifest['truth_children'] != spec['denominator']:
        raise ValueError('Original truth denominator changed')
    return SourceTargets(spec['source'], immutable(ids), *(immutable(v) for v in values),
                         manifest, manifest_artifact['sha256'], target_artifact['sha256'])


def balanced_vector(groups, positive_ids, *, positive_target, negative_target,
                    negative_mass, permitted_empty_positive=()):
    """Every positive group has mass one; all known negatives share frozen mass."""
    groups = np.asarray(groups)
    if groups.dtype != np.int64 or groups.ndim != 1 or np.any(groups < -2):
        raise ValueError('Exact known/unknown int64 group identities required')
    positive_ids = tuple(positive_ids)
    if tuple(sorted(set(positive_ids))) != positive_ids or any(v < 0 for v in positive_ids):
        raise ValueError('Unique ordered true groups required')
    observed, counts = np.unique(groups[groups >= 0], return_counts=True)
    table = {int(k): int(n) for k, n in zip(observed, counts)}
    missing = set(positive_ids) - set(table)
    if set(table) - set(positive_ids) or missing - set(permitted_empty_positive):
        raise ValueError('Missing/extra true group in complete loss pool')
    weights = np.zeros(len(groups), np.float64)
    positive = groups >= 0
    if len(observed):
        weights[positive] = 1.0 / counts[np.searchsorted(observed, groups[positive])]
    negative = groups == -1
    negative_count = int(np.sum(negative))
    if negative_count:
        weights[negative] = float(negative_mass) / negative_count
    targets = np.zeros(len(groups), np.float64)
    targets[positive] = positive_target
    targets[negative] = negative_target
    audit = dict(complete_pool_count=len(groups), positive_count=int(np.sum(positive)),
        known_negative_count=negative_count, unknown_count=int(np.sum(groups == -2)),
        positive_group_counts=table, positive_groups_with_no_pairs=sorted(missing),
        positive_target=float(positive_target), negative_target=float(negative_target),
        known_negative_total_weight=float(negative_mass) if negative_count else 0.0,
        empty_negative_reason=None if negative_count else 'NO_KNOWN_NEGATIVE_IN_COMPLETE_POOL',
        empty_positive_reason=None if table else 'NO_POSITIVE_RELATION_IN_COMPLETE_POOL',
        weights_sha256=array_sha(weights), targets_sha256=array_sha(targets))
    return immutable(weights), immutable(targets), audit


@dataclass(frozen=True)
class PathTargets:
    path_id: int
    runtime_ids: np.ndarray
    original_ordinals: np.ndarray
    weights: tuple
    targets: tuple
    scope: dict
    graph_fingerprint: str = None
    pair_hashes: dict = None


def source_path_targets(source, path_id):
    try:
        meta = next(v for v in source.manifest['per_path'] if v['path_id'] == path_id)
    except StopIteration as error:
        raise ValueError('Path absent from frozen source manifest') from error
    ordinals = np.flatnonzero(source.runtime_ids[:, 0] == path_id).astype(np.int64)
    ids = source.runtime_ids[ordinals]
    a0, d0 = meta['child_offset'], meta['duplicate_offset']
    groups = (source.nodes[ordinals],
              source.children[a0:a0 + meta['child_count']],
              source.duplicates[d0:d0 + meta['duplicate_count']])
    if (array_sha(ordinals) != meta['original_global_node_ordinals_sha256']
            or array_sha(ids) != meta['literal_runtime_ids_sha256']):
        raise ValueError('Literal runtime IDs or original global ordinals changed')
    for name, value in zip(('nodes', 'children', 'duplicates'), groups):
        if array_sha(value) != meta['masks_sha256'][name]:
            raise ValueError('Per-path complete mask changed')
    events, children = tuple(meta['event_ids']), tuple(meta['child_ids'])
    E, C = len(events), len(children)
    if meta['path_units_denominator'] != max(1, E + C):
        raise ValueError('Per-path event/child denominator changed')
    vectors = (
        balanced_vector(groups[0], events, positive_target=1., negative_target=-1., negative_mass=max(1, E)),
        balanced_vector(groups[1], children, positive_target=1., negative_target=-1., negative_mass=max(1, C)),
        balanced_vector(groups[2], events, positive_target=3., negative_target=0., negative_mass=max(1, E),
                        permitted_empty_positive=meta['events_without_duplicate_pairs']),
    )
    analytic = not len(ids) and not E and not C
    if analytic != meta['trivial_no_candidates_and_no_GT']:
        raise ValueError('Only a proved no-candidate/no-GT path can be analytic zero')
    scope = dict(loss_kind=LOSS_KIND, source=source.source, path_id=int(path_id),
        manifest_sha256=source.manifest_sha256, target_archive_sha256=source.target_archive_sha256,
        original_global_node_ordinals_sha256=array_sha(ordinals), literal_runtime_ids_sha256=array_sha(ids),
        masks_sha256=meta['masks_sha256'], path_units_denominator=max(1, E + C),
        source_path_denominator=source.manifest['source_path_denominator'],
        counts=dict(nodes=len(ids), children=len(groups[1]), duplicates=len(groups[2]),
                    messages=len(ids) * (len(ids) - 1)),
        truth_events=E, truth_children=C, analytic_empty=analytic,
        vectors={name: values[2] for name, values in zip(('u', 'a', 'd'), vectors)},
        source_denominator_applied_here=False, unknown_never_automatically_negative=True)
    scope['target_scope_sha256'] = object_sha(scope)
    return PathTargets(int(path_id), immutable(ids), immutable(ordinals),
                       tuple(v[0] for v in vectors), tuple(v[1] for v in vectors), scope)


def bind_runtime_graph(targets, graph, original_ordinals):
    """Bind current runtime-only payload and complete local pair ordinals."""
    from .boundary_incidence_net_197 import query_graph_195_fingerprint
    ids = np.asarray([node.runtime_id for node in graph.nodes], dtype=np.int64).reshape(-1, 3)
    if (array_sha(ids) != targets.scope['literal_runtime_ids_sha256']
            or array_sha(np.asarray(original_ordinals)) != targets.scope['original_global_node_ordinals_sha256']
            or tuple(sorted(graph.path_contexts)) != (targets.path_id,)):
        raise ValueError('Actual path graph replaced/omitted the frozen path')
    layout = TrainingRelationLayout(ids, (targets.path_id,))
    pair_hashes = {}
    for kind, count_key in (('child', 'children'), ('duplicate', 'duplicates'), ('message', 'messages')):
        count = layout.pair_count(kind)
        if count != graph.pair_count(kind) or count != targets.scope['counts'][count_key]:
            raise ValueError('Actual complete relation pool changed')
        # A streaming digest uses the same dtype/shape prefix as array_sha.
        digest = sha256(); digest.update(b'int64'); digest.update(str((count, 2)).encode())
        observed = 0
        for pairs in layout.iter_pairs(kind):
            digest.update(pairs.tobytes()); observed += len(pairs)
        if observed != count:
            raise ValueError('Incomplete actual relation enumeration')
        pair_hashes[kind] = digest.hexdigest()
    fingerprint = query_graph_195_fingerprint(graph)
    return PathTargets(targets.path_id, targets.runtime_ids, targets.original_ordinals,
                       targets.weights, targets.targets, targets.scope, fingerprint, pair_hashes)


def numpy_path_loss(targets, scores):
    """Independent no-autograd audit of the exact registered numerical formula."""
    parts = {}
    for name, weights, truth in zip(('u', 'a', 'd'), targets.weights, targets.targets):
        value = np.asarray(scores[name], dtype=np.float64)
        if value.shape != weights.shape or not np.isfinite(value).all():
            raise ValueError('Complete finite ordinal scores required')
        parts[name] = float(np.sum(weights * np.square(value - truth), dtype=np.float64))
    return sum(parts.values()) / targets.scope['path_units_denominator'], parts


def supervised_path_loss(scores, targets):
    """Float64 balanced MSE; source denominator belongs only to the transaction."""
    import torch
    from .boundary_incidence_net_197 import _validate_scores
    _validate_scores(scores)
    if not targets.graph_fingerprint or scores.graph_fingerprint != targets.graph_fingerprint:
        raise ValueError('Current complete graph/score binding required')
    counts = targets.scope['counts']
    coverage = scores.coverage
    if (not coverage or coverage.get('complete_relations') is not True or coverage.get('truncated') is not False
            or coverage.get('nodes') != counts['nodes'] or coverage.get('child') != counts['children']
            or coverage.get('duplicate') != counts['duplicates']
            or coverage.get('message') != [counts['messages'], counts['messages']]):
        raise ValueError('Both full message passes and all child/duplicate scores required')
    for kind, name in (('child', 'child_pairs'), ('duplicate', 'duplicate_pairs')):
        if array_sha(getattr(scores, name).detach().cpu().numpy()) != targets.pair_hashes[kind]:
            raise ValueError('Scores/pair ordinal order differs from current supervision')
    if tuple(len(getattr(scores, name)) for name in ('u', 'a', 'd')) != tuple(len(w) for w in targets.weights):
        raise ValueError('Complete score and target vectors required')
    devices = {str(getattr(scores, name).device) for name in ('u', 'a', 'd')}
    if len(devices) != 1:
        raise ValueError('One actual score device required')
    loss = None; parts = {}
    for name, weights, truth in zip(('u', 'a', 'd'), targets.weights, targets.targets):
        value = getattr(scores, name).to(torch.float64)
        weight = torch.tensor(np.array(weights, copy=True), dtype=torch.float64, device=value.device)
        target = torch.tensor(np.array(truth, copy=True), dtype=torch.float64, device=value.device)
        term = (weight * (value - target).square()).sum()
        parts[name] = float(term.detach().cpu())
        loss = term if loss is None else loss + term
    loss = loss / targets.scope['path_units_denominator']
    value = float(loss.detach().cpu())
    if not np.isfinite(value) or value < 0 or not loss.requires_grad:
        raise ValueError('Finite nonnegative differentiable complete-path loss required')
    detail = dict(version=VERSION, **targets.scope, graph_fingerprint=targets.graph_fingerprint,
                  local_pair_hashes=targets.pair_hashes, loss_parts=parts, path_loss=value,
                  actual_neural_forward=True, training_native_solver_calls=0)
    return loss, detail


def analytic_empty_path(targets):
    if not targets.graph_fingerprint or not targets.scope['analytic_empty']:
        raise ValueError('Current runtime-bound no-candidate/no-GT proof required')
    return None, dict(version=VERSION, **targets.scope, graph_fingerprint=targets.graph_fingerprint,
        local_pair_hashes=targets.pair_hashes, loss_parts=dict(u=0., a=0., d=0.), path_loss=0.,
        actual_neural_forward=False, analytic_proof_kind='EMPTY_REGISTERED_SCORE_SUPERVISION',
        training_native_solver_calls=0)
