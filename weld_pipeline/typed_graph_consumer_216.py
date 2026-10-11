"""Typed 209 graph to the unchanged 197 numerical inputs; lazy Torch only.

This adapter does not discover a primitive, produce point features, execute a
raw head, fit a normalizer or train a model. Preparation needs only NumPy.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from .primitive_incidence_209 import (
    PrimitiveQueryGraph, PrimitiveError, ROLES, PAIR_FEATURE_NAMES,
)

VERSION = 'typed_209_complete_graph_consumer_216_v1'
EXPECTED_209_SHA = '666eceba1f6762d35ae6d872c456e43b7842dd40931e666a84d1406f3a50ee98'
EXPECTED_197_SHA = '72586ce7cd104c23fc427235c4fff766002a6b5875a31cc03b538e7b95bbedda'
NODE_GEOMETRY_NAMES = (
    'query_position_mm', 'predicted_position_mm', 'centre_minus_query_mm',
    'predicted_x_relative_to_path_origin_mm',
    'predicted_y_relative_to_path_origin_mm',
    'predicted_z_relative_to_path_origin_mm', 'support',
)


def _guard_file(name, expected):
    if sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
        raise PrimitiveError('Frozen consumer source changed: ' + name)


def _sealed(array):
    value = np.ascontiguousarray(array)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def typed_graph_fingerprint(graph):
    """Includes ALL q, raw FG, geometry, closure, provenance and pair ordinals."""
    _guard_file('primitive_incidence_209.py', EXPECTED_209_SHA)
    if not isinstance(graph, PrimitiveQueryGraph):
        raise PrimitiveError('Complete typed 209 graph required; 195 proxy refused')
    return graph.fingerprint()


@dataclass(frozen=True)
class TypedNetworkPayload:
    features425: object
    raw7: object
    roles: object
    geometry7: object
    path_span_mm: object
    path_ids: object
    graph_fingerprint: str
    context_path_ids: tuple
    relation_counts: tuple


def prepare_typed_network_payload(graph):
    fingerprint = typed_graph_fingerprint(graph)
    # Reject the whole numerical request, while the original graph/FP ledger
    # remains available. No invalid node, query or path is silently discarded.
    for context in graph.path_contexts.values():
        if context.primitive is None or context.geometry_error:
            raise PrimitiveError('INVALID_TYPED_CONTEXT: ' + context.geometry_error)
        q = context.query_positions_mm
        length = context.primitive.length_mm
        if len(q) < 2 or q[0] != 0. or q[-1] != length:
            raise PrimitiveError('INCOMPLETE_ACTUAL_Q_COVERAGE: full request refused')
    values = graph.node_tensors()
    spans = np.asarray([graph.path_contexts[n.path_id].primitive.length_mm
                        for n in graph.nodes], np.float64)
    paths = np.asarray([n.path_id for n in graph.nodes], np.int64)
    arrays = (
        np.asarray(values['node_features'], np.float32),
        np.asarray(values['raw7'], np.float32),
        np.asarray(values['roles'], np.int64),
        np.asarray(values['node_geometry'], np.float64), spans, paths,
    )
    n = len(graph.nodes)
    shapes = ((n, 425), (n, 7), (n,), (n, 7), (n,), (n,))
    for value, shape in zip(arrays, shapes):
        if value.shape != shape or not np.isfinite(value).all():
            raise PrimitiveError('Finite full typed numerical payload required')
    geometry = arrays[3]
    if not np.array_equal(geometry[:, 2], geometry[:, 1] - geometry[:, 0]):
        raise PrimitiveError('Typed centre minus actual q differs')
    if len(paths) and (np.any(paths < 0) or np.any(arrays[2] < 0) or np.any(arrays[2] > 2)):
        raise PrimitiveError('Typed path/role ordinal invalid')
    return TypedNetworkPayload(*(_sealed(a) for a in arrays), fingerprint,
        tuple(sorted(graph.path_contexts)),
        tuple((kind, graph.pair_count(kind)) for kind in ('child', 'message', 'duplicate')))


def payload_receipt(payload):
    if not isinstance(payload, TypedNetworkPayload):
        raise TypeError('TypedNetworkPayload required')
    hashes = {}
    for key in ('features425', 'raw7', 'roles', 'geometry7', 'path_span_mm', 'path_ids'):
        value = getattr(payload, key)
        hashes[key] = sha256(str(value.dtype).encode() + str(value.shape).encode()
                             + value.tobytes()).hexdigest()
    return dict(version=VERSION, graph_fingerprint=payload.graph_fingerprint,
        nodes=len(payload.roles), complete_context_path_ids=list(payload.context_path_ids),
        complete_relation_counts=dict(payload.relation_counts), array_sha256=hashes,
        span_definition='actual_typed_primitive_length_mm',
        no_coordinate_projection_clipping_or_candidate_pruning=True,
        actual_PCD_provider_verified=False, actual_197_forward_backward='NOT_RUN',
        raw7_head_training='UNTRAINED', full_contract_pass=False)


def verify_payload_binding(graph, payload):
    if not isinstance(payload, TypedNetworkPayload):
        raise TypeError('TypedNetworkPayload required')
    expected = prepare_typed_network_payload(graph)
    if (payload.graph_fingerprint != expected.graph_fingerprint or
            payload.context_path_ids != expected.context_path_ids or
            payload.relation_counts != expected.relation_counts):
        raise PrimitiveError('Typed payload graph or complete path binding differs')
    for key in ('features425', 'raw7', 'roles', 'geometry7', 'path_span_mm', 'path_ids'):
        a, b = np.asarray(getattr(payload, key)), getattr(expected, key)
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            raise PrimitiveError('Typed payload differs from graph: ' + key)
    return True


def bind_unchanged_197_inputs(graph, payload=None):
    """Real Torch conversion, deferred until the main solver releases memory.

    Caller supplies a separately verified normalizer/model for the exact
    feature scope. No checkpoint or normalizer is inferred by this adapter.
    """
    payload = prepare_typed_network_payload(graph) if payload is None else payload
    verify_payload_binding(graph, payload)
    _guard_file('boundary_incidence_net_197.py', EXPECTED_197_SHA)
    import torch
    from .boundary_incidence_net_197 import (
        NodeInputs, CompletePathRelations,
        NODE_GEOMETRY_NAMES as native_geometry, PAIR_FEATURE_NAMES as native_pairs,
        ROLE_NAMES,
    )
    if (tuple(native_geometry) != NODE_GEOMETRY_NAMES or
            tuple(native_pairs) != PAIR_FEATURE_NAMES or tuple(ROLE_NAMES) != ROLES):
        raise PrimitiveError('Frozen 197 numerical schema differs')
    inputs = NodeInputs(
        torch.from_numpy(np.array(payload.features425, copy=True)),
        torch.from_numpy(np.array(payload.raw7, copy=True)),
        torch.from_numpy(np.array(payload.roles, copy=True)),
        torch.from_numpy(np.array(payload.geometry7, copy=True)),
        torch.from_numpy(np.array(payload.path_span_mm, copy=True)),
    )
    relations = CompletePathRelations(payload.path_ids, payload.roles,
                                     graph.pair_features, payload.graph_fingerprint)
    for kind, count in payload.relation_counts:
        if relations.pair_count(kind) != count:
            raise PrimitiveError('Typed complete relation count differs from 197')
    return inputs, relations
