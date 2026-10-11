"""SHA-bound runtime graph artifacts and separate original-lineage labels.

The runtime loader reads three runtime files only. It cannot read GT or choose
queries by labels. Complete canonical paths, including empty ones, are retained.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from weld_pipeline.boundary_query_graph_195 import BoundaryProposal, PathContext, QueryGraph, ROLES
from weld_pipeline.structured_graph_solver_v1 import GraphTruth, TruthEvent, TruthChild

VERSION = "sha_bound_runtime_and_offline_artifacts_200_v1"


class ArtifactError(ValueError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(directory, source, suffix, expected):
    if not isinstance(source, str) or not source or Path(source).name != source or any(c in source for c in "/\\:"):
        raise ArtifactError("single source basename required")
    path = Path(directory)/(""+source+suffix)
    wanted = expected.get(path.name)
    if wanted is None or file_sha256(path) != wanted:
        raise ArtifactError("Missing bound SHA or changed artifact: "+path.name)
    return path


def load_runtime_graph(directory, source, expected_sha256):
    """Reads graph JSON, full runtime nodes NPZ and full FG context NPZ only."""
    meta_path = _file(directory, source, "_candidate_graph.json", expected_sha256)
    node_path = _file(directory, source, "_all_runtime_nodes.npz", expected_sha256)
    context_path = _file(directory, source, "_runtime_FG_context.npz", expected_sha256)
    metadata = json.loads(meta_path.read_text("utf8"))
    if metadata["all_runtime_nodes_sha256"] != file_sha256(node_path) or metadata["runtime_FG_context_sha256"] != file_sha256(context_path):
        raise ArtifactError("runtime payloads differ from original195 candidate JSON")
    with np.load(node_path, allow_pickle=False) as archive:
        required = ("runtime_ids", "features425", "raw7", "roles", "query_position_mm", "predicted_position_mm", "predicted_xyz_mm", "support")
        arrays = {key: archive[key] for key in required}
    identity = arrays["runtime_ids"]
    n = len(identity)
    shapes = dict(runtime_ids=(n, 3), features425=(n, 425), raw7=(n, 7), roles=(n,), query_position_mm=(n,),
                  predicted_position_mm=(n,), predicted_xyz_mm=(n, 3), support=(n,))
    for key, shape in shapes.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ArtifactError("runtime artifact shape/finiteness mismatch: "+key)
    if identity.dtype != np.int64 or arrays["roles"].dtype != np.int64:
        raise ArtifactError("exact int64 runtime identities/roles required")
    if arrays["features425"].dtype != np.float32 or arrays["raw7"].dtype != np.float32:
        raise ArtifactError("exact frozen float32 features/raw required")
    contexts = []
    with np.load(context_path, allow_pickle=False) as archive:
        for key, value in sorted(metadata["paths"].items(), key=lambda pair: int(pair[0])):
            path = int(key)
            count = int(value["query_count"])
            prefix = archive["fg_prefix_"+key]
            square = archive["fg_square_prefix_"+key]
            if prefix.shape != (count+1,) or square.shape != prefix.shape or not np.isfinite(prefix).all() or not np.isfinite(square).all():
                raise ArtifactError("full frozen FG context shape/finiteness mismatch")
            contexts.append(PathContext(path, count, float(value["step_mm"]), tuple(value["origin_mm"]), tuple(value["direction"]),
                                        value["raw_sha256"], prefix, square))
    context_by_id = {context.path_id: context for context in contexts}
    nodes = []
    for i, (path, query, role) in enumerate(identity):
        if path not in context_by_id or role not in range(3) or query < 0 or query >= context_by_id[path].query_count:
            raise ArtifactError("runtime ID absent from complete path context")
        if arrays["roles"][i] != role or arrays["query_position_mm"][i] != query*context_by_id[path].step_mm:
            raise ArtifactError("runtime role/query position altered")
        nodes.append(BoundaryProposal(tuple(identity[i]), int(path), ROLES[role], float(arrays["query_position_mm"][i]),
            float(arrays["predicted_position_mm"][i]), arrays["predicted_xyz_mm"][i], float(arrays["support"][i]),
            arrays["raw7"][i], arrays["features425"][i]))
    admission = {int(key): value for key, value in metadata.get("admission_receipts", {}).items()}
    graph = QueryGraph(nodes, contexts, admission)
    if len(graph.nodes) != n or graph.explicit_duplicate_records_removed or graph.complete_hashes() != metadata["hashes"]:
        raise ArtifactError("full195 graph identity/payload/relationship hashes changed")
    return graph


def load_training_truth(directory, source, graph, expected_sha256):
    """Train-only original195 event lineage; never used by runtime construction."""
    array_path = _file(directory, source, "_offline_labels.npz", expected_sha256)
    lineage_path = _file(directory, source, "_offline_lineage.json", expected_sha256)
    with np.load(array_path, allow_pickle=False) as archive:
        events = archive["event_index"]
    metadata = json.loads(lineage_path.read_text("utf8"))
    if metadata["event_index_sha256"] != file_sha256(array_path):
        raise ArtifactError("original195 lineage label archive SHA mismatch")
    if events.dtype != np.int64 or events.shape != (len(graph.nodes),):
        raise ArtifactError("original full195 node-event labels misaligned")
    event_table = tuple(TruthEvent(int(e["id"]), e["role"], int(e["parent_id"]), int(e["boundary_order"])) for e in metadata["events"])
    children = tuple(TruthChild(int(c["id"]), int(c["start_event"]), int(c["end_event"])) for c in metadata["true_children"])
    return GraphTruth(events, event_table, children)


def complete_path_views(graph, truth=None):
    """All canonical paths, including empty ones; preserve local exact lineage.

200's fixed complete-path loss cannot independently split one GT identity over
paths. That future case is refused, although199 full graph solver supports it.
The frozen195 set must establish no such coupling before this runner is used.
"""
    event_paths = {}
    if truth is not None:
        indexes = np.asarray(truth.node_event_index)
        if indexes.shape != (len(graph.nodes),):
            raise ArtifactError("full node-event index required before path reduction")
        if np.any(indexes == -2):
            raise NotImplementedError("PARTIAL_UNKNOWN_MODE_NOT_IMPLEMENTED")
        for event in truth.events:
            paths = {graph.nodes[i].path_id for i in np.flatnonzero(indexes == event.id)}
            if len(paths) != 1:
                raise NotImplementedError("CROSSPATH_OR_MISSING_GT_EVENT_PATH_REDUCTION_NOT_IMPLEMENTED")
            event_paths[event.id] = next(iter(paths))
        if any(event_paths[c.start_event] != event_paths[c.end_event] for c in truth.true_children):
            raise NotImplementedError("CROSSPATH_GT_CHILD_PATH_REDUCTION_NOT_IMPLEMENTED")
    output = []
    for path, context in sorted(graph.path_contexts.items()):
        ids = graph.paths[path]
        local_graph = QueryGraph([graph.nodes[i] for i in ids], [context], {path: graph.admission_receipts.get(path, {})})
        local_truth = None
        if truth is not None:
            table = tuple(e for e in truth.events if event_paths[e.id] == path)
            children = tuple(c for c in truth.true_children if event_paths[c.start_event] == path)
            local_truth = GraphTruth(np.asarray(truth.node_event_index)[ids].copy(), table, children)
        output.append((int(path), local_graph, local_truth, ids.copy()))
    if truth is not None:
        if sum(len(t.events) for _, _, t, _ in output) != len(truth.events) or sum(len(t.true_children) for _, _, t, _ in output) != len(truth.true_children):
            raise ArtifactError("path reduction dropped full truth denominator")
    if sum(len(g.nodes) for _, g, _, _ in output) != len(graph.nodes):
        raise ArtifactError("path reduction dropped runtime candidates")
    return tuple(output)
