"""One shared H-line event model: support and candidate-relative event centers.

Inference accepts raw millimeter XYZ and trained heads only. It does not read
reference files, source names, workpiece IDs, or prepared feature caches.
"""
import numpy as np

from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame, restore_probability_order
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates
from weld_pipeline.fresh_point_context import point_structure
from weld_pipeline.joint_relation_heatmaps import LOCAL_CHANNELS, make_graph
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline.near_event_votes import RADIUS_MM, vote_heatmaps
from weld_pipeline.small_finite_surface_events_v3 import small_surface_planes, small_face_positions
from weld_pipeline.physical_event_candidates import candidate_positions
from weld_pipeline.workpiece_relative_context import WorkpieceRelativeContext

VERSION = 'shared_forest_candidate_relative_event_centers_v1'
FEATURE_COUNT = 363
HEADS = ('foreground', 'start_support', 'end_support', 'cut_support',
         'start_center', 'end_center', 'cut_center')


def proximity_features(sequence, geometry, planes, indices=None):
    length = len(sequence['xyz']); step = float(sequence['sample_step_mm'])
    indices = np.arange(length) if indices is None else np.asarray(indices, np.int64)
    positions = indices.astype(float)*step
    original, _ = small_face_positions(sequence, geometry, planes)
    small, _ = candidate_positions(sequence, planes, finite=True)
    columns = []; counts = []
    for pool in (original, small):
        insertion = np.searchsorted(pool, positions)
        left_exists = insertion > 0; right_exists = insertion < len(pool)
        left = pool[np.clip(insertion-1, 0, len(pool)-1)]-positions
        right = pool[np.clip(insertion, 0, len(pool)-1)]-positions
        columns.extend((np.where(left_exists, np.clip(left, -128., 0.)/128., -1.),
                        np.where(right_exists, np.clip(right, 0., 128.)/128., 1.),
                        left_exists.astype(float), right_exists.astype(float)))
        count = np.searchsorted(pool, positions+16., side='right')-np.searchsorted(pool, positions-16.)
        counts.append(np.minimum(count, 16)/16.)
    columns.extend(counts)
    columns.append(positions/max((length-1)*step, 1e-9))
    return np.column_stack(columns).astype(np.float32)


def combine_features(local_and_point_context, node, event):
    local = np.asarray(local_and_point_context, np.float32)
    if local.ndim != 2 or local.shape[1] != 97 or np.asarray(node).shape != (255,) or event.shape != (len(local), 11):
        raise ValueError('Expected 97 point, 255 node and 11 geometry channels')
    result = np.c_[local, np.broadcast_to(np.asarray(node, np.float32), (len(local), 255)), event]
    if result.shape[1] != FEATURE_COUNT or not np.isfinite(result).all():
        raise ValueError('Finite shared forest features required')
    return result


def centers_to_offsets(centers, length, step):
    centers = np.asarray(centers, float)
    if centers.shape != (length, 3) or length < 2 or not np.isfinite(centers).all():
        raise ValueError('Three finite normalized event centers required')
    locations = np.arange(length, dtype=float)*step
    span = (length-1)*step
    return (centers*span-locations[:, None])/RADIUS_MM


def build_inputs(xyz_mm):
    xyz = np.asarray(xyz_mm, float)
    sequences, geometry = make_mixed_sequences(xyz)
    structural, process = point_structure(xyz, sequences)
    graph = make_graph(xyz, sequences, structural, process, FiniteContactPlates(xyz))
    context = WorkpieceRelativeContext(xyz)
    local = []; point = []; node_context = []; entries = []; row = 0
    for node, index in enumerate(graph['sequence_indices']):
        index = int(index); seq, _, transform = canonical_frame(sequences[index])
        stop = row+len(seq['xyz'])
        local.append(seq['features'][:, LOCAL_CHANNELS])
        point.append(context.describe_sequence(seq, np.arange(len(seq['xyz']))))
        mid = (seq['xyz'][0].astype(float)+seq['xyz'][-1].astype(float))/2.
        normals = seq['features'][len(seq['xyz'])//2, 60:66].reshape(2, 3)
        node_context.append(context.describe(mid, normals, seq['direction']))
        entries.append(dict(sequence=index, node=node, type=seq['type'], start=row, stop=stop, transform=transform))
        row = stop
    planes, evidence = small_surface_planes(xyz, geometry['straight_geometry'])
    return sequences, dict(local=np.concatenate(local), point_context=np.concatenate(point),
        nodes=np.c_[graph['nodes'], np.asarray(node_context, np.float32)], entries=entries,
        geometry=geometry['straight_geometry'], planes=planes, plane_evidence=evidence)


def infer(bundle, data, sequences, chunk_size=8192):
    if bundle['config']['version'] != VERSION or set(bundle['heads']) != set(HEADS):
        raise ValueError('Expected one fixed shared event-center bundle')
    probabilities = [np.zeros((len(s['xyz']), 7), np.float32) for s in sequences]
    for entry in data['entries']:
        index = entry['sequence']
        if entry['type'] != 'horizontal' or sequences[index].get('curve_kind') == 'arc':
            continue
        seq, _, transform = canonical_frame(sequences[index])
        if transform != entry['transform']:
            raise ValueError('Point-derived canonical ordering differs')
        size = len(seq['xyz']); raw = np.zeros((size, 7), float)
        for a in range(0, size, chunk_size):
            b = min(size, a+chunk_size); ids = np.arange(a, b)
            sl = slice(entry['start']+a, entry['start']+b)
            local = np.c_[data['local'][sl], data['point_context'][sl]]
            event = proximity_features(seq, data['geometry'], data['planes'], ids)
            x = combine_features(local, data['nodes'][entry['node']], event)
            for channel, key in enumerate(HEADS):
                raw[a:b, channel] = bundle['heads'][key].predict(x)
        offsets = centers_to_offsets(raw[:, 4:7], size, seq['sample_step_mm'])
        votes = vote_heatmaps(np.clip(raw[:, 0], 0., 1.), np.clip(raw[:, 1:4], 0., 1.), offsets,
                             seq['sample_step_mm'])
        probabilities[index] = restore_probability_order(votes, transform)
    return probabilities
