"""Experimental shared H-line predictor; finite XYZ and one trained checkpoint only."""
import numpy as np
import torch

from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates
from weld_pipeline.fresh_point_context import point_structure
from weld_pipeline.joint_relation_heatmaps import make_graph, LOCAL_CHANNELS
from weld_pipeline.joint_workpiece_event_votes import JointWorkpieceEventVotes, VERSION, infer_votes
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline.workpiece_relative_context import WorkpieceRelativeContext


def build_inputs(xyz):
    xyz = np.asarray(xyz, float)
    sequences, geometry = make_mixed_sequences(xyz)
    structural, process = point_structure(xyz, sequences)
    graph = make_graph(xyz, sequences, structural, process, FiniteContactPlates(xyz))
    context = WorkpieceRelativeContext(xyz)
    local = []; points = []; nodes = []; entries = []; row = 0
    for node, index in enumerate(graph['sequence_indices']):
        index = int(index); sequence, _, transform = canonical_frame(sequences[index])
        stop = row+len(sequence['xyz'])
        local.append(sequence['features'][:, LOCAL_CHANNELS])
        points.append(context.describe_sequence(sequence, np.arange(len(sequence['xyz']))))
        mid = (sequence['xyz'][0].astype(float)+sequence['xyz'][-1].astype(float))/2.
        ns = sequence['features'][len(sequence['xyz'])//2, 60:66].reshape(2, 3)
        nodes.append(context.describe(mid, ns, sequence['direction']))
        entries.append(dict(sequence=index, node=node, type=sequence['type'], start=row, stop=stop, transform=transform))
        row = stop
    arrays = dict(graph)
    arrays['nodes'] = np.c_[graph['nodes'], np.asarray(nodes, np.float32)]
    torch_graph = tuple(torch.from_numpy(arrays[k]).cuda()
                        for k in ('nodes', 'edge_index', 'edge_type', 'edge_features'))
    data = dict(features=np.concatenate(local), point_context=np.concatenate(points),
        entries=entries, horizontal=[e for e in entries if e['type'] == 'horizontal'], torch_graph=torch_graph)
    return sequences, geometry, data


def load_model(checkpoint):
    bundle = torch.load(checkpoint, map_location='cpu')
    config = bundle['config']
    if (config['version'] != VERSION or config['point_channels'] != 97
            or config['node_channels'] != 255 or bundle['epoch'] != 120):
        raise ValueError('Expected fixed 97/255 workpiece event model')
    model = JointWorkpieceEventVotes().cuda()
    model.load_state_dict(bundle['model']); model.eval()
    return model


def predict_xyz(xyz_mm, model):
    sequences, geometry, data = build_inputs(xyz_mm)
    probabilities = infer_votes(model, data, sequences)
    predictions = decode_intervals(sequences, probabilities,
        threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
    return predictions, sequences, probabilities
