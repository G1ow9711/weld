"""Point-derived relative workpiece context added to the unchanged event-vote model."""
import numpy as np
import torch
from torch import nn

from weld_pipeline.canonical_horizontal_patch_heatmaps import restore_probability_order
from weld_pipeline.joint_event_votes import JointEventVotes, vote_heatmaps

VERSION = 'joint_workpiece_event_votes_v1'
LOCAL_COUNT = 97
NODE_COUNT = 255


class JointWorkpieceEventVotes(JointEventVotes):
    def __init__(self):
        super().__init__()
        original_point = self.point_encoder[0]
        original_node = self.node_encoder[0]
        point = nn.Conv1d(LOCAL_COUNT, self.width, 1)
        node = nn.Linear(NODE_COUNT, self.width)
        with torch.no_grad():
            # Existing weights match the old from-scratch seed exactly. Added columns start neutral.
            point.weight.zero_(); point.weight[:, :75].copy_(original_point.weight)
            point.bias.copy_(original_point.bias)
            node.weight.zero_(); node.weight[:, :233].copy_(original_node.weight)
            node.bias.copy_(original_node.bias)
        self.point_encoder[0] = point; self.node_encoder[0] = node


@torch.no_grad()
def infer_votes(model, data, sequences, chunk_size=1024, halo=64, batch_size=8):
    model.eval(); device = next(model.parameters()).device
    context = model.encode_graph(*data['torch_graph'])
    raw = {e['sequence']:np.zeros((e['stop']-e['start'], 7), np.float32) for e in data['horizontal']}
    blocks = []
    for entry in data['horizontal']:
        length = entry['stop']-entry['start']
        for start in range(0, length, chunk_size):
            blocks.append((entry, start, min(length, start+chunk_size)))
    for begin in range(0, len(blocks), batch_size):
        items = blocks[begin:begin+batch_size]; size = max(b-a+2*halo for _, a, b in items)
        x = np.zeros((len(items), LOCAL_COUNT, size), np.float32); indices = []
        for bi, (entry, a, b) in enumerate(items):
            length = entry['stop']-entry['start']; lo = a-halo
            left, right = max(0, lo), min(length, b+halo)
            sl = slice(entry['start']+left, entry['start']+right)
            x[bi, :75, left-lo:right-lo] = data['features'][sl].T
            x[bi, 75:, left-lo:right-lo] = data['point_context'][sl].T
            indices.append(entry['node'])
        prediction = model.point_forward(torch.from_numpy(x).to(device),
            context[torch.tensor(indices, device=device)]).cpu().numpy()
        for bi, (entry, a, b) in enumerate(items):
            raw[entry['sequence']][a:b] = prediction[bi, :, halo:halo+b-a].T
    result = [np.zeros((len(s['xyz']), 7), np.float32) for s in sequences]
    for entry in data['horizontal']:
        index = entry['sequence']; r = raw[index]
        sigmoid = lambda v:1./(1.+np.exp(-np.clip(v, -60., 60.)))
        p = vote_heatmaps(sigmoid(r[:, 0]), sigmoid(r[:, 1:4]), r[:, 4:7], sequences[index]['sample_step_mm'])
        result[index] = restore_probability_order(p, entry['transform'])
    return result
