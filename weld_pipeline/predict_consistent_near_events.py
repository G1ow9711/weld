"""Pure XYZ development inference for the fixed consistency/mining trials."""
import torch

from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.joint_workpiece_event_votes import JointWorkpieceEventVotes
from weld_pipeline.near_event_votes import VERSION, infer_votes
from weld_pipeline.predict_joint_workpiece_events import build_inputs

VARIANTS = ('center_consistency', 'hard_background', 'combined')


def load_model(checkpoint):
    state = torch.load(checkpoint, map_location='cpu'); cfg = state['config']
    if (cfg['version'] != VERSION or cfg['trial_number'] != 160 or cfg['radius_mm'] != 16.
            or cfg['point_channels'] != 97 or cfg['node_channels'] != 255
            or state['epoch'] != 120 or state['variant'] not in VARIANTS):
        raise ValueError('Expected frozen epoch120 trial160 checkpoint')
    model = JointWorkpieceEventVotes().cuda()
    model.load_state_dict(state['model']); model.eval()
    return model


def predict_xyz(xyz_mm, model):
    sequences, _, data = build_inputs(xyz_mm)
    probabilities = infer_votes(model, data, sequences)
    predictions = decode_intervals(sequences, probabilities,
        threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
    return predictions, sequences, probabilities
