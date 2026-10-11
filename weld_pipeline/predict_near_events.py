"""Experimental H-line event model, raw XYZ only, independent of old predictors."""
import torch

from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.joint_workpiece_event_votes import JointWorkpieceEventVotes
from weld_pipeline.near_event_votes import VERSION, infer_votes
from weld_pipeline.predict_joint_workpiece_events import build_inputs


def load_model(checkpoint):
    bundle = torch.load(checkpoint, map_location='cpu'); cfg = bundle['config']
    if (cfg['version'] != VERSION or cfg['radius_mm'] != 16.
            or cfg['point_channels'] != 97 or cfg['node_channels'] != 255
            or bundle['epoch'] != 120 or bundle['variant'] not in ('physical_matched', 'millimeter_loss')):
        raise ValueError('Fixed epoch120 16mm near-event checkpoint required')
    model = JointWorkpieceEventVotes().cuda()
    model.load_state_dict(bundle['model']); model.eval()
    return model


def predict_xyz(xyz_mm, model):
    sequences, _, data = build_inputs(xyz_mm)
    probabilities = infer_votes(model, data, sequences)
    predictions = decode_intervals(sequences, probabilities,
        threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
    return predictions, sequences, probabilities
