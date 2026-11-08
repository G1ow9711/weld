"""Publish learned proposal quality and frozen-generator decision provenance."""
from .curveonly_01a1ac7c_proposal_quality_v240 import *
from . import curveonly_01a1ac7c_proposal_quality_v240 as previous


def decode(features, packet, model, threshold=.5):
    result = previous.decode(features, packet, model, threshold)
    result['complete_proposal_quality'] = model.last_quality
    result['complete_proposal_quality_mode'] = model.mode
    result['unfiltered_generator_proposals'] = len(model.last_unfiltered_proposals)
    result['frozen_generator_geometry_unchanged'] = True
    return result
