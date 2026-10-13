"""Append authenticated visual curves without inventing legacy feature rows."""
import numpy as np
from .sjg_curve_queries_907 import CurveCarrier


class VisualAugmentedInput:
    def __init__(self, original, addition):
        if addition['original_profile_count'] != len(original.profiles):
            raise ValueError('Visual additions bind a different original profile count')
        self.original = original
        self.original_profile_count = len(original.profiles)
        self.profiles = list(original.profiles) + addition['additional_profiles']
        self.lengths = np.r_[original.lengths, [CurveCarrier(p['representation']).length
            for p in addition['additional_profiles']]]
        if not np.isfinite(self.lengths).all():
            raise ValueError('Nonfinite visual finite input')

    def curve(self, pid):
        return CurveCarrier(self.profiles[int(pid)]['representation'])

    def features_at(self, pid, positions_mm):
        if int(pid) >= self.original_profile_count:
            raise ValueError('Visual curves require genuine raw/visual features, not legacy 457 columns')
        return self.original.features_at(pid, positions_mm)

    def tokens(self, pid):
        if int(pid) >= self.original_profile_count:
            raise ValueError('Visual curves require their own event encoder, not fabricated legacy tokens')
        return self.original.tokens(pid)
