"""Separate positive existence from uncertainty between positive counts.

The same fitted count probabilities are used. No probability calibration,
feature changes, additional fitted heads, or per-case thresholds are involved.
"""
import numpy as np


class HierarchicalCount:
    def __init__(self, base):
        self.base=base
        self.classes_=base.classes_.copy()

    def predict_proba(self, x):
        raw=self.base.predict_proba(x)
        positive=np.flatnonzero(self.classes_>0)
        if not len(positive):
            return raw
        out=np.zeros_like(raw)
        zero=np.flatnonzero(self.classes_==0)
        out[:,zero]=raw[:,zero]
        winner=positive[raw[:,positive].argmax(1)]
        out[np.arange(len(raw)),winner]=raw[:,positive].sum(1)
        return out
