"""Cheap qualified-STEP count view; other heads retain all observations."""
import numpy as np
from .curveonly_01a1ac7c_set_v18 import PROFILE

MASK = np.asarray(sorted(set(list(range(19,31))+list(range(53,65))+
    [68+9*j+i for j in range(13) for i,k in enumerate(PROFILE) if k in (19,20,29,30)])),int)
assert len(MASK)==76


def view(rows):
    x=np.asarray(rows);out=x.copy()
    qualified=(x[:,31]>=1.-1e-6)&(x[:,32]>=1.-1e-6)
    out[np.ix_(np.flatnonzero(qualified),MASK)]=0.
    return out


class QualifiedCount:
    def __init__(self,estimator):self.estimator=estimator

    @property
    def classes_(self):return self.estimator.classes_

    def predict_proba(self,rows):return self.estimator.predict_proba(view(rows))
