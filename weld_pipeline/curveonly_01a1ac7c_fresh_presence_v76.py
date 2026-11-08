"""An independently fitted source-presence task before conditional count mode.

The decision masses expose P(source present) at the old conditional count mode.
They are not posterior probabilities over the individual count classes.
"""
import numpy as np
from . import curveonly_01a1ac7c_graph_context_v65 as graph

SEMANTICS='FRESH_BINARY_SOURCE_PRESENCE_MASS_AT_OLD_CONDITIONAL_COUNT_MODE_NOT_COUNT_POSTERIOR'


class FreshPresenceCount:
    def __init__(self,count,presence):
        self.raw_count=count;self.presence=presence;self.classes_=count.classes_.copy()
        self.n_features_in_=count.n_features_in_
        assert self.n_features_in_==presence.n_features_in_
        assert set(presence.classes_)=={0,1} and 0 in self.classes_ and np.any(self.classes_>0)
        self.decision_mass_semantics=SEMANTICS

    def predict_proba(self,x):
        raw=self.raw_count.predict_proba(x);binary=self.presence.predict_proba(x)
        positive=np.flatnonzero(self.classes_>0);zero=int(np.flatnonzero(self.classes_==0)[0])
        probability=binary[:,int(np.flatnonzero(self.presence.classes_==1)[0])]
        winner=positive[raw[:,positive].argmax(1)];out=np.zeros_like(raw)
        out[:,zero]=1.-probability;out[np.arange(len(raw)),winner]=probability
        return out


class FreshPresenceCurveSet(graph.GraphCurveSet):
    def __init__(self,base,presence):
        super().__init__(FreshPresenceCount(base.count,presence),base.interval,base.style)

    def predict(self,features,packet):
        out=super().predict(features,packet)
        if self.last_timing is not None:
            self.last_timing.update(count_decision_mass_semantics=SEMANTICS,
                source_presence_fitted_separately=True,conditional_count_unchanged=True)
        return out
