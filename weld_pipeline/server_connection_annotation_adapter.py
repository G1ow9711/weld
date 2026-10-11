"""Experimental pure-input binary mass adapter, preserving old relation ratios."""
import numpy as np

class AnnotationConnectionAdapter:
    classes_=np.arange(4)

    def __init__(self,original,gate):
        self.original=original
        self.gate=gate
        assert set(gate.classes_)=={0,1}
        assert original.n_features_in_==gate.n_features_in_==32
        self.n_features_in_=32

    def predict_proba(self,x):
        p=np.zeros((len(x),4),float)
        p[:,self.original.classes_.astype(int)]=self.original.predict_proba(x)
        # Arc-containing edges have no supervision in this experiment.
        line=(x[:,9]==1)&(x[:,11]==1)
        mass=p[:,1:].sum(1)
        eligible=line&(mass>0)
        if eligible.any():
            q=self.gate.predict_proba(x[eligible])[:,list(self.gate.classes_).index(1)]
            p[eligible,1:]*=(q/mass[eligible])[:,None]
            p[eligible,0]=1-q
        assert np.isfinite(p).all() and np.all(p>=0) and np.allclose(p.sum(1),1)
        return p
