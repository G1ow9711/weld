"""Pure-input finite-interval context for connection inference."""
import numpy as np
from weld_pipeline.sjg_joint_link_runtime_1107 import node_geometry
from weld_pipeline.server_roundoff_gap_links import guarded_relative_link_blocks


def node_context(nodes,graph):
    ends=np.asarray([node_geometry(n,graph)[0] for n in nodes])
    length=np.asarray([abs(n['hi']-n['lo']) for n in nodes])
    carrier=np.asarray([graph.curve(graph.profile(n['fragment'])).length for n in nodes])
    assert np.all(length>0) and np.all(carrier>0)
    ratios=np.asarray([[n['lo'],n['hi']] for n in nodes])/carrier[:,None]
    weld=np.asarray([n['weld_type'] for n in nodes])
    return ends,length,ratios,weld


def interval_features(a,b,ends,length,ratios,weld):
    distances=np.linalg.norm(ends[a,:,None,:]-ends[b,None,:,:],axis=3).reshape(-1,4)
    ratio=np.minimum(length[a],length[b])/np.maximum(length[a],length[b])
    return np.column_stack([length[a],length[b],ratios[a],ratios[b],distances,ratio,weld[a]==weld[b]]).astype(np.float32)


def context_link_blocks(nodes,case,gap_resolver=None):
    context=node_context(nodes,case['graph'])
    for links,x,legal in guarded_relative_link_blocks(nodes,case,gap_resolver):
        extra=interval_features(links[:,0],links[:,1],*context)
        yield links,np.concatenate([x,extra],axis=1),legal


class ContextConnectionAdapter:
    classes_=np.arange(4)
    n_features_in_=44

    def __init__(self,original,gate):
        assert original.n_features_in_==32 and gate.n_features_in_==44
        assert set(gate.classes_)=={0,1}
        self.original=original;self.gate=gate

    def predict_proba(self,x):
        assert x.shape[1]==44
        p=np.zeros((len(x),4),float)
        p[:,self.original.classes_.astype(int)]=self.original.predict_proba(x[:,:32])
        mass=p[:,1:].sum(1);eligible=(x[:,9]==1)&(x[:,11]==1)&(mass>0)
        if eligible.any():
            q=self.gate.predict_proba(x[eligible])[:,list(self.gate.classes_).index(1)]
            p[eligible,1:]*=(q/mass[eligible])[:,None];p[eligible,0]=1-q
        assert np.isfinite(p).all() and np.all(p>=0) and np.allclose(p.sum(1),1)
        return p
