"""Pro375: frozen run375 policy plus independently supervised parent validity."""
import numpy as np
from weld_pipeline.sjg_duplicate_head_1209 import DuplicateScorer
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_local_runtime_904 import digest

SHAPES=dict(W1=(160,16),b1=(16,),W2=(16,),b2=(1,))
SCHEMA='frozen375-complete-parent-incompatibility160-16-1-v1'


class IncompatibilityScorer:
    choose_next=staticmethod(DuplicateScorer.choose_next)
    def __init__(self,frozen,frozen_sha,seed=1031):
        self.frozen=frozen;self.frozen_sha=frozen_sha;self.base_sha=frozen.base_sha;self.seed=seed
        for v in frozen.parameters.values():v.setflags(write=False)
        self.resume_rng=np.random.Generator(np.random.PCG64(seed))
        self.parameters=dict(W1=self.resume_rng.normal(0,.02,(160,16)),b1=np.zeros(16),W2=np.zeros(16),b2=np.zeros(1))
        self.initialization_rng_state=self.resume_rng.bit_generator.state

    def logits(self,u):
        assert u.ndim==2 and u.shape[1]==160 and np.isfinite(u).all()
        p=self.parameters;h=np.tanh(u@p['W1']+p['b1']);z=h@p['W2']+p['b2'][0]
        assert np.isfinite(z).all()
        return z,(u,h)

    def backward(self,cache,delta):
        u,h=cache;p=self.parameters;dh=delta[:,None]*p['W2']*(1-h*h)
        return dict(W1=u.T@dh,b1=dh.sum(0),W2=h.T@delta,b2=np.array([delta.sum()]))

    def forward(self,u,features,selected,logits):
        assert logits.shape==(len(u),) and np.isfinite(logits).all()
        scores,_=self.frozen.forward(u,features,selected);scores=scores.copy()
        scores[:-1]-=np.maximum(0.,logits)
        return scores

    def save(self,path):
        arrays=dict(self.parameters,schema=np.array(SCHEMA),seed=np.array(self.seed,dtype=np.int64),
            base_sha256=np.array(self.base_sha),frozen_residual_sha256=np.array(self.frozen.frozen_sha),
            frozen_duplicate_sha256=np.array(self.frozen_sha))
        atomic_binary_write(path,lambda stream:np.savez(stream,**arrays))

    @classmethod
    def load(cls,path,duplicate_path,residual_path,base_path):
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)==set(SHAPES)|{'schema','seed','base_sha256','frozen_residual_sha256','frozen_duplicate_sha256'}
            assert str(z['schema'])==SCHEMA and str(z['base_sha256'])==digest(base_path)
            assert str(z['frozen_residual_sha256'])==digest(residual_path) and str(z['frozen_duplicate_sha256'])==digest(duplicate_path)
            result=cls(DuplicateScorer.load(duplicate_path,residual_path,base_path),digest(duplicate_path),int(z['seed']))
            for key,shape in SHAPES.items():
                v=z[key];assert v.shape==shape and v.dtype==np.float64 and np.isfinite(v).all()
                result.parameters[key]=v.copy()
        return result


def labels_from_edges(edges,known=None):
    mask=np.ones(len(edges),bool) if known is None else np.asarray(known,bool)
    assert mask.shape==(len(edges),)
    return np.array([int(not row) if k else -1 for row,k in zip(edges,mask)],np.int8)


def candidate_loss(z,labels):
    z=np.asarray(z,np.float64);labels=np.asarray(labels,np.int8)
    assert z.shape==labels.shape and np.isfinite(z).all() and np.isin(labels,[-1,0,1]).all()
    active=[(y,np.flatnonzero(labels==y)) for y in (0,1) if np.any(labels==y)]
    delta=np.zeros_like(z);loss=0.
    for y,ids in active:
        sign=1.-2*y;v=sign*z[ids];weight=1./(len(active)*len(ids))
        loss+=weight*np.logaddexp(0.,v).sum();delta[ids]=weight*sign*np.exp(-np.logaddexp(0.,-v))
    assert np.isfinite(loss) and np.isfinite(delta).all()
    return float(loss),delta


class IncompatibilityAdam:
    def __init__(self,model):
        self.step_count=0;self.first={k:np.zeros_like(v) for k,v in model.parameters.items()}
        self.second={k:np.zeros_like(v) for k,v in model.parameters.items()}

    def update(self,model,gradients,lr=.001,weight_decay=.0001,clip=5.):
        assert set(gradients)==set(SHAPES)
        for k,v in gradients.items():assert v.shape==SHAPES[k] and np.isfinite(v).all()
        norm=float(np.sqrt(sum(np.sum(v*v) for v in gradients.values())))
        if not np.isfinite(norm) or norm<0:raise ValueError('Invalid gradient norm')
        scale=1. if norm<=clip else clip/norm;self.step_count+=1
        for k,p in model.parameters.items():
            g=gradients[k]*scale+weight_decay*p
            self.first[k]=.9*self.first[k]+.1*g;self.second[k]=.999*self.second[k]+.001*g*g
            p-=lr*(self.first[k]/(1-.9**self.step_count))/(np.sqrt(self.second[k]/(1-.999**self.step_count))+1e-8)
            assert np.isfinite(p).all()
        return dict(step=self.step_count,raw_gradient_norm=norm,clip_scale=scale)
