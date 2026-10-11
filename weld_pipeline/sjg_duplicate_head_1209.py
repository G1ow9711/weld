"""Frozen run373 policy plus a symmetric learned whole-pair deduction."""
import numpy as np
from weld_pipeline.sjg_pair_residual_1203 import ResidualScorer,ResidualRelations
from weld_pipeline.sjg_nearest_parent_features_1184 import NearestParentRelations
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_local_runtime_904 import digest

SHAPES=dict(W1=(16,8),b1=(8,),W2=(8,),b2=(1,))
SCHEMA='frozen367-frozen373-symmetric-duplicate16-8-1-v1'


def reverse_column(column,index,lengths):
    """Exact original reverse fields; abs(log(a/b)) is recomputed, not negated."""
    reverse=column.copy();reverse[:,0]=column[:,1];reverse[:,1]=column[:,0]
    reverse[:,3]=[abs(np.log(float(lengths[index])/float(v))) for v in lengths]
    reverse[index]=0.
    return reverse


class DuplicateScorer:
    choose_next=staticmethod(ResidualScorer.choose_next)
    def __init__(self,frozen,frozen_sha,seed=1031):
        self.frozen=frozen;self.frozen_sha=frozen_sha;self.base_sha=frozen.base_sha;self.seed=seed
        for v in frozen.parameters.values():v.setflags(write=False)
        self.resume_rng=np.random.Generator(np.random.PCG64(seed))
        self.parameters=dict(W1=self.resume_rng.normal(0,.02,(16,8)),b1=np.zeros(8),W2=np.zeros(8),b2=np.zeros(1))
        self.initialization_rng_state=self.resume_rng.bit_generator.state

    def pair(self,a,b,jacobian=False):
        p=self.parameters;ha=np.tanh(a@p['W1']+p['b1']);hb=np.tanh(b@p['W1']+p['b1'])
        value=.5*((ha@p['W2']+p['b2'][0])+(hb@p['W2']+p['b2'][0]))
        if not jacobian:return value,None
        da=(1-ha*ha)*p['W2'];db=(1-hb*hb)*p['W2']
        dw=.5*(a[:,:,None]*da[:,None,:]+b[:,:,None]*db[:,None,:])
        jac=np.c_[dw.reshape(len(a),128),.5*(da+db),.5*(ha+hb),np.ones(len(a))]
        assert jac.shape==(len(a),145)
        return value,jac

    def forward(self,u,features,selected):
        frozen_features,pool=features;pool.check();assert pool.selected==list(selected)
        scores,_=self.frozen.forward(u,frozen_features,selected)
        scores=scores.copy();scores[:-1]-=pool.maximum
        return scores,pool

    def backward(self,pool,delta):
        pool.check();assert pool.with_gradient
        return unpack(-np.asarray(delta)[:-1]@pool.maximum_jacobian)

    def save(self,path):
        arrays=dict(self.parameters,schema=np.array(SCHEMA),seed=np.array(self.seed,dtype=np.int64),
            base_sha256=np.array(self.base_sha),frozen_residual_sha256=np.array(self.frozen_sha))
        atomic_binary_write(path,lambda stream:np.savez(stream,**arrays))

    @classmethod
    def load(cls,path,frozen_path,base_path):
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)==set(SHAPES)|{'schema','seed','base_sha256','frozen_residual_sha256'}
            assert str(z['schema'])==SCHEMA and str(z['base_sha256'])==digest(base_path)
            assert str(z['frozen_residual_sha256'])==digest(frozen_path)
            result=cls(ResidualScorer.load(frozen_path,base_path),digest(frozen_path),int(z['seed']))
            for k,shape in SHAPES.items():
                v=z[k];assert v.shape==shape and v.dtype==np.float64 and np.isfinite(v).all()
                result.parameters[k]=v.copy()
        return result


def pack(grad):return np.concatenate([grad[k].reshape(-1) for k in SHAPES])


def unpack(vector):
    result={};offset=0
    for key,shape in SHAPES.items():
        count=int(np.prod(shape));result[key]=vector[offset:offset+count].reshape(shape).copy();offset+=count
    assert offset==145
    return result


class DuplicatePool:
    def __init__(self,model,n,with_gradient=False):
        self.model=model;self.n=n;self.with_gradient=with_gradient;self.selected=[]
        self.snapshot={k:v.copy() for k,v in model.parameters.items()}
        self.maximum=np.zeros(n);self.argmax=np.full(n,-1,np.int64)
        self.maximum_jacobian=np.zeros((n,145)) if with_gradient else None
        # Per-candidate accumulated sufficient statistics across selected pairs;
        # excludes selected candidate rows when read. No prefix/pair tensor.
        self.aux_counts=np.zeros((n,2),np.int64) if with_gradient else None
        self.aux_losses=np.zeros((n,2)) if with_gradient else None
        self.aux_gradients=np.zeros((n,2,145)) if with_gradient else None

    def check(self):
        assert all(np.array_equal(v,self.model.parameters[k]) for k,v in self.snapshot.items()), 'Invalidate learned head scores and derivatives after update'

    def add(self,index,a,b,labels=None):
        self.check();assert index not in self.selected
        h,jac=self.model.pair(a,b,self.with_gradient)
        replace=(h>self.maximum)|((h==self.maximum)&(h>0)&((self.argmax<0)|(index<self.argmax)))
        self.maximum[replace]=h[replace];self.argmax[replace]=index
        if self.with_gradient:
            self.maximum_jacobian[replace]=jac[replace]
            if labels is not None:
                ids=np.flatnonzero(labels>=0);y=labels[ids];sign=1.-2.*y
                value=sign*h[ids];derivative=sign*np.exp(-np.logaddexp(0.,-value))
                self.aux_counts[ids,y]+=1
                self.aux_losses[ids,y]+=np.logaddexp(0.,value)
                self.aux_gradients[ids,y]+=derivative[:,None]*jac[ids]
        self.selected.append(index)

    def auxiliary(self,available):
        self.check();assert self.with_gradient
        mask=np.asarray(available,bool)[:-1]
        counts=self.aux_counts[mask].sum(0);active=counts>0
        if not active.any():return 0.,unpack(np.zeros(145)),counts.tolist()
        losses=self.aux_losses[mask].sum(0);grad=self.aux_gradients[mask].sum(0)
        loss=float(np.mean(losses[active]/counts[active]))
        return loss,unpack(np.mean(grad[active]/counts[active,None],axis=0)),counts.tolist()


class DuplicateRelations(ResidualRelations):
    def __init__(self,parents,metadata,model,key=None,cache=None,supervision=None,with_gradient=False):
        super().__init__(parents,metadata,model.frozen,key,cache)
        self.head=DuplicatePool(model,len(parents),with_gradient)
        self.lengths=[m['length_mm'] for m in metadata];self.supervision=supervision

    def add(self,index):
        column=self.raw_column(index);reverse=reverse_column(column,index,self.lengths)
        labels=self.supervision.column(self.key,index,column,reverse) if self.supervision is not None else None
        NearestParentRelations.add_column(self,index,column);self.pool.add(index,column)
        self.head.add(index,column,reverse,labels)

    def features(self):return super().features(),self.head


class DuplicateAdam:
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
