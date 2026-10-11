"""Frozen best scorer plus a signed, differentiable all-selected-pair residual."""
import weakref
import numpy as np
from weld_pipeline.sjg_nearest_parent_scorer_1185 import NearestParentScorer
from weld_pipeline.sjg_nearest_parent_features_1184 import NearestParentRelations
from weld_pipeline.sjg_parent_set_scorer_1159 import ParentSetScorer
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_local_runtime_904 import digest

SCHEMA='frozen367-pair16-8-maxmean16-residual178-16-1-v1'
SHAPES=dict(Wr=(16,8),br=(8,),W1=(178,16),b1=(16,),W2=(16,),b2=(1,))


class ResidualScorer:
    choose_next=staticmethod(ParentSetScorer.choose_next)
    def __init__(self,base,base_sha,seed=1031):
        self.base=base;self.base_sha=base_sha;self.seed=seed
        for value in base.parameters.values():value.setflags(write=False)
        rng=np.random.Generator(np.random.PCG64(seed))
        self.parameters=dict(Wr=rng.normal(0,.02,SHAPES['Wr']),br=np.zeros(8),
            W1=rng.normal(0,.02,SHAPES['W1']),b1=np.zeros(16),W2=np.zeros(16),b2=np.zeros(1))
        self.initialization_rng_state=rng.bit_generator.state;self.resume_rng=rng
        assert sum(v.size for v in self.parameters.values())==3017

    def forward(self,unary,features,selected):
        old,z,pool=features;assert pool.selected==list(selected)
        scores,_=self.base.forward(unary,old,selected)
        p=self.parameters;x=np.empty((len(unary),178),np.float64)
        x[:,:160]=unary;x[:,160:176]=z;x[:,176:]=[np.log1p(len(selected)),float(not selected)]
        h=np.tanh(x@p['W1']+p['b1']);delta=h@p['W2']+p['b2'][0]
        scores=scores.copy();scores[:-1]+=delta
        assert np.isfinite(scores).all()
        return scores,(x,h,pool)

    def backward(self,cache,score_gradient):
        x,h,pool=cache;d=np.asarray(score_gradient,np.float64)
        assert d.shape==(len(x)+1,) and np.isfinite(d).all()
        dc=d[:-1];p=self.parameters;dh=dc[:,None]*p['W2']*(1-h*h)
        dz=(dh@p['W1'].T)[:,160:176]
        wr,br=pool.backward(dz)
        return dict(Wr=wr,br=br,W1=x.T@dh,b1=dh.sum(0),W2=h.T@dc,b2=np.array([dc.sum()]))

    def save(self,path):
        arrays=dict(self.parameters,schema=np.array(SCHEMA),seed=np.array(self.seed,dtype=np.int64),base_sha256=np.array(self.base_sha))
        atomic_binary_write(path,lambda stream:np.savez(stream,**arrays))

    @classmethod
    def load(cls,path,base_path):
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)==set(SHAPES)|{'schema','seed','base_sha256'} and str(z['schema'])==SCHEMA
            assert str(z['base_sha256'])==digest(base_path)
            model=cls(NearestParentScorer.load(base_path),str(z['base_sha256']),int(z['seed']))
            for k,shape in SHAPES.items():
                v=z[k];assert v.shape==shape and v.dtype==np.float64 and np.isfinite(v).all()
                model.parameters[k]=v.copy()
        return model


class EncodedSet:
    def __init__(self,model,n,column):
        self.model=model;self.n=n;self.column=column;self.selected=[]
        self.total=np.zeros((n,8));self.maximum=np.full((n,8),-np.inf)
        self.argmax=np.full((n,8),-1,dtype=np.int64)
        self.wr=model.parameters['Wr'].copy();self.br=model.parameters['br'].copy()

    def check(self):
        assert np.array_equal(self.wr,self.model.parameters['Wr']) and np.array_equal(self.br,self.model.parameters['br']), 'Learned encoding cache must be invalidated after update'

    def add(self,index,column=None):
        self.check();assert index not in self.selected
        r=self.column(index) if column is None else column
        h=np.tanh(r@self.wr+self.br)
        replace=(h>self.maximum)|((h==self.maximum)&((self.argmax<0)|(index<self.argmax)))
        self.maximum[replace]=h[replace];self.argmax[replace]=index
        self.total+=h;self.selected.append(index)

    def features(self):
        self.check()
        if not self.selected:return np.zeros((self.n,16))
        return np.concatenate((self.maximum,self.total/len(self.selected)),axis=1)

    def backward(self,dz):
        self.check();assert dz.shape==(self.n,16)
        wr=np.zeros((16,8));br=np.zeros(8)
        if not self.selected:return wr,br
        active=np.flatnonzero(np.any(dz!=0,axis=1))
        if not len(active):return wr,br
        # The pairwise loss touches at most two candidate logits. Recompute only
        # their rows per selected column; never retain all prefix/pair tensors.
        for index in self.selected:
            r=self.column(index)[active];h=np.tanh(r@self.wr+self.br)
            dh=dz[active,8:]/len(self.selected)+dz[active,:8]*(self.argmax[active]==index)
            da=dh*(1-h*h);wr+=r.T@da;br+=da.sum(0)
        return wr,br


class ResidualRelations(NearestParentRelations):
    def __init__(self,parents,metadata,model,key=None,cache=None):
        super().__init__(parents,metadata,key,cache)
        weak=weakref.ref(self)
        self.pool=EncodedSet(model,len(parents),lambda index:weak().raw_column(index))

    def raw_column(self,index):
        if self.cache is not None:return self.cache.column(self.key,index,self)
        column=np.zeros((len(self.parents),16))
        for i in range(len(self.parents)):
            if i!=index:column[i]=self.pair(i,index)
        return column

    def add(self,index):
        column=self.raw_column(index);super().add_column(index,column);self.pool.add(index,column)

    def features(self):return super().features(),self.pool.features(),self.pool


class ResidualAdam:
    def __init__(self,model):
        self.step_count=0;self.first={k:np.zeros_like(v) for k,v in model.parameters.items()}
        self.second={k:np.zeros_like(v) for k,v in model.parameters.items()}

    def update(self,model,gradients,lr=.001,weight_decay=.0001,clip=5.):
        assert set(gradients)==set(SHAPES)
        for k,v in gradients.items():assert v.shape==SHAPES[k] and np.isfinite(v).all()
        norm=float(np.sqrt(sum(np.sum(v*v) for v in gradients.values())))
        scale=min(1.,clip/max(norm,np.finfo(float).tiny));self.step_count+=1
        for k,p in model.parameters.items():
            g=gradients[k]*scale+weight_decay*p
            self.first[k]=.9*self.first[k]+.1*g;self.second[k]=.999*self.second[k]+.001*g*g
            first=self.first[k]/(1-.9**self.step_count);second=self.second[k]/(1-.999**self.step_count)
            p-=lr*first/(np.sqrt(second)+1e-8);assert np.isfinite(p).all()
        return dict(step=self.step_count,raw_gradient_norm=norm,clip_scale=scale)
