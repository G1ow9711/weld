"""Function-preserving 194 -> 210 expansion with an explicit old affine branch."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_parent_set_scorer_1159 import ParentSetScorer,SHAPES as OLD_SHAPES
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write

SCHEMA='parent-set-mlp-210-32-1-nearest16-stop-322-v1'
SHAPES=dict(OLD_SHAPES,W1=(210,32))


def expand(value):
    result=np.zeros((210,32),dtype=np.float64)
    result[:192]=value[:192];result[208:]=value[192:]
    return result


def old_coordinates(value):
    return np.concatenate((value[:192],value[208:]),axis=0)


class NearestParentScorer(ParentSetScorer):
    def __init__(self,seed=1031):
        super().__init__(seed)
        self.parameters['W1']=expand(self.parameters['W1'])

    def forward(self,unary,relations,selected):
        unary=np.asarray(unary,dtype=np.float64);relations=np.asarray(relations,dtype=np.float64)
        if unary.ndim!=2 or unary.shape[1]!=160 or relations.shape!=(len(unary),48):raise ValueError('Expected U160 R48')
        if not np.isfinite(unary).all() or not np.isfinite(relations).all():raise ValueError('Nonfinite features')
        selected=tuple(selected)
        if len(set(selected))!=len(selected) or any(not isinstance(i,(int,np.integer)) or not 0<=i<len(unary) for i in selected):raise ValueError('Invalid selected indices')
        if not selected and np.any(relations!=0):raise ValueError('Empty relations must be zero')
        state=np.array([np.log1p(len(selected)),float(not selected)])
        old_inputs=np.empty((len(unary),194),np.float64)
        old_inputs[:,:160]=unary;old_inputs[:,160:192]=relations[:,:32];old_inputs[:,192:]=state
        inputs=np.empty((len(unary),210),np.float64)
        inputs[:,:192]=old_inputs[:,:192];inputs[:,192:208]=relations[:,32:];inputs[:,208:]=state
        all_mean=unary.mean(0) if len(unary) else np.zeros(160)
        selected_mean=unary[list(selected)].mean(0) if selected else np.zeros(160)
        stop_input=np.concatenate((all_mean,selected_mean,state));p=self.parameters
        old_affine=old_inputs@old_coordinates(p['W1'])+p['b1']
        hidden=np.tanh(old_affine+relations[:,32:]@p['W1'][192:208])
        logits=np.r_[hidden@p['W2']+p['b2'][0],stop_input@p['Ws']+p['bs'][0]]
        if not np.isfinite(logits).all():raise FloatingPointError('Nonfinite output')
        return logits,(inputs,hidden,stop_input)

    def save(self,path):
        arrays=dict(self.parameters,schema=np.array(SCHEMA),seed=np.array(self.seed,dtype=np.int64))
        atomic_binary_write(Path(path),lambda stream:np.savez(stream,**arrays))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!=set(SHAPES)|{'schema','seed'} or str(data['schema'])!=SCHEMA:raise ValueError('Wrong schema')
            model=cls(int(data['seed']))
            for key,shape in SHAPES.items():
                value=data[key]
                if value.dtype!=np.float64 or value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid parameter')
                model.parameters[key]=value.copy()
        return model


class NearestAdamState:
    def __init__(self,model):
        self.step_count=0
        self.first={k:np.zeros_like(v) for k,v in model.parameters.items()}
        self.second={k:np.zeros_like(v) for k,v in model.parameters.items()}

    def update(self,model,gradients,lr=1e-3,weight_decay=1e-4,clip=5.):
        if set(gradients)!=set(SHAPES):raise ValueError('Incomplete gradients')
        if not (np.isfinite([lr,weight_decay,clip]).all() and lr>0 and weight_decay>=0 and clip>0):raise ValueError('Invalid optimizer settings')
        for k,v in gradients.items():
            if v.shape!=SHAPES[k] or not np.isfinite(v).all():raise ValueError('Invalid gradient')
        # Preserve the original coordinate reduction; append the new rows' norm.
        old_norm=sum(np.sum((old_coordinates(v) if k=='W1' else v)**2) for k,v in gradients.items())
        norm=float(np.sqrt(old_norm+np.sum(gradients['W1'][192:208]**2)))
        scale=min(1.,clip/max(norm,np.finfo(float).tiny));self.step_count+=1
        for k,p in model.parameters.items():
            g=gradients[k]*scale+weight_decay*p
            self.first[k]=.9*self.first[k]+.1*g
            self.second[k]=.999*self.second[k]+.001*g*g
            first=self.first[k]/(1.-.9**self.step_count)
            second=self.second[k]/(1.-.999**self.step_count)
            p-=lr*first/(np.sqrt(second)+1e-8)
            if not np.isfinite(p).all():raise FloatingPointError('Nonfinite optimizer output')
        return dict(step=self.step_count,raw_gradient_norm=norm,clip_scale=scale)


def migrate(old,old_optimizer):
    model=NearestParentScorer(old.seed);optimizer=NearestAdamState(model)
    for key in SHAPES:
        for source,target in ((old.parameters,model.parameters),(old_optimizer.first,optimizer.first),(old_optimizer.second,optimizer.second)):
            target[key]=expand(source[key]) if key=='W1' else source[key].copy()
    optimizer.step_count=old_optimizer.step_count
    model.resume_rng=old.resume_rng
    return model,optimizer
