"""Arc-length evidence and a single additive interval margin; no truth in features.

Validity means observed, computable face evidence, not physical material occupancy.
These functions neither fit normalizers nor insert/move geometric endpoints.
"""
import numpy as np

BINS=4
SEQ_DIM=56
BIN_DIM=60
FEATURE_DIM=2*180+BINS*BIN_DIM+4+1
# Observed face one, face two, and paired-normal evidence use separate lengths.
GROUP=np.zeros(56,np.int8)
GROUP[np.r_[0:11,52:54]]=1
GROUP[np.r_[11:22,54:56]]=2
GROUP[24]=3

class ArcLengthEvidence:
    def __init__(self,grid,x,mean,scale,closed=False):
        self.grid=np.asarray(grid,np.float64);x=np.asarray(x,np.float64)
        assert x.shape==(len(self.grid),56) and len(x)>=2
        assert self.grid[0]==0 and np.all(np.diff(self.grid)>0) and np.isfinite(x).all()
        mean=np.asarray(mean,np.float64);scale=np.asarray(scale,np.float64)
        assert mean.shape==scale.shape==(56,) and np.all(scale>0)
        assert np.all((x[:,22:24]==0)|(x[:,22:24]==1))
        self.length=float(self.grid[-1]);self.closed=closed
        self.y=(x-mean)/scale;self.d=np.diff(self.grid)
        face=(x[:-1,22:24]>.5)&(x[1:,22:24]>.5)
        self.valid=np.column_stack((np.ones(len(face),bool),face,face.all(1))).astype(np.float64)
        self.valid_prefix=np.vstack((np.zeros((1,4)),np.cumsum(self.d[:,None]*self.valid,axis=0)))
        integral=.5*(self.y[:-1]+self.y[1:])*self.d[:,None]*self.valid[:,GROUP]
        self.prefix=np.vstack((np.zeros((1,56)),np.cumsum(integral,axis=0)))

    def positions(self,q):
        q=np.asarray(q,np.float64)
        assert np.isfinite(q).all()
        if self.closed:
            cycles=np.floor(q/self.length);q=q-cycles*self.length
        else:
            assert np.all(q>=-1e-8) and np.all(q<=self.length+1e-8)
            q=np.clip(q,0,self.length);cycles=np.zeros_like(q)
        i=np.clip(np.searchsorted(self.grid,q,side='right')-1,0,len(self.grid)-2)
        return i,q-self.grid[i],cycles

    def valid_at(self,q):
        i,h,cycles=self.positions(q)
        return self.valid_prefix[i]+h[...,None]*self.valid[i]+cycles[...,None]*self.valid_prefix[-1]

    def feature_integral_at(self,q):
        i,h,cycles=self.positions(q)
        value=self.y[i]*h[...,None]+.5*(self.y[i+1]-self.y[i])*(h*h/self.d[i])[...,None]
        return self.prefix[i]+value*self.valid[i][:,GROUP]+cycles[...,None]*self.prefix[-1]

    def bounds(self,lo,hi):
        lo=np.atleast_1d(np.asarray(lo,np.float64));hi=np.atleast_1d(np.asarray(hi,np.float64))
        assert lo.shape==hi.shape and lo.ndim==1 and np.all(hi>lo)
        assert np.all(hi-lo<=self.length+1e-8)
        # Closed intervals may cross the parameter origin in unwrapped arc length.
        return lo[:,None]+(hi-lo)[:,None]*np.linspace(0,1,5)[None,:]

    def features(self,lo,hi):
        edges=self.bounds(lo,hi);pieces=[]
        for b in range(BINS):
            left,right=edges[:,b],edges[:,b+1]
            den=self.valid_at(right)-self.valid_at(left)
            num=self.feature_integral_at(right)-self.feature_integral_at(left)
            values=np.divide(num,den[:,GROUP],out=np.zeros_like(num),where=den[:,GROUP]>0)
            coverage=den/(right-left)[:,None]
            assert np.all(coverage>=-1e-9) and np.all(coverage<=1+1e-9)
            pieces.append(np.c_[values,np.clip(coverage,0,1)])
        result=np.concatenate(pieces,axis=1);assert result.shape==(len(edges),240)
        return result

    def project(self,weights):
        """Project once per carrier; full-domain scoring never materializes Npair x 56."""
        return ProjectedEvidence(self,weights)

class ProjectedEvidence:
    def __init__(self,evidence,weights):
        self.e=evidence;self.w=np.asarray(weights,np.float64)
        assert self.w.ndim==3 and self.w.shape[1:]==(4,60)
        n=len(evidence.grid);t=len(weights)
        self.values=np.zeros((n,4,4,t))
        for b in range(4):
            for g in range(4):
                ids=np.flatnonzero(GROUP==g)
                self.values[:,b,g,:]=evidence.y[:,ids]@self.w[:,b,ids].T
        integral=.5*(self.values[:-1]+self.values[1:])*evidence.d[:,None,None,None]*evidence.valid[:,None,:,None]
        self.prefix=np.concatenate((np.zeros((1,4,4,t)),np.cumsum(integral,axis=0)),axis=0)

    def at(self,q,b):
        e=self.e;i,h,cycles=e.positions(q)
        value=self.values[i,b]*h[:,None,None]+.5*(self.values[i+1,b]-self.values[i,b])*(h*h/e.d[i])[:,None,None]
        return self.prefix[i,b]+value*e.valid[i,:,None]+cycles[:,None,None]*self.prefix[-1,b]

    def scores(self,lo,hi):
        edges=self.e.bounds(lo,hi);result=np.zeros((len(edges),len(self.w)))
        for b in range(4):
            left,right=edges[:,b],edges[:,b+1];den=self.e.valid_at(right)-self.e.valid_at(left)
            num=self.at(right,b)-self.at(left,b)
            value=np.divide(num,den[:,:,None],out=np.zeros_like(num),where=den[:,:,None]>0)
            result+=value.sum(1)+(den/(right-left)[:,None])@self.w[:,b,56:60].T
        return result

def complete_features(left,right,interior,scalars):
    assert left.shape==right.shape and left.shape[1]==180
    assert interior.shape==(len(left),240) and scalars.shape==(len(left),4)
    result=np.c_[left,right,interior,scalars,np.ones(len(left))]
    assert result.shape[1]==FEATURE_DIM and np.isfinite(result).all()
    return result

def sigmoid(x):
    return np.exp(-np.logaddexp(0.,-np.asarray(x)))

def loss_gradient(theta,x,base,parents,negative_mask,carrier_ids,G):
    """Select one complete witness per parent; never mix per-child best views.

    parents: each item is a list of complete views; each view is [(row,class),...].
    Weights sum to one across original children of this workpiece, regardless of aliases.
    """
    score=base+x@theta.T;dscore=np.zeros_like(score);value=0.;choices=[]
    assert score.shape==negative_mask.shape and score.shape[1]==4
    assert sum(len(p[0]) for p in parents)==G
    assert all(len(v)==len(p[0]) for p in parents for v in p)
    hard={}
    for carrier in np.unique(carrier_ids):
        rows=np.flatnonzero(carrier_ids==carrier);rr,cc=np.where(negative_mask[rows])
        if len(rr):
            best=int(score[rows[rr],cc].argmax());hard[int(carrier)]=(int(rows[rr[best]]),int(cc[best]))
    for p in parents:
        alternatives=[]
        for view in p:
            loss=0.;terms=[]
            for i,c in view:
                assert not negative_mask[i,c]
                loss+=float(np.logaddexp(0.,-score[i,c]));terms.append((i,c,-float(sigmoid(-score[i,c]))))
                neg=hard.get(int(carrier_ids[i]))
                if neg is not None:
                    j,k=neg;z=score[j,k]-score[i,c];loss+=float(np.logaddexp(0.,z));g=float(sigmoid(z))
                    terms.extend(((j,k,g),(i,c,-g)))
            alternatives.append((loss,terms))
        selected=min(range(len(alternatives)),key=lambda k:alternatives[k][0]);choices.append(selected)
        loss,terms=alternatives[selected];value+=loss/G
        for i,c,g in terms:dscore[i,c]+=g/G
    count=int(negative_mask.sum())
    if count:
        value+=float(np.logaddexp(0.,score[negative_mask]).sum())/count
        dscore[negative_mask]+=sigmoid(score[negative_mask])/count
    return value,dscore.T@x,dict(selected_whole_parent_views=choices,score_gradient=dscore,
                                known_negative_explanations=count)

def adam_update(theta,gradient,state,lr=.01,weight_decay=.0001,clip=5.):
    # Coupled weight decay is applied once: equivalent to 0.5*lambda*||theta||^2.
    gradient=gradient+weight_decay*theta;norm=float(np.linalg.norm(gradient))
    gradient=gradient*min(1.,clip/max(norm,1e-300));step=int(state['step'])+1
    m=.9*state['m']+.1*gradient;v=.999*state['v']+.001*gradient*gradient
    new=theta-lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    assert np.isfinite(new).all()
    return new,dict(step=step,m=m,v=v),norm

def controls():
    rng=np.random.default_rng(71287);grid=np.array([0.,.7,2.1,4.,8.])
    x=rng.normal(size=(5,56));x[:,22:24]=1;x[2,22]=0
    mean=rng.normal(size=56);scale=rng.uniform(.5,2,size=56)
    e=ArcLengthEvidence(grid,x,mean,scale)
    lo=np.array([0.,.1,2.8,7.99]);hi=np.array([8.,3.7,6.,8.]);f=e.features(lo,hi)
    # Independently split each bin at every original knot and integrate trapezoids.
    maximum=0.
    for row,(a,b) in enumerate(zip(lo,hi)):
        boundaries=np.linspace(a,b,5)
        for k in range(4):
            a,b=boundaries[k:k+2];knots=np.r_[a,grid[(grid>a)&(grid<b)],b]
            num=np.zeros(56);den=np.zeros(4)
            for l,r in zip(knots[:-1],knots[1:]):
                mid=(l+r)/2;index=np.searchsorted(grid,mid)-1;valid=e.valid[index]
                yl=np.array([np.interp(l,grid,e.y[:,j]) for j in range(56)])
                yr=np.array([np.interp(r,grid,e.y[:,j]) for j in range(56)])
                num+=(r-l)*(yl+yr)/2*valid[GROUP];den+=(r-l)*valid
            reference=np.r_[np.divide(num,den[GROUP],out=np.zeros(56),where=den[GROUP]>0),den/(b-a)]
            maximum=max(maximum,float(abs(reference-f[row,k*60:(k+1)*60]).max()))
    assert maximum<1e-9
    w=rng.normal(size=(4,4,60));projected=e.project(w).scores(lo,hi)
    projection_error=float(abs(projected-f@w.reshape(4,240).T).max());assert projection_error<1e-9
    reverse=ArcLengthEvidence(8-grid[::-1],x[::-1],mean,scale)
    reversal=float(abs(f.reshape(-1,4,60)-reverse.features(8-hi,8-lo).reshape(-1,4,60)[:,::-1]).max());assert reversal<1e-9
    xp=x.copy();xp[-1]=xp[0];closed=ArcLengthEvidence(grid,xp,mean,scale,True)
    a=np.array([7.,-1.,0.]);b=np.array([9.,1.,8.]);cf=closed.features(a,b)
    periodic=float(abs(cf-closed.features(a+8,b+8)).max());assert periodic<1e-9
    cp=float(abs(closed.project(w).scores(a,b)-cf@w.reshape(4,240).T).max());assert cp<1e-9
    zero=x.copy();zero[:,22:24]=0;empty=ArcLengthEvidence(grid,zero,mean,scale).features(np.array([1.]),np.array([7.])).reshape(4,60)
    assert np.all(empty[:,:56][:,GROUP!=0]==0) and np.all(empty[:,57:60]==0)
    xx=rng.normal(size=(7,5));theta=rng.normal(size=(4,5))*.1;base=rng.normal(size=(7,4))
    neg=np.zeros((7,4),bool);neg[4:6]=True;rows=np.array([0,0,0,0,0,0,1])
    parents=[[[(0,0),(1,0)],[(2,2),(3,2)]]]
    loss,grad,meta=loss_gradient(theta,xx,base,parents,neg,rows,2);assert np.all(meta['score_gradient'][6]==0)
    finite=np.zeros_like(theta);eps=1e-6
    for i in range(4):
        for j in range(5):
            plus=theta.copy();minus=theta.copy();plus[i,j]+=eps;minus[i,j]-=eps
            finite[i,j]=(loss_gradient(plus,xx,base,parents,neg,rows,2)[0]-loss_gradient(minus,xx,base,parents,neg,rows,2)[0])/(2*eps)
    gradient_error=float(abs(finite-grad).max());assert gradient_error<1e-6
    no_mix_base=np.zeros((4,4));no_mix_base[0,0]=5;no_mix_base[1,0]=-5;no_mix_base[2,2]=-5;no_mix_base[3,2]=5
    value,_,_=loss_gradient(np.zeros((4,1)),np.zeros((4,1)),no_mix_base,parents,np.zeros((4,4),bool),np.arange(4),2)
    assert value>2.5
    duplicated=[[parents[0][0],parents[0][1],parents[0][0]]]
    assert loss_gradient(theta,xx,base,duplicated,neg,rows,2)[0]==loss
    return dict(status='PASS',explicit_integral_max_error=maximum,projection_max_error=projection_error,
                reverse_max_error=reversal,periodic_max_error=periodic,periodic_projection_error=cp,
                empty_support_has_zero_values_and_zero_coverage=True,gradient_max_error=gradient_error,
                whole_parent_view_not_independent_child_best=True,alias_duplication_no_extra_positive_mass=True,
                UNKNOWN_no_score_gradient=True,feature_dim=FEATURE_DIM,fit_calls=0)
