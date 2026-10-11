"""Lazy point-derived centroid/covariance and plate support features."""
import numpy as np

FEATURES=141


class ContextFeatures:
    def __init__(self,sequence,structural,mean,covariance):
        self.base=sequence['features'];self.xyz=sequence['xyz']
        self.structural=np.asarray(structural,np.float32)
        self.mean=np.asarray(mean,float)
        c=np.asarray(covariance,float)/1e6
        self.covariance=c[np.triu_indices(3)].astype(np.float32)
        t=self.base[:,57:60].astype(float)
        difference=np.linalg.norm(np.diff(t,axis=0),axis=1)
        self.curvature=(np.r_[difference,0.]+np.r_[0.,difference])*.5
        self.shape=(len(self.base),FEATURES);self.dtype=np.dtype('float32')

    def __len__(self):
        return len(self.base)

    def __getitem__(self,key):
        indices,columns=key if isinstance(key,tuple) else (key,slice(None))
        if isinstance(columns,int) and columns<84 and columns not in range(49,52):
            return self.base[indices,columns]
        core=np.array(self.base[indices],copy=True)
        scalar=core.ndim==1
        if scalar:
            core=core[None,:]
        q=np.atleast_2d(self.xyz[indices]).astype(float)
        core[:,49:52]=(q-self.mean)/1000.
        extras=np.column_stack([np.broadcast_to(self.structural,(len(core),50)),
                                np.broadcast_to(self.covariance,(len(core),6)),
                                np.atleast_1d(self.curvature[indices])])
        result=np.column_stack([core,extras]).astype(np.float32)
        result=result[0] if scalar else result
        return result[...,columns]


def context_sequences(sequences,structural,mean,covariance):
    return [{**s,'features':ContextFeatures(s,structural[i],mean,covariance)} for i,s in enumerate(sequences)]
