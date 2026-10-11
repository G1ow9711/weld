"""Exact vectorized min/max range queries over sorted point projections."""
import numpy as np


class RangeExtrema:
    def __init__(self,position,value,block_size=64):
        position=np.asarray(position,float);value=np.asarray(value,float)
        if len(position)!=len(value):raise ValueError('Lengths differ')
        if not np.isfinite(position).all() or not np.isfinite(value).all():raise ValueError('Nonfinite range data')
        order=np.argsort(position,kind='stable');self.t=position[order];self.z=value[order];self.block=block_size
        if not len(self.z):self.low=[];self.high=[];return
        size=int(np.ceil(len(self.z)/block_size))*block_size
        lo=np.full(size,np.inf);hi=np.full(size,-np.inf);lo[:len(value)]=self.z;hi[:len(value)]=self.z
        self.low=[lo.reshape(-1,block_size).min(1)];self.high=[hi.reshape(-1,block_size).max(1)]
        stride=1
        while 2*stride<=len(self.low[0]):
            self.low.append(np.minimum(self.low[-1][:-stride],self.low[-1][stride:]))
            self.high.append(np.maximum(self.high[-1][:-stride],self.high[-1][stride:]));stride*=2

    def query(self,queries,half_width):
        q=np.asarray(queries,float)
        if half_width<=0:raise ValueError('Positive half width required')
        l=np.searchsorted(self.t,q-half_width,side='left');r=np.searchsorted(self.t,q+half_width,side='right')
        counts=r-l;low=np.full(len(q),np.nan);high=low.copy()
        if not len(self.z):return low,high,counts
        block=self.block;offsets=np.arange(block)
        for start in range(0,len(q),8192):
            stop=min(len(q),start+8192);a=l[start:stop];b=r[start:stop]
            first=(a+block-1)//block;last=b//block
            left_ids=a[:,None]+offsets;right_ids=np.maximum(a,last*block)[:,None]+offsets
            left_valid=(left_ids<np.minimum(b,first*block)[:,None]);right_valid=right_ids<b[:,None]
            left_values=self.z[np.minimum(left_ids,len(self.z)-1)];right_values=self.z[np.minimum(right_ids,len(self.z)-1)]
            lo=np.minimum(np.where(left_valid,left_values,np.inf).min(1),np.where(right_valid,right_values,np.inf).min(1))
            hi=np.maximum(np.where(left_valid,left_values,-np.inf).max(1),np.where(right_valid,right_values,-np.inf).max(1))
            full=last-first
            for k in range(len(self.low)):
                mask=(full>=2**k)&(full<2**(k+1))
                if not mask.any():continue
                left=first[mask];right=last[mask]-2**k
                lo[mask]=np.minimum(lo[mask],np.minimum(self.low[k][left],self.low[k][right]))
                hi[mask]=np.maximum(hi[mask],np.maximum(self.high[k][left],self.high[k][right]))
            active=b>a;low[start:stop][active]=lo[active];high[start:stop][active]=hi[active]
        return low,high,counts
