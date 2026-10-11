"""Algebraic descriptor reversal for finite LINE carriers only."""
import numpy as np


def reverse_line_descriptors(x,lengths_mm):
    if x.ndim!=2 or x.shape[1]!=160 or not np.all(x[:,97]==1):raise ValueError('Only line descriptors supported')
    lengths_mm=np.asarray(lengths_mm,float)
    if lengths_mm.shape!=(len(x),) or np.any(lengths_mm<=0):raise ValueError('Require original positive carrier lengths')
    y=x.copy();y[:,:96]=x[:,:96].reshape(-1,16,6)[:,::-1].reshape(-1,96)
    y[:,104:106]=-x[:,104:106]
    for base in (112,128):
        known=x[:,base]>0;L=lengths_mm[known]/1000.
        y[known,base+4]=L-x[known,base+4];y[known,base+6]=-x[known,base+6]
        y[known,base+10]=L-x[known,base+13];y[known,base+12]=-x[known,base+15]
        y[known,base+13]=L-x[known,base+10];y[known,base+15]=-x[known,base+12]
    y[:,144:152]=x[:,144:152][:,::-1];y[:,152:160]=x[:,152:160][:,::-1]
    return y
