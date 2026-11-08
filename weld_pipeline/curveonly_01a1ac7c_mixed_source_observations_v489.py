"""Use the existing carrier-native global fields after actual raw observations.

V487 replay exposed that V28 replaces raw station-fit length/radius/turn with
native carrier values. Reproduce that calculation; retain the original check.
"""
import types
import numpy as np
from . import curveonly_01a1ac7c_mixed_source_observations_v487 as previous


def raw34(candidates,xyz,base,pose):
    features,packet=previous.raw34(candidates,xyz,base,pose)
    for ci,(native,c) in enumerate(zip(candidates,packet['candidates'])):
        lo,hi=packet['offsets'][ci:ci+2];x=features[lo:hi]
        points=np.asarray(c['points']);stations=np.asarray(c['stations']);s=previous.cad.arclength(stations)
        length=float(previous.cad.arclength(points)[-1]);circle=native.get('circle')
        scale=float(np.expm1(x[0,0])/max(x[0,33],1e-12))
        tangent=np.gradient(stations,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-12)
        turn=abs(circle['sweep']) if circle else float(np.arccos(np.clip((tangent[1:]*tangent[:-1]).sum(1),-1,1)).sum())
        x[:,0]=np.log1p(length);x[:,1]=np.log1p(circle['radius']) if circle else 0.;x[:,2]=turn
        x[:,3]=circle is not None;x[:,4]=np.linalg.norm(points[0]-points[-1])<.1
        x[:,9]=s/max(s[-1],1e-12);x[:,10]=np.log1p(np.minimum(s,s[-1]-s));x[:,33]=length/max(scale,1.)
        c['circle']=circle
    return features,packet


append=types.FunctionType(previous.append.__code__,dict(previous.__dict__,raw34=raw34))
cad,frame,finite,volume,section,direction,encoder=[getattr(previous,k) for k in ['cad','frame','finite','volume','section','direction','encoder']]
