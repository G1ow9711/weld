"""Same population moments with stable weighted online accumulation."""
import inspect
import types
import numpy as np
from . import curveonly_01a1ac7c_whole_intervals_v316 as original
from . import curveonly_01a1ac7c_structural_whole_model_v324 as projection
from . import curveonly_01a1ac7c_material_interval_solver_v327 as previous

NAMES,SETTINGS,ACTIVE,REMOVED=previous.NAMES,previous.SETTINGS,previous.ACTIVE,previous.REMOVED
base,profile,provenance,TOL=previous.base,previous.profile,previous.provenance,previous.TOL
pairs,fields,select=previous.pairs,previous.fields,previous.select


def stable_moments(values,widths,indexes):
    """All requested native cell intervals in O(cell_count^2*feature_count).

Each start advances once over its cells. The nonnegative M2 update avoids
subtracting almost equal squared sums, and a constant has exactly zero M2.
The definition remains an arclength-weighted population mean and variance.
"""
    values=np.asarray(values,float);widths=np.asarray(widths,float);indexes=np.asarray(indexes,int)
    assert np.all(widths>0.) and len(values)==len(widths)
    a,b=indexes.T;length=b-a;starts=int(a.max())+1;maximum=int(length.max())
    order=np.argsort(length,kind='stable');limits=np.searchsorted(length[order],np.arange(maximum+2))
    mean=np.zeros((starts,values.shape[1]));m2=np.zeros_like(mean);weight=np.zeros(starts)
    means=np.empty((len(indexes),values.shape[1]));std=np.empty_like(means)
    for size in range(1,maximum+1):
        active=min(starts,len(values)-size+1);positions=np.arange(active)+size-1
        w=widths[positions];combined=weight[:active]+w;delta=values[positions]-mean[:active]
        m2[:active]+=delta*delta*(weight[:active]*w/combined)[:,None]
        mean[:active]+=delta*(w/combined)[:,None];weight[:active]=combined
        rows=order[limits[size]:limits[size+1]];begin=a[rows]
        means[rows]=mean[begin];std[rows]=np.sqrt(m2[begin]/weight[begin,None])
    assert np.isfinite(means).all() and np.isfinite(std).all()
    return means,std


_source=inspect.getsource(original.features)
for _old,_new in [
    ('np.vstack([event_context[:-1], event_context])','np.vstack([event_context, event_context[1:]])'),
    ('np.vstack([xyz[:-1], xyz])','np.vstack([xyz, xyz[1:]])'),
    ('np.vstack([tangent[:-1], tangent])','np.vstack([tangent, tangent[1:]])'),
]:
    assert _source.count(_old)==1;_source=_source.replace(_old,_new)
_old='''    prefix = np.vstack([np.zeros(130), np.cumsum(x*widths[:, None], axis=0)])
    second = np.vstack([np.zeros(130), np.cumsum(x*x*widths[:, None], axis=0)])
    mean = (prefix[b]-prefix[a])/width[:, None]
    std = np.sqrt(np.maximum(0., (second[b]-second[a])/width[:, None]-mean*mean))'''
assert _source.count(_old)==1
_source=_source.replace(_old,'    mean, std = stable_moments(x, widths, indexes)')
_namespace=dict(original.__dict__,stable_moments=stable_moments)
exec(compile(_source,__file__,'exec'),_namespace)
raw_features=_namespace['features']


def features(*args,**kwargs):
    f,indexes,bounds=raw_features(*args,**kwargs)
    return projection.project(f),indexes,bounds


class WholeIntervalModel(previous.WholeIntervalModel):
    predict=types.FunctionType(original.WholeIntervalModel.predict.__code__,dict(original.__dict__,features=features,select=select))


decode,input_curves=previous.decode,previous.input_curves
