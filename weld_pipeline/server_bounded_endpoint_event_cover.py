"""Input-only finite-line event subset with a fixed metric covering bound.

Keep both observed extrema of every <=1mm cell, plus all original anchors.
Coordinates remain existing events. Any omitted event is <=0.5mm from a kept
cell extremum; this does not guarantee the same 3mm target membership.
"""
import numpy as np

def bounded_event_cover(events,length_mm,anchors,cell_mm=1.):
 q=np.asarray(events,np.float64);a=np.asarray(anchors,np.float64);length=float(length_mm)
 if length<=0 or not np.isfinite(length) or cell_mm<=0:raise ValueError('Positive finite length and cell width required')
 if not np.isfinite(q).all() or np.any(q<0) or np.any(q>length) or np.any(np.diff(q)<=0):raise ValueError('Finite strictly increasing events inside carrier required')
 if not len(q):return np.unique(a)
 cells=max(1,int(np.ceil(length/cell_mm)));scaled=q/length*cells;bins=np.minimum(np.floor(scaled).astype(np.int64),cells-1)
 # Closed-cell boundary membership prevents orientation-dependent side assignment.
 rounded=np.rint(scaled);edge=(abs(scaled-rounded)<=1e-10)&(rounded>0)&(rounded<cells)
 ids=np.arange(len(q));groups=np.r_[bins,np.maximum(rounded[edge].astype(np.int64)-1,0),rounded[edge].astype(np.int64)];rows=np.r_[ids,ids[edge],ids[edge]];order=np.lexsort((q[rows],groups));groups=groups[order];rows=rows[order];first=np.r_[True,groups[1:]!=groups[:-1]];last=np.r_[groups[:-1]!=groups[1:],True];keep=np.unique(rows[first|last]);return np.union1d(q[keep],a)
