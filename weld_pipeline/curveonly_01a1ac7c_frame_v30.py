"""Canonical input curve-plane normal for unsigned plane identity.

SVD normals have two equivalent signs. The old probe columns accidentally
depended on that arbitrary sign. Canonicalizing swaps all four +/- pairs
together and preserves their relative material-side evidence.
"""
import time
import numpy as np
from . import curveonly_01a1ac7c_carrier_union_v28 as union

PAIRS=((21,22),(23,24),(25,26),(27,28))
LEFT=[a for a,b in PAIRS];RIGHT=[b for a,b in PAIRS]


def plane_normal(candidate):
    fit=candidate.get('circle')
    if fit:return np.asarray(fit['normal'],float)
    points=np.array(candidate['stations']);_,_,v=np.linalg.svd(points-points.mean(0),full_matrices=False)
    return v[-1]


def canonical(features,packet,normal_provider=None):
    start=time.perf_counter();x=np.asarray(features).copy();records=[]
    for ci,c in enumerate(packet['candidates']):
        n=plane_normal(c)if normal_provider is None else np.asarray(normal_provider(ci,c))
        axis=int(np.flatnonzero(np.abs(n)>=np.max(np.abs(n))-1e-10)[0]);flip=bool(n[axis]<0)
        a,b=packet['offsets'][ci:ci+2]
        if flip:
            before=x[a:b].copy();x[a:b,LEFT]=before[:,RIGHT];x[a:b,RIGHT]=before[:,LEFT]
        records.append(dict(candidate=ci,normal=(n*(-1 if flip else 1)).tolist(),pivot_axis=axis,
            original_arbitrary_sign_flipped=flip))
    audit=dict(policy='Largest absolute XYZ normal component positive; first axis wins ties within1e-10',
        original_curves=len(records),flipped=sum(v['original_arbitrary_sign_flipped']for v in records),
        records=records,no_geometric_queries=True,no_geometry_or_label_change=True,
        raw_and_physical_weld_side_normals_not_changed=True,CPU_seconds=time.perf_counter()-start)
    return x,dict(packet,normal_frame_canonicalization=audit)


def build(features,packet):
    x,p=canonical(features,packet);return union.build(x,p)
