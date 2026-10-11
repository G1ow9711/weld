"""Experimental numerical gap guard; preserves original model and geometry.

Uses the existing finite-profile roundoff scale (64 float64 eps), not a
physical gap threshold. A distance inside this band cannot establish a
positive gap; it does not establish physical weld contact either.
"""
import numpy as np
from weld_pipeline.sjg_joint_link_runtime_1107 import node_geometry
from weld_pipeline.sjg_relative_forest_runtime_1139 import relative_link_blocks as _original_blocks


def resolved_positive_gap(a,b,length_scale_mm):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    assert a.shape[-1]==b.shape[-1]==3
    scale=np.maximum(1.,np.maximum(np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)))
    scale=np.maximum(scale,np.asarray(length_scale_mm,np.float64))
    assert np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(scale).all()
    distance=np.linalg.norm(a-b,axis=-1)
    tolerance=64*np.finfo(np.float64).eps*scale
    return distance>tolerance,tolerance


def guarded_relative_link_blocks(nodes,case,gap_resolver=None):
    # Captured original iterator permits callers to install this adapter
    # without recursively wrapping the replacement.
    graph=case['graph'];geometry=[node_geometry(n,graph)[0] for n in nodes]
    ends=np.asarray(geometry);lengths=np.asarray([graph.curve(graph.profile(n['fragment'])).length for n in nodes])
    for links,features,legal in _original_blocks(nodes,case,gap_resolver):
        a,b=links[:,0],links[:,1]
        positive,_=resolved_positive_gap(ends[a,1],ends[b,0],np.maximum(lengths[a],lengths[b]))
        changed=legal[:,2]&~positive
        if changed.any():
            legal=legal.copy();features=features.copy();legal[changed,2]=False;features[changed,31]=0.
        keep=legal.any(1)
        if keep.any():yield links[keep],features[keep],legal[keep]
