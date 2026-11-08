"""Periodic native closed-curve observations; geometry and weld decisions unchanged."""
import inspect,types
import numpy as np
from scipy.spatial import cKDTree
from .curveonly_01a1ac7c_primitive_output_v219 import *
from . import curveonly_01a1ac7c_primitive_output_v219 as previous
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_local_intervals_v95 as local
from . import curveonly_01a1ac7c_exact_mesh_index_v110 as finite
from . import curveonly_01a1ac7c_material_sections_v124 as volume

NAMES=tuple(cad.NAMES[j] for j in local.RAW_COLUMNS)+tuple(finite.NAMES)+tuple('solid_section_'+x for x in volume.NAMES)+( 
    'native_fraction_from_start','native_fraction_to_end','native_closed','native_log_length')
assert len(NAMES)==111


def tangent_curvature(points,station,periodic=True):
    p=np.asarray(points,float);s=np.asarray(station,float)
    closed=len(p)>3 and np.linalg.norm(p[0]-p[-1])<=1e-6
    if periodic and closed:
        u=p[:-1];t=np.roll(u,-1,axis=0)-np.roll(u,1,axis=0)
        t/=np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-12)
        ds=np.linalg.norm(np.roll(u,-1,axis=0)-u,axis=1)
        curvature=np.linalg.norm(np.roll(t,-1,axis=0)-np.roll(t,1,axis=0),axis=1)/np.maximum(ds+np.roll(ds,1),1e-8)
        return np.vstack([t,t[0]]),np.r_[curvature,curvature[0]]
    t=np.gradient(p,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-12)
    curvature=np.linalg.norm(np.gradient(t,axis=0),axis=1)/np.maximum(np.gradient(s),1e-8)
    return t,curvature


def profile(points,base,tree,raw,periodic=True):
    p=np.asarray(points,float);s=cad.arclength(p);t,curvature=tangent_curvature(p,s,periodic)
    fit=cad.circle_fit(p)
    n=np.asarray(fit['normal']) if fit else np.linalg.svd(p-p.mean(0),full_matrices=False)[2][-1]
    pivot=int(np.flatnonzero(np.abs(n)>=np.max(np.abs(n))-1e-10)[0]);n=n*(-1 if n[pivot]<0 else 1)
    normal=np.broadcast_to(n,p.shape);b=np.cross(t,normal);b/=np.maximum(np.linalg.norm(b,axis=1)[:,None],1e-12)
    x=np.asarray(raw,np.float32).copy();x[:,5]=abs(normal@base);x[:,6]=abs(t@base);x[:,11]=curvature
    dist=tree.query(p,k=8,workers=1,distance_upper_bound=30.)[0];x[:,19]=np.minimum(dist[:,0],30.);x[:,20]=np.minimum(dist[:,-1],30.)
    column=21
    for delta in (2.,5.):
        for direction in (normal,-normal,b,-b):
            x[:,column]=np.minimum(tree.query(p+delta*direction,workers=1,distance_upper_bound=30.)[0],30.);column+=1
    x[:,29]=np.minimum(tree.query(p-3*t,workers=1,distance_upper_bound=30.)[0],30.)
    x[:,30]=np.minimum(tree.query(p+3*t,workers=1,distance_upper_bound=30.)[0],30.)
    return x


def input_curves(record):
    """Same existing raw pipeline with one changed tangent/curvature calculation."""
    original=cad.input_curves;had_helper=hasattr(cad,'_curveonly_periodic_profile');old_helper=getattr(cad,'_curveonly_periodic_profile',None)
    source=inspect.getsource(original)
    a='tangent=np.gradient(p,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-12)'
    b='tangent,curvature=_curveonly_periodic_profile(p,s)'
    assert source.count(a)==1;source=source.replace(a,b)
    a='curvature=np.linalg.norm(np.gradient(tangent,axis=0),axis=1)/np.maximum(np.gradient(s),1e-8)'
    assert source.count(a)==1;source=source.replace(a,'# periodic curvature computed with the tangent above')
    namespace=dict(cad.__dict__,_curveonly_periodic_profile=tangent_curvature)
    exec(compile(source,__file__,'exec'),namespace)
    cad.input_curves=namespace['input_curves'];cad._curveonly_periodic_profile=tangent_curvature
    try:return previous.input_curves(record)
    finally:
        cad.input_curves=original
        if had_helper:cad._curveonly_periodic_profile=old_helper
        else:del cad._curveonly_periodic_profile


def repair(record,features,packet):
    """Reuse unchanged inputs only after replay of the old physical profile."""
    selected=[ci for ci,c in enumerate(packet['candidates']) if not c.get('source_material_path') and
        c.get('source_curve_count')==1 and len(c.get('original_native_members',[]))==1 and
        np.linalg.norm(np.asarray(c['points'])[0]-c['points'][-1])<=1e-6 and len(c['stations'])>3]
    if not selected:
        return np.asarray(features).copy(),packet,dict(changed_candidates=[],changed_values=0,
            original_geometry_and_nonprofile_columns_unchanged=True,cached_precision_not_cold=True)
    xyz=cad.read_xyz(record['raw']['path']);tree=cKDTree(xyz,balanced_tree=False)
    output=np.asarray(features).copy();records=[];rawcols=list(local.RAW_COLUMNS)
    for ci in selected:
        c=packet['candidates'][ci];lo,hi=packet['offsets'][ci:ci+2];old=features[lo:hi,:34]
        before=profile(c['stations'],np.asarray(packet['base_axis']),tree,old,False)
        error=float(np.abs(before[:,rawcols]-old[:,rawcols]).max())
        descendants=[i for i,q in enumerate(packet['candidates']) if any(x['curve']==ci for x in q.get('source_material_path',[]))]
        if error>1e-5 or descendants:
            fresh,newpacket=input_curves(record)
            assert packet['offsets']==newpacket['offsets']
            assert all(all(a[k]==b[k] for k in ('points','stations','station_s_mm')) for a,b in zip(packet['candidates'],newpacket['candidates']))
            return fresh,newpacket,dict(full_raw_fallback=True,reason='CACHE_PROFILE_OR_DESCENDANT_NOT_CERTIFIED',candidate=ci,old_profile_error=error,descendants=descendants,cached_precision_not_cold=False)
        after=profile(c['stations'],np.asarray(packet['base_axis']),tree,old,True)
        output[lo:hi,:34]=after
        records.append(dict(candidate=ci,rows=hi-lo,legacy_profile_max_error=error,
            changed_values=int(np.count_nonzero(after!=old)),max_delta=float(np.abs(after-old).max())))
    assert np.array_equal(output[:,34:],features[:,34:])
    return output,packet,dict(changed_candidates=records,changed_values=int(np.count_nonzero(output!=features)),
        finite12_and_solid74_unchanged=True,original_geometry_and_nonprofile_columns_unchanged=True,
        cached_precision_not_cold=True,no_should_weld_or_endpoint_change=True)
