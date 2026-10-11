"""Straight portions of point-observed rounded hole boundaries, for mixed training."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.arc_surface_candidates import arc_surface_candidates
from weld_pipeline.circle_candidates import cylinder_columns
from weld_pipeline.typed_sequences import OFFSETS,DEPTHS


def tangent_pairs(candidates):
    ends=[]
    for ci,c in enumerate(candidates):
        points=np.asarray(c['segments'][0]['points'])
        for p in points[[0,-1]]:
            radial=p[:2]-c['arc']['center_xy_mm'];radial/=np.linalg.norm(radial)
            ends.append((ci,p,radial))
    possible=[]
    for i,(ci,a,na) in enumerate(ends):
        for j in range(i+1,len(ends)):
            cj,b,nb=ends[j]
            if ci==cj or abs(a[2]-b[2])>.01 or na@nb<.995:continue
            delta=b-a;length=float(np.linalg.norm(delta))
            if length<3:continue
            u=delta/length
            if abs(u[:2]@na)>.01 or abs(u[:2]@nb)>.01:continue
            possible.append((length,i,j))
    used=set();pairs=[]
    for length,i,j in sorted(possible):
        if i in used or j in used:continue
        used.update([i,j]);ci,a,na=ends[i];cj,b,nb=ends[j]
        pairs.append({'a':a,'b':b,'normal_xy':(na+nb)/np.linalg.norm(na+nb),
                      'z_range_mm':candidates[ci]['arc']['z_range_mm'],'arc_indices':[ci,cj]})
    return pairs


def make_tangent_sequences(xyz,step_mm=2.,candidates_and_stats=None):
    xyz=np.asarray(xyz,float);arcs,stats=arc_surface_candidates(xyz) if candidates_and_stats is None else candidates_and_stats;pairs=tangent_pairs(arcs)
    if not pairs:return [],{'pairs':0,'sequences':0}
    anchors=cylinder_columns(xyz);tree=cKDTree(xyz);center=(xyz.min(0)+xyz.max(0))/2;extent=np.ptp(xyz,axis=0)
    sequences=[];rejected=[]
    def query(tree,q):
        d=tree.query(q.reshape(-1,3),workers=2,distance_upper_bound=50.)[0]
        return (np.minimum(d,50)/50).reshape(q.shape[:-1])
    for pi,pair in enumerate(pairs):
        a,b=pair['a'],pair['b'];length=np.linalg.norm(b-a);u=(b-a)/length
        normal=pair['normal_xy'];offset=float(normal@a[:2])
        positions=(anchors-a[:2])@u[:2];distance=anchors@normal-offset
        near=(positions>=-2)&(positions<=length+2)&(np.abs(distance)<=5)
        bins,counts=np.unique(np.round(distance[near],1),return_counts=True)
        if not len(bins) or counts.max()<3:
            rejected.append({'pair':pi,'reason':'no supporting upright plane within 5 mm'});continue
        mode=bins[np.argmax(counts)];local=distance[near]
        wall_offset=offset+float(np.median(local[np.abs(local-mode)<=.06]))
        upright=xyz[np.abs(xyz[:,:2]@normal-wall_offset)<.15]
        plane=xyz[np.abs(xyz[:,2]-a[2])<.15]
        if not len(upright) or not len(plane):continue
        t=np.arange(-20,length+20+step_mm/2,step_mm);q=a+t[:,None]*u
        n1=np.r_[normal,0.];zlow,zhigh=pair['z_range_mm']
        sign=-1. if a[2]-zlow>=zhigh-a[2] else 1.;n2=np.array([0.,0.,sign])
        offsets=np.array([n1*x+n2*y for y in DEPTHS for x in OFFSETS])
        base=query(tree,q[:,None]+offsets[None])
        extra=np.column_stack([(q-center)/1000,np.tile(extent/1000,(len(q),1)),
            (t-t[0])/1000,(t[-1]-t)/1000,np.tile(np.r_[u,n1,n2],(len(q),1)),np.zeros(len(q)),q/10000])
        first=query(cKDTree(upright),q[:,None]+DEPTHS[None,:,None]*n2)
        second=query(cKDTree(plane),q[:,None]+OFFSETS[None,:,None]*n1)
        features=np.column_stack([base,extra,first,second]).astype(np.float32)
        sequences.append({'xyz':q.astype(np.float32),'features':features,'direction':u,
                          'sample_step_mm':step_mm,'type':'horizontal','candidate_id':f'arc_tangent_{pi}',
                          'wall_offset_from_hole_boundary_mm':float(wall_offset-offset)})
    return sequences,{'pairs':len(pairs),'sequences':len(sequences),'rejected':rejected,
                       'scope':'Point-only rounded-boundary tangents; unclassified weld candidates.'}
