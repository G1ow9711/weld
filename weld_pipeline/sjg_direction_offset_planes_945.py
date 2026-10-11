"""Recover repeated plane offsets without requiring local planar normal seeds."""
import numpy as np
from weld_pipeline.sjg_observed_cylinders_919 import basis_for

def recover(xyz,selected,progress=None,maximum_axes=16):
    keys=np.floor(xyz-xyz.min(0)).astype(np.int32);_,ids=np.unique(keys,axis=0,return_index=True)
    sample=xyz[ids];axes=[];new=[];stats=[]
    for face in selected:
        n=np.asarray(face['normal'],float)
        if any(abs(n@v)>np.cos(np.deg2rad(.1)) for v in axes):continue
        axes.append(n)
        if len(axes)==maximum_axes:break
    for ai,axis in enumerate(axes):
        heights=sample@axis;bins,counts=np.unique(np.round(heights/.2).astype(np.int64),return_counts=True)
        lookup=dict(zip(bins.tolist(),counts.tolist()));peaks=[]
        for b,c in zip(bins,counts):
            if c>=32 and c>=lookup.get(b-1,0) and c>=lookup.get(b+1,0):peaks.append(int(b))
        added=0
        for b in peaks:
            offset=float(np.median(heights[abs(heights-b*.2)<=.2]));n=axis.copy()
            if any(abs(n@f['normal'])>1-1e-8 and abs(offset-np.sign(n@f['normal'])*f['offset'])<.2 for f in selected+new):continue
            valid=True
            for _ in range(3):
                near=sample[abs(sample@n-offset)<=.2]
                if len(near)<32:valid=False;break
                center=near.mean(0);delta=near-center;values,vectors=np.linalg.eigh(delta.T@delta/len(near))
                if values[1]<.01:valid=False;break
                n=vectors[:,0]
                if n@axis<0:n=-n
                if n@axis<np.cos(np.deg2rad(.1)):valid=False;break
                offset=float(center@n)
            if not valid:continue
            if any(abs(n@f['normal'])>1-1e-8 and abs(offset-np.sign(n@f['normal'])*f['offset'])<.2 for f in selected+new):continue
            point_ids=np.flatnonzero(abs(xyz@n-offset)<=.2)
            new.append(dict(normal=n,offset=offset,center=np.asarray(xyz[point_ids]).mean(0),basis=basis_for(n),point_ids=point_ids,
                proposal_branch='XYZ_shared_direction_independent_offset',material_id=None,material_status='UNKNOWN_NOT_INFERRED',
                unresolved_subsampling_holes=True,canonical_support=len(near)))
            added+=1
        stats.append(dict(axis=axis.tolist(),peaks=len(peaks),new_planes=added))
        if progress:progress('XYZ_OFFSET_AXIS',ai+1,len(axes),'PEAKS',len(peaks),'NEW_PLANES',added)
    return new,stats

def controls():
    # Sparse parallel plates. Direction is supplied as an observed hypothesis;
    # offset discovery must retain both without local-normal supervision.
    y,z=np.meshgrid(np.arange(0.,601.,20.),np.arange(0.,101.,20.))
    xyz=np.concatenate([np.c_[np.full(y.size,x),y.ravel(),z.ravel()] for x in (0.,10.)])
    seed=[dict(normal=np.array([1.,0.,0.]),offset=1000.)]
    faces,_=recover(xyz,seed);offsets=sorted(f['offset'] for f in faces)
    assert len(offsets)==2 and np.allclose(offsets,[0,10])
    return dict(two_sparse_parallel_plate_offsets_recovered=True,STEP_or_label_reads=0)
