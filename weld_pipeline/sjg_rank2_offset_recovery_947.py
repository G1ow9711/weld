"""Reuse rank-two directional votes without the brittle planar-eigenvalue gate."""
import numpy as np
from weld_pipeline.sjg_observed_cylinders_919 import basis_for

def recover(xyz,selected,data,progress=None,selection='global'):
    sample=data['xyz'];axes=[];ranked=[];stats=[]
    for face in selected:
        n=np.asarray(face['normal'],float)
        if any(abs(n@v)>np.cos(np.deg2rad(.1)) for v in axes):continue
        axes.append(n)
        if len(axes)==16:break
    for ai,axis in enumerate(axes):
        eligible=data['rank2']&(abs(data['normal']@axis)>=np.cos(np.deg2rad(2.)))
        heights=sample[eligible]@axis
        bins,counts=np.unique(np.round(heights/.2).astype(np.int64),return_counts=True);lookup=dict(zip(bins.tolist(),counts.tolist()))
        added=0
        for b,c in zip(bins,counts):
            if c<32 or c<lookup.get(b-1,0) or c<lookup.get(b+1,0):continue
            offset=float(np.median(heights[abs(heights-b*.2)<=.2]));n=axis.copy()
            if any(abs(n@f['normal'])>1-1e-8 and abs(offset-np.sign(n@f['normal'])*f['offset'])<.2 for f in selected+ranked):continue
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
            if any(abs(n@f['normal'])>1-1e-8 and abs(offset-np.sign(n@f['normal'])*f['offset'])<.2 for f in selected+ranked):continue
            ranked.append(dict(normal=n,offset=offset,canonical_support=len(near),directional_votes=int(c),direction_family=ai));added+=1
        stats.append(dict(axis=axis.tolist(),rank2_directional_points=int(eligible.sum()),new_hypotheses=added))
        if progress:progress('RANK2_OFFSET_AXIS',ai+1,len(axes),'NEW_HYPOTHESES',added)
    ranked.sort(key=lambda f:(-f['directional_votes'],-f['canonical_support']));faces=[]
    if selection=='direction_balanced':
        groups=[[f for f in ranked if f['direction_family']==i] for i in range(len(axes))]
        chosen=[];depth=0
        while len(chosen)<128 and any(len(g)>depth for g in groups):
            for group in groups:
                if len(group)>depth and len(chosen)<128:chosen.append(group[depth])
            depth+=1
    else:
        assert selection=='global';chosen=ranked[:128]
    for f in chosen:
        ids=np.flatnonzero(abs(xyz@f['normal']-f['offset'])<=.2)
        faces.append(dict(**f,point_ids=ids,center=np.asarray(xyz[ids]).mean(0),basis=basis_for(f['normal']),
            proposal_branch='rank2_direction_without_planarity_gate',material_id=None,material_status='UNKNOWN_NOT_INFERRED',
            unresolved_subsampling_holes=True))
    return faces,stats
