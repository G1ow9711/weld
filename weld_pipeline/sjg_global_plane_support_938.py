"""Verify input-derived plane hypotheses against the complete observed cloud."""
import numpy as np
from weld_pipeline.sjg_rank_adaptive_mesh_936 import remesh
from weld_pipeline.sjg_observed_cylinders_919 import basis_for


def expand_planes(xyz,hypotheses,maximum=32,progress=None):
    keys=np.floor(xyz-xyz.min(0)).astype(np.int32);_,ids=np.unique(keys,axis=0,return_index=True)
    sample=xyz[ids];seen=set();ranked=[]
    for i,face in enumerate(hypotheses):
        normal=np.asarray(face['normal'],float);offset=float(face['offset'])
        sign=1 if normal[np.argmax(abs(normal))]>=0 else -1;normal=normal*sign;offset*=sign
        key=tuple(np.r_[np.round(normal/1e-5).astype(np.int64),round(offset/.05)])
        if key in seen:continue
        seen.add(key);near=np.flatnonzero(abs(sample@normal-offset)<=.2)
        if len(near)<32:continue
        points=sample[near];center=points.mean(0);delta=points-center
        values=np.linalg.eigvalsh(delta.T@delta/len(points))
        if values[1]<.01:continue
        # Extent alone rewards sparse accidental cuts across distant objects.
        # Use density-normalized observed inlier count as primary evidence.
        score=float(np.sqrt(max(0.,values[1]*values[2])))
        ranked.append(dict(source_hypothesis=i,normal=normal,offset=offset,score=score,canonical_support=len(near)))
        if progress and i%100==0:progress('GLOBAL_PLANE_VERIFY',i+1,len(hypotheses),'SUPPORTED',len(ranked))
    ranked.sort(key=lambda p:(-p['canonical_support'],-p['score'],p['source_hypothesis']))
    faces=[];selected=[]
    for hypothesis in ranked:
        normal=hypothesis['normal'];offset=hypothesis['offset']
        if any(abs(normal@f['normal'])>1-1e-8 and abs(offset-np.sign(normal@f['normal'])*f['offset'])<.2 for f in faces):continue
        ids=np.flatnonzero(abs(xyz@normal-offset)<=.2);center=xyz[ids].mean(0);basis=basis_for(normal)
        uv=(xyz[ids]-center)@basis;_,unique=np.unique(np.round(uv,5),axis=0,return_index=True);ids=ids[unique];uv=uv[unique]
        triangles,stats=remesh(uv)
        if not len(triangles):continue
        faces.append(dict(normal=normal,offset=offset,center=center,basis=basis,point_ids=ids,triangles=triangles,
            proposal_branch='global_observed_support',source_hypothesis=hypothesis['source_hypothesis'],
            material_id=None,material_status='UNKNOWN_NOT_INFERRED',unresolved_subsampling_holes=True))
        selected.append(dict(source_hypothesis=hypothesis['source_hypothesis'],spatial_score=hypothesis['score'],
                             canonical_support=hypothesis['canonical_support'],original_support=len(ids),mesh=stats))
        if progress:progress('GLOBAL_PLANE_RECOVERED',len(faces),maximum,'HYPOTHESIS',hypothesis['source_hypothesis'],'POINTS',len(ids))
        if len(faces)>=maximum:break
    return faces,dict(input_hypotheses=len(hypotheses),unique_hypotheses=len(seen),supported_hypotheses=len(ranked),
        ranking='canonical_inlier_count_then_extent',
        maximum_planes=maximum,selection_budget_reached=len(faces)==maximum,selected=selected)
