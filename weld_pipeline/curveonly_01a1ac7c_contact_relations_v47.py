"""Finite local support relations for candidate paths, not weld approval."""
import numpy as np
from . import curveonly_01a1ac7c as cad
from .curveonly_01a1ac7c_finite_join_v46 import Context as Base,near_endpoint,TOLERANCE_MM

# Same3mm finite other-material observation used by original feature15.
# This is a candidate evidence radius, not an allowed weld-gap business rule.
OBSERVED_NEAR_MM=3.


class Context(Base):
    def face(self,world,allowed=None,excluded=(),limit=TOLERANCE_MM):
        q=(np.asarray(world,float)-self.transform[:3,3])@self.transform[:3,:3];lo,hi=q.min(0),q.max(0)
        for mi,faces in enumerate(self.faces):
            if mi in excluded or (allowed is not None and mi not in allowed):continue
            boxes=np.asarray(self.boxes[mi])
            if not len(boxes):continue
            valid=np.flatnonzero(np.all(boxes[:,:3]<=lo+limit,axis=1)&np.all(boxes[:,3:]>=hi-limit,axis=1))
            self.broadphase_faces+=len(valid)
            for fi in valid:
                distances=[]
                for p in q:
                    d=cad.finite_distance(faces[int(fi)],p);self.calls+=1;distances.append(d)
                    if d>limit:break
                if len(distances)==len(q) and max(distances)<=limit:
                    f=cad.TopoDS.Face_s(faces[int(fi)])
                    return dict(material=mi,face=int(fi),distances_mm=distances,maximum_distance_mm=max(distances),
                        surface_type=str(cad.BRepAdaptor_Surface(f,True).GetType()),sample_world=np.asarray(world,float).tolist(),
                        evidence_radius_mm=limit,not_weld_gap_allowance=True,
                        status='LOCAL_FINITE_CANDIDATE_SUPPORT_ONLY')
        return None


def relation(a,b,ae,be,context,include_near=True):
    mi,mj=a.get('material',-1),b.get('material',-1)
    if mi<0 or mj<0 or mi==mj:return None
    qa,qb=np.asarray(near_endpoint(a,ae)),np.asarray(near_endpoint(b,be));mid=.5*(qa[0]+qb[0])
    for limit in ((TOLERANCE_MM,OBSERVED_NEAR_MM) if include_near else (TOLERANCE_MM,)):
        left=context.face(qa,allowed={mj},limit=limit)
        right=context.face(qb,allowed={mi},limit=limit) if left is not None else None
        if left is not None and right is not None:
            return dict(kind='MUTUAL_FINITE_MATERIAL_PAIR',source_materials=[mi,mj],evidence_radius_mm=limit,
                faces=[left,right],nominal_endpoint_clearance_mm=float(np.linalg.norm(qa[0]-qb[0])),
                original_materials_boundaries_and_clearance_retained=True,should_weld_and_process_continuity=False)
        common=context.face(np.vstack([qa,mid,qb]),excluded={mi,mj},limit=limit)
        if common is not None:
            return dict(kind='SHARED_THIRD_FINITE_FACE',source_materials=[mi,mj],evidence_radius_mm=limit,
                faces=[common],nominal_endpoint_clearance_mm=float(np.linalg.norm(qa[0]-qb[0])),
                original_materials_boundaries_and_clearance_retained=True,should_weld_and_process_continuity=False)
    return None
