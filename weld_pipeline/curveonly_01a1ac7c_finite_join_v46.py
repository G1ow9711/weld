"""Local finite-face witnesses for input curve continuation candidates.

This checks local material support only. It never declares a weld, deletes a
process cut, or certifies full-curve contact or side identity. Nominal material
clearance is measured and preserved; it is not labeled as a weld gap.
"""
import time
import numpy as np
from . import curveonly_01a1ac7c as cad
from .curveonly_01a1ac7c_sampler_v5 import TOLERANCE_MM


class Context:
    def __init__(self,solids,transform,faces=None,faceboxes=None):
        self.solids=solids;self.transform=np.asarray(transform,float)
        self.faces=faces if faces is not None else [list(cad.items(s,cad.TopAbs_FACE)) for s in solids]
        self.boxes=faceboxes if faceboxes is not None else [np.asarray([cad.bbox(f) for f in fs]) for fs in self.faces]
        self.calls=0;self.broadphase_faces=0

    def common_face(self,world,excluded):
        q=(np.asarray(world,float)-self.transform[:3,3])@self.transform[:3,:3]
        lo,hi=q.min(0),q.max(0)
        for mi,faces in enumerate(self.faces):
            if mi in excluded:continue
            boxes=np.asarray(self.boxes[mi])
            if not len(boxes):continue
            valid=np.flatnonzero(np.all(boxes[:,:3]<=lo+TOLERANCE_MM,axis=1)&np.all(boxes[:,3:]>=hi-TOLERANCE_MM,axis=1))
            self.broadphase_faces+=len(valid)
            for fi in valid:
                distances=[]
                for p in q:
                    d=cad.finite_distance(faces[int(fi)],p);self.calls+=1;distances.append(d)
                    if d>TOLERANCE_MM:break
                if len(distances)==len(q) and max(distances)<=TOLERANCE_MM:
                    face=cad.TopoDS.Face_s(faces[int(fi)])
                    return dict(material=mi,face=int(fi),distances_mm=distances,
                        maximum_distance_mm=max(distances),surface_type=str(cad.BRepAdaptor_Surface(face,True).GetType()),
                        sample_world=np.asarray(world,float).tolist(),tolerance_mm=TOLERANCE_MM,
                        status='LOCAL_FINITE_SUPPORT_NOT_SHOULD_WELD_OR_FULL_CONTACT')
        return None


def near_endpoint(candidate,end):
    p=np.asarray(candidate['points'],float)
    if end:p=p[::-1]
    s=cad.arclength(p)
    return [p[int(np.argmin(np.abs(s-value)))] for value in (0.,5.,15.)]


def witness(a,b,ae,be,context):
    first=near_endpoint(a,ae);second=near_endpoint(b,be)
    points=np.asarray([*first,.5*(first[0]+second[0]),*second])
    mi,mj=a.get('material',-1),b.get('material',-1)
    if mi<0 or mj<0 or mi==mj:return None
    result=context.common_face(points,{mi,mj})
    if result is None:return None
    return dict(common_finite_face=result,source_materials=[mi,mj],
        nominal_endpoint_clearance_mm=float(np.linalg.norm(first[0]-second[0])),
        original_boundaries_retained=True,weld_and_process_continuity_not_accepted=True)
