"""Bounded finite material contacts; no welding decision or labels here."""
import time,itertools
import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from . import curveonly_01a1ac7c as cad

def tight_box(solid):
    cad.BRepMesh_IncrementalMesh(solid,.3,False,.2,False);points=[]
    for f in cad.items(solid,cad.TopAbs_FACE):
        loc=cad.TopLoc_Location();tri=cad.BRep_Tool.Triangulation_s(cad.TopoDS.Face_s(f),loc)
        if tri is not None:points.extend(cad.vec(tri.Node(i).Transformed(loc.Transformation()))for i in range(1,tri.NbNodes()+1))
    if not points:return cad.bbox(solid)
    p=np.array(points);return np.r_[p.min(0)-.3,p.max(0)+.3]

def curves(solids,transform):
    start=time.perf_counter();boxes=np.array([tight_box(s)for s in solids]);out=[];pairs=[];seen=set()
    for i,j in itertools.combinations(range(len(solids)),2):
        a,b=boxes[i],boxes[j]
        if np.any(a[:3]>b[3:]+1e-5)or np.any(b[:3]>a[3:]+1e-5):continue
        tick=time.perf_counter();section=BRepAlgoAPI_Section(solids[i],solids[j],False);section.SetRunParallel(False);section.Build();count=0
        if section.IsDone():
            for e in cad.items(section.Shape(),cad.TopAbs_EDGE):
                p=cad.curve_points(cad.TopoDS.Edge_s(e))
                if p is None or len(p)<3:continue
                p=p@transform[:3,:3].T+transform[:3,3];length=cad.arclength(p)[-1]
                if length<1e-5:continue
                samples=cad.resample(p,max(length/16.,.1));key=min(np.round(samples,3).tobytes(),np.round(samples[::-1],3).tobytes())
                if key in seen:continue
                seen.add(key);out.append(dict(points=p.tolist(),material=i,participating_materials=[i,j],other_distance=[0.,0.,0.],STEP=True,pose=True,material_extent=np.sort(cad.bbox(solids[i])[3:]-cad.bbox(solids[i])[:3]).tolist(),source='FINITE_MATERIAL_SECTION'));count+=1
        pairs.append(dict(materials=[i,j],done=section.IsDone(),curve_edges=count,seconds=time.perf_counter()-tick))
    return out,dict(material_instances=len(solids),intersecting_bbox_pairs=len(pairs),pairs=pairs,seconds=time.perf_counter()-start,labels_or_helper_weld_curves=False)
