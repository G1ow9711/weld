"""Offline finite STEP contact graphs for experiment A, never deployment.

Trimmed face intervals come from native boundary curves and native inside
queries. The graph constructor has no weld-label argument. Raw XYZ patches
are sampled independently; any unavailable geometry remains a reported miss.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text("utf-8-sig"))
def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

def boundary_polyline(tag):
    import gmsh
    lo, hi = gmsh.model.getParametrizationBounds(1, tag)
    def value(t): return np.asarray(gmsh.model.getValue(1, tag, [float(t)]))
    def split(a, b, pa, pb, depth):
        qs = [a+(b-a)*q for q in (.25,.5,.75)]
        pts = [value(q) for q in qs]
        d = pb-pa
        error = [np.linalg.norm(p-(pa+np.clip((p-pa)@d/max(d@d,1e-20),0,1)*d)) for p in pts]
        if max(error) <= .05: return [pa,pb]
        if depth >= 18: raise RuntimeError("CAD boundary chord refinement did not converge")
        return split(a,qs[1],pa,pts[1],depth+1)[:-1]+split(qs[1],b,pts[1],pb,depth+1)
    return np.asarray(split(float(lo[0]),float(hi[0]),value(lo[0]),value(hi[0]),0))

def finite_faces(catalogue, transform):
    import gmsh
    rotation, translation = transform[:3,:3],transform[:3,3]
    curves = {}
    faces = []
    for volume in catalogue["volumes"]:
        for f in volume["planar_faces"]:
            tag = int(f["face"])
            boundaries = gmsh.model.getBoundary([(2,tag)], combined=False, oriented=False)
            lines = []
            for dim, curve in boundaries:
                if dim != 1: continue
                if curve not in curves: curves[curve] = boundary_polyline(curve)
                lines.append(curves[curve])
            if not lines: continue
            raw = np.concatenate(lines)
            world = raw@rotation.T+translation
            n = rotation@np.asarray(f["parametric_normal"])
            n /= np.linalg.norm(n)
            center = rotation@np.asarray(f["center_mm"])+translation
            faces.append(dict(tag=tag,volume=int(volume["volume"]),normal=n,center=center,
                              area=float(f["area_mm2"]),lo=world.min(0),hi=world.max(0),
                              lines=[x@rotation.T+translation for x in lines]))
    return faces

def line_intervals(face, origin, direction, transform):
    import gmsh
    side = np.cross(face["normal"],direction)
    cuts = []
    for line in face["lines"]:
        rel = line-origin
        tt, ss = rel@direction, rel@side
        for k in range(len(line)-1):
            a,b = float(ss[k]),float(ss[k+1])
            if abs(a)<1e-6: cuts.append(float(tt[k]))
            if abs(b)<1e-6: cuts.append(float(tt[k+1]))
            if a*b < 0: cuts.append(float(tt[k]+(tt[k+1]-tt[k])*a/(a-b)))
    if not cuts: return []
    cuts = sorted(cuts)
    values = [cuts[0]]
    for v in cuts[1:]:
        if v-values[-1] > 1e-6: values.append(v)
    rotation,translation=transform[:3,:3],transform[:3,3]
    intervals=[]
    for lo,hi in zip(values[:-1],values[1:]):
        if hi-lo<.05: continue
        world=origin+(lo+hi)*.5*direction
        cad=(world-translation)@rotation
        if gmsh.model.isInside(2,face["tag"],cad.tolist()):
            if intervals and abs(intervals[-1][1]-lo)<1e-6: intervals[-1][1]=hi
            else: intervals.append([lo,hi])
    return intervals

def construct_contacts(faces, transform):
    contacts = {}
    for i,a in enumerate(faces):
        for j in range(i+1,len(faces)):
            b=faces[j]
            if np.any(np.maximum(a["lo"],b["lo"])>np.minimum(a["hi"],b["hi"])+.15): continue
            d=np.cross(a["normal"],b["normal"])
            if np.linalg.norm(d)<.15: continue
            d/=np.linalg.norm(d)
            if d[np.argmax(np.abs(d))]<0: d=-d
            normals=np.stack([a["normal"],b["normal"]])
            offsets=np.array([a["normal"]@a["center"],b["normal"]@b["center"]])
            origin=normals.T@np.linalg.solve(normals@normals.T,offsets)
            aa=line_intervals(a,origin,d,transform)
            bb=line_intervals(b,origin,d,transform)
            for alo,ahi in aa:
                for blo,bhi in bb:
                    lo,hi=max(alo,blo),min(ahi,bhi)
                    if hi-lo<1.: continue
                    p0,p1=origin+lo*d,origin+hi*d
                    # Geometric aliases remain explicit. Different material pairs
                    # retain separate nodes even at the same coordinates.
                    key=(tuple(sorted((a["volume"],b["volume"]))),
                         tuple(np.round(np.r_[p0,p1],3)))
                    alias=dict(faces=[i,j],materials=[a["volume"],b["volume"]])
                    if key in contacts:
                        contacts[key]["aliases"].append(alias)
                    else:
                        contacts[key]=dict(p0=p0,p1=p1,direction=d,length=float(hi-lo),
                            face_indices=[i,j],materials=alias["materials"],aliases=[alias])
        if i%50==0: print("FINITE_FACE_PAIRS",i+1,"/",len(faces),"contacts",len(contacts),flush=True)
    return list(contacts.values())

def graph_arrays(faces,contacts,xyz):
    if not contacts: raise RuntimeError("No finite contacts; retain source as representation failure")
    tree=cKDTree(xyz)
    node_features=[]; patches=[]; query_features=[]; ptr=[0]
    for c in contacts:
        a,b=[faces[i] for i in c["face_indices"]]
        center=(c["p0"]+c["p1"])*.5
        f=np.r_[c["direction"],np.log1p(c["length"])/10,
                a["normal"],b["normal"],a["normal"]@b["normal"],
                (a["hi"]-a["lo"])/1000,(b["hi"]-b["lo"])/1000,
                (a["center"]-center)/1000,(b["center"]-center)/1000,
                np.log1p([a["area"],b["area"]])/20,float(a["volume"]==b["volume"])]
        node_features.append(f)
        q=np.linspace(0,c["length"],max(2,int(np.ceil(c["length"]/2))+1))
        pos=c["p0"]+q[:,None]*c["direction"]
        distance,ids=tree.query(pos,k=min(16,len(xyz)),workers=2)
        patch=np.clip((xyz[ids]-pos[:,None,:])/50,-20,20).astype(np.float32)
        patches.append(patch)
        query_features.append(np.stack([q/c["length"],q/1000,(c["length"]-q)/1000,
                                       distance[:,0]/50],axis=1).astype(np.float32))
        c["query_step_mm"]=float(q[1]-q[0])
        ptr.append(ptr[-1]+len(q))
    # Relations distinguish shared face and shared material.
    edges=[]; edge_type=[]
    face_sets=[set(x for alias in c["aliases"] for x in alias["faces"]) for c in contacts]
    mat_sets=[set(c["materials"]) for c in contacts]
    for i in range(len(contacts)):
        for j in range(i+1,len(contacts)):
            typ=1 if face_sets[i]&face_sets[j] else (2 if mat_sets[i]&mat_sets[j] else 0)
            if typ:
                edges.extend([(i,j),(j,i)]);edge_type.extend([typ,typ])
    return dict(nodes=np.asarray(node_features,np.float32), patches=np.concatenate(patches),
                query=np.concatenate(query_features), ptr=np.asarray(ptr,np.int64),
                edge_index=np.asarray(edges,np.int64).reshape(-1,2).T,
                edge_type=np.asarray(edge_type,np.int64))

def serial(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value,list): return [serial(v) for v in value]
    return value
