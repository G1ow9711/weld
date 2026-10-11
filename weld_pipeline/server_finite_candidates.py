"""Pure XYZ finite contacts for the explicit Linux reconstruction branch.

No target, STEP, workbook, source identity or frozen prediction reads.
Plane and boundary-arc algorithms retain their original source modules.
"""
from pathlib import Path
import shutil
import numpy as np
from weld_pipeline import sjg_rank2_finite_planes_925 as planes
from weld_pipeline.sjg_polygonal_arc_extension_923 import face_arcs
from weld_pipeline.sjg_polygonal_arcs_921 import face_arcs as base_face_arcs

def guard():
    status={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmData:')):status[line.split(':')[0]]=int(line.split()[1])*1024
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapFree:'))}
    cg=Path('/sys/fs/cgroup');limit=(cg/'memory.max').read_text().strip();used=int((cg/'memory.current').read_text())
    remaining=int(limit)-used if limit!='max' else memory['MemAvailable']
    available=min(remaining,memory['MemAvailable'])
    assert available>=int(2.5*1024**3),('Linux conservative headroom guard',available)
    assert status.get('VmData',0)<=12*1024**3,('Linux VmData guard',status)
    assert shutil.disk_usage(Path(__file__).resolve().parents[1]).free>=5*1024**3
    return dict(available_physical_bytes=available,cgroup_current=used,cgroup_limit=limit,**status,metric_mapping='min(host MemAvailable,cgroup remaining); VmData conservative virtual data; not Windows commit equivalence')

def intervals(face,xyz,origin,direction,tolerance=1e-7):
    tri=xyz[face['point_ids']][face['triangles']]
    signed=(tri-origin)@np.cross(face['normal'],direction);q=(tri-origin)@direction
    mask=(signed.min(1)<=tolerance)&(signed.max(1)>=-tolerance);signed=signed[mask];q=q[mask]
    lower=np.full(len(q),np.inf);upper=np.full(len(q),-np.inf)
    for i,j in ((0,1),(1,2),(2,0)):
        at=np.abs(signed[:,i])<=tolerance
        lower[at]=np.minimum(lower[at],q[at,i]);upper[at]=np.maximum(upper[at],q[at,i])
        cross=signed[:,i]*signed[:,j]<0
        v=q[cross,i]+(q[cross,j]-q[cross,i])*signed[cross,i]/(signed[cross,i]-signed[cross,j])
        lower[cross]=np.minimum(lower[cross],v);upper[cross]=np.maximum(upper[cross],v)
    valid=np.isfinite(lower)&np.isfinite(upper)&(upper-lower>1e-6);merged=[]
    for lo,hi in sorted(zip(lower[valid],upper[valid])):
        if merged and lo<=merged[-1][1]+1e-5:merged[-1][1]=max(merged[-1][1],float(hi))
        else:merged.append([float(lo),float(hi)])
    return merged

def contacts(faces,xyz,progress=None):
    result=[]
    bounds=[(xyz[f['point_ids']].min(0),xyz[f['point_ids']].max(0)) for f in faces]
    for i,a in enumerate(faces):
        for j in range(i+1,len(faces)):
            if np.any(np.maximum(bounds[i][0],bounds[j][0])>np.minimum(bounds[i][1],bounds[j][1])+.4):continue
            b=faces[j];direction=np.cross(a['normal'],b['normal']);norm=np.linalg.norm(direction)
            if norm<.15:continue
            direction/=norm
            if direction[np.argmax(abs(direction))]<0:direction=-direction
            normals=np.stack([a['normal'],b['normal']]);origin=normals.T@np.linalg.solve(normals@normals.T,[a['offset'],b['offset']])
            tolerance=max(1e-6,float(a.get("max_plane_residual_mm",0))+float(b.get("max_plane_residual_mm",0)))
            for alo,ahi in intervals(a,xyz,origin,direction,tolerance):
                for blo,bhi in intervals(b,xyz,origin,direction,tolerance):
                    lo,hi=max(alo,blo),min(ahi,bhi)
                    if hi-lo<=1e-6:continue
                    result.append(dict(representation=dict(kind='line',endpoints_mm=[(origin+lo*direction).tolist(),(origin+hi*direction).tolist()]),face_indices=[i,j],length_mm=float(hi-lo),status='UNCLASSIFIED_FINITE_CONTACT',material_pair=None,weld_type=None))
        if progress:progress('CONTACT_FACE',i+1,len(faces),'CONTACTS',len(result))
    return result

def extract(xyz,progress,plane_method="rank2"):
    global planes
    if plane_method=="local":
        from weld_pipeline import sjg_local_xyz_planes_913 as planes
    elif plane_method=="global":
        from weld_pipeline import server_global_planes as planes
    elif plane_method!="rank2":raise ValueError(plane_method)
    guard()
    # Explicit Linux guard substitution; original 925 geometry stays unchanged.
    planes.check_resources=guard
    faces,stats=planes.extract_planes(xyz,progress=progress)
    lines=contacts(faces,xyz,progress);arcs=[];arc_stats=[]
    for i,face in enumerate(faces):
        base,base_audit=base_face_arcs(xyz,face)
        found,audit=face_arcs(xyz,face)
        found=base+found
        audit=dict(base=base_audit,extension=audit)
        for c in found:c['face_indices']=[i]
        arcs.extend(found);arc_stats.append(audit);progress('BOUNDARY_ARCS',i+1,len(faces),'ARCS',len(arcs))
    return faces,lines+arcs,dict(plane_method=plane_method,planes=stats,arc_faces=arc_stats,lines=len(lines),arcs=len(arcs),cylinder_branch_pending=True,material_identity_pending=True,model_fit=False)
