"""Pure-XYZ surface features with separately constructed offline STEP labels."""
import numpy as np
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_geometry_776 import read, digest
from weld_pipeline.sjg_prepare_776 import ROOT, OUT
from weld_pipeline.sjg_xyz_batch_791 import normal_cloud

GEOMETRY=OUT/'run16'


def input_features(si,inventory):
    folder=GEOMETRY/f'source_{si:02d}';freeze=read(folder/'input_freeze.json')
    for name,sha in freeze['files'].items():assert digest(folder/name)==sha
    raw=ROOT/inventory[si]['raw']['original_pcd_path']
    assert digest(raw)==freeze['original_PCD_sha256']
    xyz=read_xyz_pcd(raw)*1000.;normals,_,planar=normal_cloud(xyz)
    faces=read(folder/'surfaces.json')
    centers=np.array([f['center'] for f in faces]);directions=np.array([f['normal'] for f in faces])
    features=[]
    with np.load(folder/'surface_support.npz',allow_pickle=False) as arrays:
        for fi,f in enumerate(faces):
            ids=arrays[f['point_ids_key']];triangles=arrays[f['triangles_key']]
            p=xyz[ids];normal=np.asarray(f['normal']);center=np.asarray(f['center'])
            uv=(p-center)@f['basis'];extent=np.sort(np.ptp(uv,axis=0))
            variance=np.sort(np.linalg.eigvalsh(np.cov(uv.T)))
            tri=p[triangles];area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()
            rectangle=max(float(np.prod(extent)),1e-8)
            agreement=np.abs(normals[ids]@normal)
            residual=np.abs(p@normal-f['offset'])
            degree=np.bincount(triangles.ravel(),minlength=len(p))
            parallel=(np.abs(directions@normal)>.9999);parallel[fi]=False
            distance=np.sort(np.abs((centers[parallel]-center)@normal))
            nearest=np.full(3,1000.);nearest[:min(3,len(distance))]=distance[:3]
            radius=np.asarray(f['adaptive_radius_p50_p95_mm'])
            vector=np.r_[abs(normal[2]),np.log1p(len(ids))/10,np.log1p(len(triangles))/10,
                np.log1p(extent)/10,np.log1p(variance)/20,np.log1p(area)/20,
                np.clip(area/rectangle,0,2),np.log1p(len(ids)/max(area,1e-6)),
                residual.mean()/.2,np.quantile(residual,.95)/.2,residual.max()/.2,
                agreement.mean(),np.quantile(agreement,[.1,.5,.9]),
                np.mean(agreement>np.cos(np.deg2rad(5))),np.mean(agreement>np.cos(np.deg2rad(15))),
                np.mean(planar[ids]),np.mean((agreement>.98)&planar[ids]),
                np.mean(degree<3),np.mean(degree>=6),np.quantile(degree,[.5,.9])/10,
                np.log1p(radius)/10,np.log1p(f['sample_spacing_mm'])/10,
                np.log1p(nearest)/10, min(len(distance),100)/100]
            assert np.isfinite(vector).all()
            features.append(vector)
            if fi%100==0:print('SURFACE_FEATURES',si,fi+1,'/',len(faces),flush=True)
    return np.asarray(features,np.float32),freeze


def inside_finite_face(points,face,tolerance=.2):
    normal=np.asarray(face['normal']);axis=np.eye(3)[np.argmin(abs(normal))]
    u=np.cross(normal,axis);u/=np.linalg.norm(u);v=np.cross(normal,u)
    center=np.asarray(face['center']);relative=points-center
    x,y=relative@u,relative@v
    inside=np.zeros(len(points),bool);boundary=np.zeros(len(points),bool)
    for line in face['lines']:
        line=np.asarray(line)-center;a=line@u;b=line@v
        for i in range(len(line)-1):
            dx,dy=a[i+1]-a[i],b[i+1]-b[i]
            if abs(dy)>1e-12:
                crossing=(b[i]>y)!=(b[i+1]>y)
                inside ^= crossing&(x<a[i]+(y-b[i])*dx/dy)
            squared=dx*dx+dy*dy
            if squared>1e-16:
                t=np.clip(((x-a[i])*dx+(y-b[i])*dy)/squared,0,1)
                boundary |= (x-a[i]-t*dx)**2+(y-b[i]-t*dy)**2<=tolerance*tolerance
    return inside|boundary


def offline_surface_labels(si,inventory):
    """A reference, never an inference input; mixed hypotheses stay unknown."""
    reference=OUT/f'source_{si:02d}/geometry.json'
    if not reference.exists():return None
    binding=read(OUT/f'source_{si:02d}/input_freeze.json')
    assert digest(reference)==binding['geometry_sha256']
    truth=read(reference)['faces'];folder=GEOMETRY/f'source_{si:02d}'
    faces=read(folder/'surfaces.json')
    raw=ROOT/inventory[si]['raw']['original_pcd_path'];xyz=read_xyz_pcd(raw)*1000.
    labels=[];fractions=[];ambiguous=[]
    with np.load(folder/'surface_support.npz',allow_pickle=False) as arrays:
        for i,f in enumerate(faces):
            ids=arrays[f['point_ids_key']];points=xyz[ids]
            supported=np.zeros(len(points),bool);owners=[]
            for gt in truth:
                normal=np.asarray(gt['normal'])
                if abs(normal@f['normal'])<np.cos(np.deg2rad(.1)):continue
                if abs((np.asarray(f['center'])-gt['center'])@normal)>.2:continue
                near=np.flatnonzero((np.abs((points-gt['center'])@normal)<=.2)&
                    np.all(points>=np.asarray(gt['lo'])-.2,axis=1)&np.all(points<=np.asarray(gt['hi'])+.2,axis=1))
                if not len(near):continue
                match=inside_finite_face(points[near],gt)
                supported[near[match]]=True
                if match.any():owners.append(int(gt['volume']))
            fraction=float(supported.mean())
            label=1. if fraction>=.9 else (0. if fraction<=.1 else -1.)
            labels.append(label);fractions.append(fraction)
            ambiguous.append(len(set(owners))>1)
    return dict(labels=np.asarray(labels,np.float32),finite_reference_fraction=np.asarray(fractions,np.float32),
        multiple_material_owners=np.asarray(ambiguous,bool),reference_sha256=digest(reference),
        mixed_support_unknown_count=sum(v<0 for v in labels),single_material_truth_not_assumed=True)
