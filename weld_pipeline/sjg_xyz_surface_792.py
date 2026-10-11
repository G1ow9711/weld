"""Pure-XYZ finite planar support prototype; no material identity is invented.

Input-only generation finishes and hashes its outputs before any weld-target
evaluation. Triangulated support retains sampled holes instead of filling each
plane's bounding box. This is a representation diagnostic, not weld inference.
"""
import argparse
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree, Delaunay, QhullError
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_geometry_776 import read, write, digest, serial
from weld_pipeline.sjg_prepare_776 import ROOT, OUT
from weld_pipeline.sjg_train_776 import check_resources, utc


def normal_cloud(xyz):
    tree=cKDTree(xyz); distance,near=tree.query(xyz,k=min(12,len(xyz)),workers=2)
    cloud=xyz[near];center=cloud.mean(1);delta=cloud-center[:,None]
    covariance=np.einsum('nki,nkj->nij',delta,delta)/cloud.shape[1]
    eigen,vectors=np.linalg.eigh(covariance)
    planar=(eigen[:,0]/eigen[:,1].clip(1e-8)<.02)&(eigen[:,1]>1e-6)
    return vectors[:,:,0],center,planar


def extract_planes(xyz, maximum_planes=512):
    normals,centers,planar=normal_cloud(xyz)
    signs=np.sign(normals[np.arange(len(normals)),np.argmax(np.abs(normals),axis=1)])
    normals=normals*signs[:,None]
    keys,inverse,counts=np.unique(np.round(normals[planar]/.02).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    usable=normals[planar];families=[]
    for k in np.argsort(-counts):
        if counts[k]<32:continue
        cluster=usable[inverse==k]
        _,fine,fine_counts=np.unique(np.round(cluster/1e-5).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
        mode=int(fine_counts.argmax())
        if fine_counts[mode]<16:continue
        normal=np.median(cluster[fine==mode],axis=0);normal/=np.linalg.norm(normal)
        if not any(abs(normal@n)>1-1e-8 for n in families):families.append(normal)
    candidates=[]
    for normal in families:
        projection=xyz@normal;order=np.argsort(projection);values=projection[order]
        bins,counts=np.unique(np.round(values/.2).astype(np.int64),return_counts=True)
        lookup={int(b):int(n) for b,n in zip(bins,counts)}
        seeds=[]
        for b,count in zip(bins,counts):
            b=int(b);left=lookup.get(b-1,0);right=lookup.get(b+1,0)
            if count<32 or count<left or count<right:continue
            if count==left:continue
            lo=np.searchsorted(values,b*.2-.3);hi=np.searchsorted(values,b*.2+.3,side='right')
            center=float(np.median(values[lo:hi]))
            if any(abs(center-old)<.3 for old in seeds):continue
            ids=order[np.searchsorted(values,center-.2):np.searchsorted(values,center+.2,side='right')].copy()
            if len(ids)<32:continue
            candidates.append(dict(normal=normal,ids=ids));seeds.append(center)
    remaining=np.ones(len(xyz),bool);faces=[]
    for iteration in range(maximum_planes):
        if not candidates:break
        counts=np.array([remaining[c['ids']].sum() for c in candidates])
        index=int(counts.argmax())
        if counts[index]<32:break
        candidate=candidates.pop(index);support=xyz[candidate['ids']];center=support.mean(0)
        _,singular,v=np.linalg.svd(support-center,full_matrices=False)
        if singular[1]<.5:continue
        normal=v[-1]
        if normal[np.argmax(np.abs(normal))]<0:normal=-normal
        offset=float(center@normal)
        ids=np.flatnonzero(np.abs(xyz@normal-offset)<.2)
        basis=v[:2].T;uv=(xyz[ids]-center)@basis
        _,unique=np.unique(np.round(uv,5),axis=0,return_index=True)
        ids=ids[unique];uv=uv[unique]
        try:triangles=Delaunay(uv).simplices
        except QhullError:continue
        distance=cKDTree(uv).query(uv,k=min(24,len(uv)),workers=2)[0]
        nearest=distance[:,1];radius=distance[:,-1]
        triangle_uv=uv[triangles]
        keep=np.ones(len(triangles),bool)
        for i,j in ((0,1),(1,2),(2,0)):
            edge=np.linalg.norm(triangle_uv[:,i]-triangle_uv[:,j],axis=1)
            # Adaptive alpha support uses observed sampling only. Holes below
            # this local scale remain unresolved, never certified absent.
            limit=np.maximum(2.,1.5*np.minimum(radius[triangles[:,i]],radius[triangles[:,j]]))
            keep &= edge<=limit
        triangles=triangles[keep]
        if not len(triangles):continue
        full_ids=np.flatnonzero(np.abs(xyz@normal-offset)<.2)
        core_count=int(remaining[full_ids].sum());remaining[full_ids]=False
        faces.append(dict(normal=normal,center=center,offset=offset,basis=basis,point_ids=ids,
            triangles=triangles,sample_spacing_mm=float(np.median(nearest[nearest>1e-6])),
            adaptive_radius_p50_p95_mm=np.percentile(radius,[50,95]),
            support_rule='edge <= 1.5 * min(endpoint 24-neighbor radii), floor 2mm',
            unresolved_subsampling_holes=True,core_points=core_count,
            max_plane_residual_mm=float(np.abs(xyz[ids]@normal-offset).max()),
            material_id=None,material_status='UNKNOWN_NOT_INFERRED'))
        print('GLOBAL_XYZ_PLANE',len(faces),'support',len(ids),'triangles',len(triangles),
              'new_points',core_count,'remaining',int(remaining.sum()),flush=True)
        check_resources()
    return faces,dict(planar_seed_points=int(planar.sum()),remaining_unassigned_points=int(remaining.sum()),
        orientation_families=len(families),maximum_planes=maximum_planes,hit_plane_cap=len(faces)>=maximum_planes)


def intervals(face,xyz,origin,direction):
    points=xyz[face['point_ids']];tri=points[face['triangles']]
    side=np.cross(face['normal'],direction)
    signed=(tri-origin)@side;q=(tri-origin)@direction
    crossing=(signed.min(1)<=1e-7)&(signed.max(1)>=-1e-7)
    values=[]
    for ss,tt in zip(signed[crossing],q[crossing]):
        cuts=[]
        for i,j in ((0,1),(1,2),(2,0)):
            if abs(ss[i])<1e-7:cuts.append(float(tt[i]))
            if ss[i]*ss[j]<0:cuts.append(float(tt[i]+(tt[j]-tt[i])*ss[i]/(ss[i]-ss[j])))
        if len(cuts)>=2 and max(cuts)-min(cuts)>1e-6:values.append((min(cuts),max(cuts)))
    merged=[]
    for lo,hi in sorted(values):
        if merged and lo<=merged[-1][1]+1e-5:merged[-1][1]=max(merged[-1][1],hi)
        else:merged.append([lo,hi])
    return merged


def contacts_from_support(faces,xyz):
    contacts=[]
    for i,a in enumerate(faces):
        for j in range(i+1,len(faces)):
            b=faces[j];direction=np.cross(a['normal'],b['normal'])
            if np.linalg.norm(direction)<.15:continue
            direction/=np.linalg.norm(direction)
            if direction[np.argmax(np.abs(direction))]<0:direction=-direction
            normals=np.stack([a['normal'],b['normal']]);offset=np.array([a['offset'],b['offset']])
            origin=normals.T@np.linalg.solve(normals@normals.T,offset)
            left=intervals(a,xyz,origin,direction);right=intervals(b,xyz,origin,direction)
            for alo,ahi in left:
                for blo,bhi in right:
                    lo,hi=max(alo,blo),min(ahi,bhi)
                    if hi-lo<1.:continue
                    contacts.append(dict(p0=origin+lo*direction,p1=origin+hi*direction,
                        direction=direction,length=float(hi-lo),face_indices=[i,j],
                        materials=None,material_status='UNKNOWN_NOT_INFERRED',
                        weld_status='UNCLASSIFIED_GEOMETRIC_CANDIDATE'))
    return contacts


def geometry_support(contacts,target):
    results=[]
    for pi,parent in enumerate(target['parents']):
        for ci,child in enumerate(parent['children']):
            if child['original_kind']!='line':
                results.append(dict(parent=pi,child=ci,supported=False,reason='CURVE_NOT_IMPLEMENTED'));continue
            points=np.asarray(child['world_representation']['endpoints_mm']);best=float('inf')
            for c in contacts:
                q=(points-c['p0'])@c['direction']
                nearest=c['p0']+np.clip(q,0,c['length'])[:,None]*c['direction']
                best=min(best,float(np.linalg.norm(nearest-points,axis=1).max()))
            results.append(dict(parent=pi,child=ci,supported=best<=3.,finite_support_distance_mm=best))
    return dict(original_G=target['original_children'],geometric_supported=sum(r['supported'] for r in results),
        rows=results,not_extraction_accuracy=True,material_semantics_unverified=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='unused_single_792');ap.add_argument('--source',type=int,default=21)
    args=ap.parse_args();out=OUT/args.run;out.mkdir(exist_ok=False);started=time.monotonic()
    try:
        row=read(OUT/'inventory.json')[args.source];path=ROOT/row['raw']['original_pcd_path']
        assert digest(path)==row['raw']['original_pcd_sha256']
        write(out/'registered_protocol.json',dict(stage='PURE_XYZ_FINITE_SURFACE_REPRESENTATION',source=args.source,
            original_PCD_sha256=digest(path),source_sha256=digest(__file__),plane_tolerance_mm=.2,
            min_core_points=32,max_planes=512,triangle_rule='adaptive endpoint 24-neighbor radii; subscale holes unresolved',plane_seeds='fine normal mode and local projection histogram maxima',prior_failed_diagnostic='run12',
            material_identity='UNKNOWN',optimizer_updates=0,STEP_reads=0,weld_GT_reads_during_generation=0,
            no_dataset_expansion=True,full_contract_pass=False))
        xyz=read_xyz_pcd(path)*1000.;check_resources()
        print('PURE_XYZ_INPUT',args.source,'points',len(xyz),'NO_EPOCH_GEOMETRY_PREPROCESSING',flush=True)
        faces,diagnostic=extract_planes(xyz);contacts=contacts_from_support(faces,xyz)
        write(out/'surfaces.json',serial(faces));write(out/'contacts.json',serial(contacts))
        write(out/'input_freeze.json',dict(surfaces_sha256=digest(out/'surfaces.json'),
            contacts_sha256=digest(out/'contacts.json'),input_PCD_sha256=digest(path),
            STEP_reads=0,weld_GT_reads_during_generation=0,all_outputs_frozen_before_target_reads=True))
        target_path=ROOT/row['target']['target_path'];assert digest(target_path)==row['target']['target_sha256']
        coverage=geometry_support(contacts,read(target_path));write(out/'geometry_support_only.json',coverage)
        write(out/'receipt.json',dict(status='PURE_XYZ_SURFACE_REPRESENTATION_DIAGNOSTIC_COMPLETE',utc=utc(),
            source=args.source,points=len(xyz),surfaces=len(faces),contacts=len(contacts),
            **diagnostic,geometric_supported=coverage['geometric_supported'],original_G=coverage['original_G'],
            optimizer_updates=0,extraction_accuracy=None,material_status='UNKNOWN_NOT_INFERRED',
            full_contract_pass=False,elapsed_seconds=time.monotonic()-started))
        print('PURE_XYZ_REPRESENTATION_COMPLETE','faces',len(faces),'contacts',len(contacts),
              'geometric_support',coverage['geometric_supported'],'/',coverage['original_G'],
              'NOT_WELD_ACCURACY',flush=True)
    except BaseException as error:
        write(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),optimizer_updates=0));raise


if __name__=='__main__':main()
