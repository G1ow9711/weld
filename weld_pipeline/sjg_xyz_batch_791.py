"""Memory-bounded pure-XYZ representation on the fixed eight-workpiece pilot."""
import argparse
import gc
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_geometry_776 import read, write, digest, serial
from weld_pipeline.sjg_prepare_776 import ROOT, OUT, TRAIN, VALID, ARC
from weld_pipeline.sjg_train_776 import check_resources, utc
from weld_pipeline import sjg_xyz_surface_790 as geometry


def normal_cloud(xyz):
    tree=cKDTree(xyz);normals=np.empty_like(xyz);centers=np.empty_like(xyz)
    planar=np.empty(len(xyz),bool)
    for begin in range(0,len(xyz),16384):
        end=min(begin+16384,len(xyz))
        near=tree.query(xyz[begin:end],k=min(12,len(xyz)),workers=2)[1]
        cloud=xyz[near];center=cloud.mean(1);delta=cloud-center[:,None]
        covariance=np.einsum('nki,nkj->nij',delta,delta)/cloud.shape[1]
        eigen,vectors=np.linalg.eigh(covariance)
        normals[begin:end]=vectors[:,:,0];centers[begin:end]=center
        planar[begin:end]=(eigen[:,0]/eigen[:,1].clip(1e-8)<.02)&(eigen[:,1]>1e-6)
    return normals,centers,planar


def intervals(face,xyz,origin,direction):
    points=xyz[face['point_ids']];tri=points[face['triangles']]
    side=np.cross(face['normal'],direction)
    signed=(tri-origin)@side;q=(tri-origin)@direction
    mask=(signed.min(1)<=1e-7)&(signed.max(1)>=-1e-7)
    signed=signed[mask];q=q[mask]
    lower=np.full(len(q),np.inf);upper=np.full(len(q),-np.inf)
    for i,j in ((0,1),(1,2),(2,0)):
        at=np.abs(signed[:,i])<1e-7
        lower[at]=np.minimum(lower[at],q[at,i]);upper[at]=np.maximum(upper[at],q[at,i])
        cross=signed[:,i]*signed[:,j]<0
        cut=q[cross,i]+(q[cross,j]-q[cross,i])*signed[cross,i]/(signed[cross,i]-signed[cross,j])
        lower[cross]=np.minimum(lower[cross],cut);upper[cross]=np.maximum(upper[cross],cut)
    valid=(upper-lower)>1e-6;order=np.argsort(lower[valid]);merged=[]
    for lo,hi in zip(lower[valid][order],upper[valid][order]):
        if merged and lo<=merged[-1][1]+1e-5:merged[-1][1]=max(merged[-1][1],float(hi))
        else:merged.append([float(lo),float(hi)])
    return merged


def contacts_from_support(faces,xyz):
    bounds=[(xyz[f['point_ids']].min(0),xyz[f['point_ids']].max(0)) for f in faces]
    contacts=[]
    for i,a in enumerate(faces):
        for j in range(i+1,len(faces)):
            if np.any(np.maximum(bounds[i][0],bounds[j][0])>np.minimum(bounds[i][1],bounds[j][1])+.4):continue
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
                        direction=direction,length=float(hi-lo),face_indices=[i,j],materials=None,
                        material_status='UNKNOWN_NOT_INFERRED',weld_status='UNCLASSIFIED_GEOMETRIC_CANDIDATE'))
        if i%8==0:print('XYZ_FACE_PAIR_PROGRESS',i+1,'/',len(faces),'candidates',len(contacts),flush=True)
    return contacts


def save_surfaces(folder,faces):
    descriptors=[];arrays={}
    for i,f in enumerate(faces):
        arrays[f'point_ids_{i}']=f['point_ids'];arrays[f'triangles_{i}']=f['triangles']
        descriptor={k:v for k,v in f.items() if k not in ('point_ids','triangles')}
        descriptor.update(point_ids_key=f'point_ids_{i}',triangles_key=f'triangles_{i}',
                          point_count=len(f['point_ids']),triangle_count=len(f['triangles']))
        descriptors.append(descriptor)
    np.savez_compressed(folder/'surface_support.npz',**arrays)
    write(folder/'surfaces.json',serial(descriptors))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run14');args=ap.parse_args()
    out=OUT/args.run;out.mkdir(exist_ok=False);started=time.monotonic()
    try:
        inventory=read(OUT/'inventory.json');selected=TRAIN+VALID+ARC
        protocol=dict(stage='PURE_XYZ_FIXED_EIGHT_REPRESENTATION',sources=selected,
            original_G=sum(inventory[i]['G'] for i in selected),normal_chunk_points=16384,
            max_planes=64,plane_tolerance_mm=.2,min_core_points=32,source_sha256=digest(__file__),
            geometry_source_sha256=digest(geometry.__file__),same_original_pilot_no_expansion=True,
            material_identity='UNKNOWN',STEP_reads=0,weld_GT_reads_during_generation=0,
            optimizer_updates=0,full_contract_pass=False)
        write(out/'registered_protocol.json',protocol)
        # Resource-equivalent replacement; changes no labels or thresholds.
        geometry.normal_cloud=normal_cloud
        frozen=[]
        for si in selected:
            row=inventory[si];folder=out/f'source_{si:02d}';folder.mkdir()
            path=ROOT/row['raw']['original_pcd_path'];assert digest(path)==row['raw']['original_pcd_sha256']
            check_resources();tick=time.monotonic();xyz=read_xyz_pcd(path)*1000.
            print('PURE_XYZ_WORKPIECE',si,'points',len(xyz),'NO_TRAINING_EPOCH',flush=True)
            faces,diagnostic=geometry.extract_planes(xyz)
            contacts=contacts_from_support(faces,xyz)
            save_surfaces(folder,faces);write(folder/'contacts.json',serial(contacts))
            freeze=dict(source=si,original_PCD_sha256=digest(path),points=len(xyz),
                surfaces=len(faces),contacts=len(contacts),**diagnostic,
                files={name:digest(folder/name) for name in ['surfaces.json','surface_support.npz','contacts.json']},
                STEP_reads=0,weld_GT_reads_during_generation=0,elapsed_seconds=time.monotonic()-tick)
            write(folder/'input_freeze.json',freeze);frozen.append(freeze)
            write(out/'progress.json',dict(stage='GENERATING_WITHOUT_LABELS',completed=len(frozen),total=len(selected),source=si))
            print('FROZEN_PURE_XYZ',si,'planes',len(faces),'contacts',len(contacts),'seconds',round(freeze['elapsed_seconds'],2),flush=True)
            del xyz,faces,contacts;gc.collect();check_resources()
        write(out/'all_inputs_frozen.json',dict(outputs=frozen,all_frozen_before_any_weld_GT_evaluation=True))
        results=[]
        for si in selected:
            folder=out/f'source_{si:02d}';freeze=read(folder/'input_freeze.json')
            assert digest(folder/'contacts.json')==freeze['files']['contacts.json']
            contacts=read(folder/'contacts.json')
            for c in contacts:
                c['p0']=np.asarray(c['p0']);c['direction']=np.asarray(c['direction'])
            target_path=ROOT/inventory[si]['target']['target_path'];assert digest(target_path)==inventory[si]['target']['target_sha256']
            coverage=geometry.geometry_support(contacts,read(target_path));write(folder/'geometry_support_only.json',coverage)
            row=dict(source=si,original_G=coverage['original_G'],geometric_supported=coverage['geometric_supported'],
                     surfaces=freeze['surfaces'],candidates=freeze['contacts'],hit_plane_cap=freeze['hit_plane_cap'],
                     remaining_unassigned_points=freeze['remaining_unassigned_points'])
            results.append(row);print('XYZ_GEOMETRY_SUPPORT_ONLY',row,flush=True)
        write(out/'receipt.json',dict(status='PURE_XYZ_FIXED_PILOT_REPRESENTATION_COMPLETE',utc=utc(),
            rows=results,original_G=sum(r['original_G'] for r in results),
            geometric_supported=sum(r['geometric_supported'] for r in results),
            not_weld_extraction_accuracy=True,extraction_accuracy=None,material_status='UNKNOWN',
            curves_not_implemented=True,optimizer_updates=0,full_contract_pass=False,
            elapsed_seconds=time.monotonic()-started))
        print('PURE_XYZ_PILOT_DIAGNOSTIC_COMPLETE | NOT CONTRACT ACCEPTANCE',flush=True)
    except BaseException as error:
        write(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),optimizer_updates=0));raise


if __name__=='__main__':main()
