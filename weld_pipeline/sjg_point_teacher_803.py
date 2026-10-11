"""Offline planar point teachers; never imported by deployed PTv3 inference."""
import argparse
import gc
from pathlib import Path
import traceback
import numpy as np
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_surface_learning_796 import inside_finite_face
from weld_pipeline.sjg_train_776 import check_resources,utc


def boundary_distance(points,face):
    distance=np.full(len(points),np.inf)
    for line in face['lines']:
        p=np.asarray(line,np.float64)
        for a,b in zip(p[:-1],p[1:]):
            v=b-a;squared=float(v@v)
            if squared<1e-16:continue
            t=np.clip((points-a)@v/squared,0,1)
            distance=np.minimum(distance,np.linalg.norm(points-(a+t[:,None]*v),axis=1))
    return distance


def point_teachers(xyz,faces,source):
    n=len(xyz);count=np.zeros(n,np.int16);instance=np.full(n,-1,np.int32)
    normal=np.zeros((n,3),np.float32);material=np.full(n,-1,np.int32)
    multiple_material=np.zeros(n,bool);boundary=np.full(n,np.inf,np.float32)
    for fi,face in enumerate(faces):
        lo=np.asarray(face['lo'])-.2;hi=np.asarray(face['hi'])+.2
        ids=np.flatnonzero(np.all((xyz>=lo)&(xyz<=hi),axis=1))
        if len(ids):
            direction=np.asarray(face['normal']);center=np.asarray(face['center'])
            ids=ids[np.abs((xyz[ids]-center)@direction)<=.2]
            ids=ids[inside_finite_face(xyz[ids],face)]
        if len(ids):
            fresh=ids[count[ids]==0];volume=int(face['volume'])
            instance[fresh]=fi;normal[fresh]=direction;material[fresh]=volume
            multiple_material[ids] |= material[ids]!=volume
            count[ids]+=1
            boundary[ids]=np.minimum(boundary[ids],boundary_distance(xyz[ids],face))
        if fi%25==0:print('OFFLINE_POINT_TEACHER',source,'face',fi+1,'/',len(faces),flush=True)
    single=count==1;covered=count>0
    instance[~single]=-1;normal[~single]=0;material[multiple_material|~covered]=-1
    # These volume identifiers are offline reference owners, not approved
    # customer material truth and not numerical input features.
    return dict(instance=instance,normal=normal,instance_known=single,
        membership_count=count,reference_volume=material,multiple_material=multiple_material,
        boundary_mm=boundary,boundary_target=np.where(covered,(boundary<=1.).astype(np.int8),-1),
        primitive_type=np.where(single,0,-1).astype(np.int8))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    protocol=dict(stage='OFFLINE_POINT_TEACHERS',start_utc=utc(),train=TRAIN,valid=VALID,arc=ARC,
        plane_distance_mm=.2,finite_boundary_tolerance_mm=.2,boundary_band_mm=1.,
        primitive_types_implemented=['plane'],normal_loss_must_be_unoriented=True,
        ambiguous_instances_unknown=True,STEP_used_only_for_offline_supervision=True,
        reference_volume_not_customer_approved_material_truth=True,
        actual_training_updates=0,contract_pass=False,source_sha256=digest(Path(__file__)))
    write(dest/'registered_protocol.json',protocol)
    try:
        inventory=read(OUT/'inventory.json');freezes=[]
        # Freeze every original input before reading any structural reference.
        for source in TRAIN+VALID+ARC:
            check_resources();raw=ROOT/inventory[source]['raw']['original_pcd_path']
            xyz=read_xyz_pcd(raw)*1000.
            file=dest/f'source_{source:02d}_xyz_mm.npy';np.save(file,xyz)
            freezes.append(dict(source=source,points=len(xyz),raw_path=str(raw),raw_sha256=digest(raw),
                xyz_file=file.name,xyz_sha256=digest(file),role='TRAIN' if source in TRAIN else 'VALID' if source in VALID else 'ARC_DIAGNOSTIC'))
            print('RAW_XYZ_FROZEN',source,len(xyz),flush=True);del xyz;gc.collect()
        write(dest/'all_inputs_frozen.json',dict(sources=freezes,utc=utc(),label_reads_before_freeze=0))
        summary=[]
        for record in freezes:
            check_resources();source=record['source'];reference=OUT/f'source_{source:02d}/geometry.json'
            if not reference.exists():
                summary.append(dict(source=source,status='NO_STRUCTURE_REFERENCE',points=record['points'],all_points_unknown=True));continue
            binding=read(reference.parent/'input_freeze.json');assert digest(reference)==binding['geometry_sha256']
            faces=read(reference)['faces'];xyz=np.load(dest/record['xyz_file'],mmap_mode='r')
            labels=point_teachers(xyz,faces,source)
            file=dest/f'source_{source:02d}_teacher.npz';np.savez_compressed(file,**labels)
            stats=dict(source=source,points=len(xyz),reference_faces=len(faces),reference_sha256=digest(reference),
                teacher_file=file.name,teacher_sha256=digest(file),single_face_points=int(labels['instance_known'].sum()),
                uncovered_points=int((labels['membership_count']==0).sum()),
                multiple_face_points=int((labels['membership_count']>1).sum()),
                multiple_material_points=int(labels['multiple_material'].sum()),
                boundary_positive=int((labels['boundary_target']==1).sum()),boundary_negative=int((labels['boundary_target']==0).sum()),
                represented_faces=int(len(np.unique(labels['instance'][labels['instance_known']]))))
            assert stats['single_face_points']+stats['uncovered_points']+stats['multiple_face_points']==len(xyz)
            summary.append(stats);write(dest/'label_summary.json',summary)
            print('POINT_LABEL_COMPLETE',stats,flush=True);del xyz,labels;gc.collect()
        write(dest/'label_summary.json',summary)
        write(dest/'receipt.json',dict(status='OFFLINE_PLANAR_POINT_TEACHERS_COMPLETE',end_utc=utc(),sources=summary,
            actual_training_updates=0,contract_pass=False,curved_primitive_supervision_not_implemented=True))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
