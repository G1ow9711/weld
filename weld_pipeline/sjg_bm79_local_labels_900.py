"""Recover the single native BM79A parent from its ordered label-only PLY."""
import argparse
import gc
import traceback
import numpy as np
import openpyxl
from weld_pipeline.sjg_curve_label_inventory_895 import ROOT,OUT,read,write,digest,utc
from weld_pipeline.sjg_local_input_897 import read_workpiece_local_surface,validate_local_pair
from weld_pipeline.contract_curve_geometry import fit_ordered_circle,sample_arc,verify_sampled_arc


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);folder=ROOT/'DataSet/9月20日/BM79A'
    cloud=folder/'BM79A-0920-WQ_weld_workpiece.ply';label_file=folder/'BM79A-0920-WQ_weld_welds.ply';book_path=folder/'BM79A-0920-WQ_weld.xlsx'
    write(dest/'registered_protocol.json',dict(stage='NATIVE_LOCAL_MIXED_LINE_ARC_LABEL_RECOVERY',source_key='BM79A_native_local',
        user_dataset_directory=str(ROOT/'DataSet'),point_input_is_geometry_only_workpiece_PLY=True,
        label_PLY_never_model_input=True,source_sha256=digest(__file__),no_training_or_selection_on_source24=True,
        no_full65_expansion=True,no_acceptance_claim=True,start_utc=utc()))
    try:
        xyz,meta=read_workpiece_local_surface(cloud);np.save(dest/'xyz_mm.npy',xyz)
        write(dest/'input_frozen_before_labels.json',dict(meta,file='xyz_mm.npy',sha256=digest(dest/'xyz_mm.npy'),
            original_file=str(cloud.relative_to(ROOT)),original_sha256=digest(cloud)))
        print('BM79_LOCAL_XYZ_FROZEN',len(xyz),flush=True);del xyz;gc.collect()
        wb=openpyxl.load_workbook(book_path,read_only=True,data_only=False);rows=list(wb['焊缝信息'].iter_rows(values_only=True));wb.close()
        keys=[str(x or '').split('\n')[-1] for x in rows[0]];records=[dict(zip(keys,r)) for r in rows[1:] if any(x is not None and x!='' for x in r)]
        assert len(records)==1;row=records[0];identity=row['ObjectId'];assert row['HasMid (Any)'] is True
        with label_file.open('r',encoding='ascii') as h:
            header=[]
            for line in h:
                header.append(line.strip())
                if line.strip()=='end_header':break
            assert 'comment unit millimeter; frame reference_workpiece' in header
            samples=np.loadtxt(h,dtype=np.float64,ndmin=2)
        points=samples[:,:3];assert len(np.unique(samples[:,3:],axis=0))==1
        steps=np.linalg.norm(np.diff(points,axis=0),axis=1);cuts=np.flatnonzero(steps<1e-7)+1;blocks=np.split(points,cuts)
        assert len(blocks)==int(row['SegmentCount'])==4
        start=np.array([row['StartMatrix.Translation.'+a] for a in 'XYZ']);end=np.array([row['EndMatrix.Translation.'+a] for a in 'XYZ'])
        assert np.linalg.norm(points[0]-start)<1e-7 and np.linalg.norm(points[-1]-end)<1e-7
        children=[];total=0.;curves=0;checks=[]
        for index,block in enumerate(blocks):
            direction=block[-1]-block[0];length=float(np.linalg.norm(direction));unit=direction/length
            line_error=float(np.linalg.norm(np.cross(block-block[0],unit),axis=1).max())
            if line_error<=1e-6:
                kind='line';representation=dict(kind=kind,endpoints_mm=block[[0,-1]].tolist());loss=np.linspace(block[0],block[-1],17)
            else:
                kind='arc';curves+=1;arc=fit_ordered_circle(block,tolerance_mm=1e-6,is_closed=False)
                sampled=sample_arc(arc);verify_sampled_arc(sampled);length=arc.length_mm
                representation=dict(kind=kind,arc_parameters=arc.to_parameters());loss=arc.point_at(np.linspace(0,length,17))
            total+=length;checks.append(dict(index=index,points=len(block),kind=kind,length_mm=length,line_residual_mm=line_error))
            children.append(dict(original_child_id=identity+':'+str(index),original_parent_id=identity,original_segment_index=index,
                native_segment_index=index,logical_segment_index=index,original_kind=kind,weld_type='horizontal',
                representation=representation,original_points_mm=block.tolist(),loss_samples_mm=loss.tolist(),coordinate_frame='workpiece_local',
                original_attributes={},attributes_status='UNKNOWN_NO_CHILD_ATTRIBUTE_TRUTH_FABRICATED',formal_E_reviewed=False))
        assert curves==3 and abs(total-float(row['Length']))<1e-6
        parent=dict(original_parent_id=identity,original_parent_index=0,original_child_count=4,children=children,
            gaps=['SHARED']*3,endpoints_mm=[points[0].tolist(),points[-1].tolist()],is_closed=False,mode='MIXED_CHAIN')
        labels=dict(coordinate_frame='workpiece_local',unit='mm',parents=[parent],original_parents=1,original_children=4,
            unresolved_children=0,original_declared_parent_count=1,original_declared_child_count=4,
            role='NEW_SMALL_TRAIN_CANDIDATE_NOT_FORMAL_TEST',label_source='native_label_PLY_order_and_duplicate_joints_plus_single_parent_Excel')
        validate_local_pair(meta,labels);write(dest/'targets_local.json',labels)
        inventory=read(OUT/'inventory.json');old_cloud_hashes={r['raw']['original_pcd_sha256'] for r in inventory}
        pcd=folder/'_BM79A_GW1_AllFaces.pcd';byte_duplicate=digest(pcd) in old_cloud_hashes
        write(dest/'receipt.json',dict(status='NATIVE_MIXED_LOCAL_LABELS_RECOVERED',original_points=meta['original_points'],
            label_points=len(points),source_key='BM79A_native_local',original_G=4,line_children=1,arc_children=curves,
            exact_curve_length_sum_mm=total,excel_length_mm=float(row['Length']),length_error_mm=abs(total-float(row['Length'])),
            maximum_original_sample_step_mm=float(steps.max()),duplicate_joint_positions=cuts.tolist(),segments=checks,
            hashes=dict(input=digest(dest/'xyz_mm.npy'),target=digest(dest/'targets_local.json'),label_PLY=digest(label_file),workbook=digest(book_path)),
            duplicate_legacy_PCD_bytes=byte_duplicate,rigid_near_duplicate_audit_pending=True,
            attributes_not_promoted_from_parent=True,optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
        print('BM79_NATIVE_LABELS_RECOVERED',checks,'LENGTH_ERROR_MM',abs(total-float(row['Length'])),'BYTE_DUPLICATE',byte_duplicate,flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
