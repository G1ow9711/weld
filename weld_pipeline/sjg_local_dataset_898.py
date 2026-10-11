"""Freeze native local point inputs, then bind labels without runtime CAD or IDs."""
import argparse
import copy
import gc
import traceback
import shutil
import numpy as np
from weld_pipeline.sjg_curve_label_inventory_895 import ROOT,OUT,read,write,digest,utc
from weld_pipeline.sjg_local_coordinate_audit_896 import workbook_labels
from weld_pipeline.sjg_local_input_897 import read_workpiece_local_surface,validate_local_pair

SOURCES=[21,4,10,25,36,42,64]


def main(reuse_inputs=None):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);inventory=read(OUT/'inventory.json');frozen=[];rows=[]
    write(dest/'registered_protocol.json',dict(stage='EXPLICIT_NATIVE_LOCAL_TRAINING_DATA',sources=SOURCES,
        train=SOURCES[:-1],development=[64],old_arc_diagnostic_unresolved=[24],
        original_eight_child_denominator=496,local_seven_child_denominator=385,
        native_geometry_only_PLY=True,unit='mm',coordinate_frame='workpiece_local',
        source_sha256=digest(__file__),loader_sha256=digest(ROOT/'weld_pipeline/sjg_local_input_897.py'),
        old_world_caches_not_reused=True,all_original_points_retained=True,
        no_runtime_STEP_or_reference_ID=True,full_contract_pass=False,start_utc=utc()))
    try:
        for source in SOURCES:
            if reuse_inputs is not None:
                previous=OUT/reuse_inputs;binding=read(previous/f'source_{source:02d}_input.json')
                assert binding['source']==source and binding['coordinate_frame']=='workpiece_local' and binding['unit']=='mm'
                assert digest(previous/binding['file'])==binding['sha256']
                assert digest(ROOT/binding['original_file'])==binding['original_sha256']
                shutil.copy2(previous/binding['file'],dest/binding['file']);write(dest/f'source_{source:02d}_input.json',binding)
                frozen.append(binding);print('VERIFIED_LOCAL_XYZ_REUSED',source,binding['original_points'],flush=True);continue
            folder=(ROOT/inventory[source]['raw']['original_pcd_path']).parent;paths=list(folder.glob('*_weld_workpiece.ply'));assert len(paths)==1
            xyz,meta=read_workpiece_local_surface(paths[0]);file=dest/f'source_{source:02d}_xyz_mm.npy';np.save(file,xyz)
            binding=dict(source=source,file=file.name,sha256=digest(file),original_file=str(paths[0].relative_to(ROOT)),
                original_sha256=digest(paths[0]),**meta)
            write(dest/f'source_{source:02d}_input.json',binding);frozen.append(binding)
            print('LOCAL_XYZ_FROZEN',source,len(xyz),'MM WORKPIECE_LOCAL SCALE=1',flush=True);del xyz;gc.collect()
        write(dest/'all_inputs_frozen_before_labels.json',dict(sources=frozen,all_before_label_reads=True))
        for source in SOURCES:
            folder=(ROOT/inventory[source]['raw']['original_pcd_path']).parent;exports=list(folder.glob('*_weld_welds.json'));books=list(folder.glob('*.xlsx'));assert len(books)==1
            workbook={r['id']:r for r in workbook_labels(books[0])};old=read(ROOT/inventory[source]['target']['target_path']);local=copy.deepcopy(old)
            local.update(coordinate_frame='workpiece_local',unit='mm',unresolved_children=0,
                old_world_field_names_removed=True,role='LOCAL_DEVELOPMENT_NOT_FORMAL_INDEPENDENT')
            maximum=0.;count=0;native_count=0
            if exports:
                assert len(exports)==1;native=read(exports[0]);assert native['coordinateFrame']=='reference_workpiece' and native['unit']=='millimeter'
                transform=np.asarray(native['exportToWorld']);rotation=transform[:3,:3];translation=transform[:3,3]
                assert np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7)
                parents={p['id']:p for p in native['welds']};native_count=sum(len(p['segments']) for p in parents.values())
            else:parents=None;rotation=translation=None
            for parent in local['parents']:
                identity=parent['original_parent_id'];assert identity in workbook
                for child in parent['children']:
                    original=child.pop('world_representation');assert original['kind']=='line'
                    if parents is not None:
                        assert identity in parents
                        assert child['original_child_id'].startswith(identity+':')
                        native_index=int(child['original_child_id'].rsplit(':',1)[1])
                        child['native_segment_index']=native_index;child['logical_segment_index']=child['original_segment_index']
                        segments={s['index']:s for s in parents[identity]['segments']};segment=segments[native_index]
                        assert segment['kind']=='line';native_points=np.asarray(segment['points'],float);ends=native_points[[0,-1]]
                        expected=(np.asarray(original['endpoints_mm'])-translation)@rotation
                        if np.linalg.norm(ends[::-1]-expected)<np.linalg.norm(ends-expected):ends=ends[::-1]
                        error=float(np.linalg.norm(ends-expected,axis=1).max());assert error<1e-5,(source,identity,error)
                        maximum=max(maximum,error)
                    else:
                        assert parent['original_child_count']==1 and len(parent['children'])==1
                        ends=np.asarray([workbook[identity]['local_start'],workbook[identity]['local_end']],float);native_count+=1
                    child['representation']=dict(kind='line',endpoints_mm=ends.tolist())
                    child['original_points_mm']=ends.tolist();child['loss_samples_mm']=np.linspace(ends[0],ends[1],17).tolist()
                    child['coordinate_frame']='workpiece_local';count+=1
                parent['endpoints_mm']=[parent['children'][0]['representation']['endpoints_mm'][0],parent['children'][-1]['representation']['endpoints_mm'][-1]]
            assert count==native_count==local['original_children'],(source,count,native_count,local['original_children'])
            binding=next(r for r in frozen if r['source']==source);validate_local_pair(binding,local)
            path=dest/f'source_{source:02d}_targets_local.json';write(path,local)
            row=dict(source=source,original_G=count,target_file=path.name,target_sha256=digest(path),
                workbook_sha256=digest(books[0]),native_json_sha256=digest(exports[0]) if exports else None,
                local_native_vs_metadata_converted_legacy_maximum_mm=maximum,
                native_local_coordinates_are_target_source=True,old_world_coordinates_used_only_for_consistency_check=bool(exports),
                coordinate_transform_applied_to_point_input=False,material_truth_not_added=True)
            rows.append(row);print('LOCAL_LABELS_BOUND',source,count,'MAX_CROSSCHECK_MM',maximum,flush=True)
        assert sum(r['original_G'] for r in rows)==385
        write(dest/'receipt.json',dict(status='NATIVE_LOCAL_DATASET_READY',rows=rows,
            original_points=sum(r['original_points'] for r in frozen),original_seven_G=385,
            old_eight_G=496,unresolved_arc_diagnostic_G=111,optimizer_updates=0,
            reused_input_run=reuse_inputs,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
