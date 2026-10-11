"""Preserve the original eight local cases and register recovered curve TRAIN data."""
import argparse
import copy
import gc
import traceback
import numpy as np
from weld_pipeline.sjg_curve_label_inventory_895 import ROOT,OUT,read,write,digest,utc
from weld_pipeline.sjg_local_input_897 import read_workpiece_local_surface,validate_local_pair
from weld_pipeline.contract_curve_geometry import CircularArc,fit_ordered_circle


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);inventory=read(OUT/'inventory.json');source=24
    folder=(ROOT/inventory[source]['raw']['original_pcd_path']).parent
    write(dest/'registered_protocol.json',dict(stage='COMPLETE_ORIGINAL_LOCAL_COHORT_WITH_CURVE_CANDIDATE',
        original_eight_G=496,additional_BM79_G=4,curve_diagnostic_never_used_for_training=True,
        old_training_sources=[21,4,10,25,36,42],development=[64],arc_diagnostic=[24],
        additional_TRAIN_candidate='BM79A_native_local',source_sha256=digest(__file__),
        full_contract_pass=False,start_utc=utc()))
    try:
        cloud=folder/'TD2-LB8A_weld_workpiece.ply';xyz,meta=read_workpiece_local_surface(cloud);np.save(dest/'source_24_xyz_mm.npy',xyz)
        binding=dict(meta,source=24,file='source_24_xyz_mm.npy',sha256=digest(dest/'source_24_xyz_mm.npy'),
            original_file=str(cloud.relative_to(ROOT)),original_sha256=digest(cloud))
        write(dest/'source_24_input.json',binding);print('LOCAL_ARC_DIAGNOSTIC_XYZ_FROZEN',len(xyz),flush=True);del xyz;gc.collect()
        export_path=folder/'TD2-LB8A_weld_welds.json';native=read(export_path)
        assert native['coordinateFrame']=='reference_workpiece' and native['unit']=='millimeter'
        transform=np.asarray(native['exportToWorld']);R=transform[:3,:3];t=transform[:3,3]
        assert np.allclose(R.T@R,np.eye(3),atol=1e-7)
        parents={w['id']:w for w in native['welds']};old=read(ROOT/inventory[24]['target']['target_path']);target=copy.deepcopy(old)
        target.update(coordinate_frame='workpiece_local',unit='mm',unresolved_children=0,role='LOCAL_ARC_DEVELOPMENT_DIAGNOSTIC_NOT_TRAIN')
        maximum=0.;arcs=lines=0
        for parent in target['parents']:
            identity=parent['original_parent_id'];segments={s['index']:s for s in parents[identity]['segments']}
            for child in parent['children']:
                index=int(child['original_child_id'].rsplit(':',1)[1]);segment=segments[index];points=np.asarray(segment['points']);previous=child.pop('world_representation')
                assert segment['kind']==previous['kind']==child['original_kind'];fractions=np.linspace(0,1,17)
                if previous['kind']=='line':
                    expected=(np.asarray(previous['endpoints_mm'])-t)@R
                    if np.linalg.norm(points[[0,-1]][::-1]-expected)<np.linalg.norm(points[[0,-1]]-expected):points=points[::-1]
                    rep=dict(kind='line',endpoints_mm=points[[0,-1]].tolist());loss=np.linspace(points[0],points[-1],17)
                    reference=np.linspace(*np.asarray(previous['endpoints_mm']),17);lines+=1
                else:
                    reference_arc=CircularArc.from_parameters(previous['arc_parameters']);reference=reference_arc.point_at(fractions*reference_arc.length_mm)
                    expected=(reference[[0,-1]]-t)@R
                    if np.linalg.norm(points[[0,-1]][::-1]-expected)<np.linalg.norm(points[[0,-1]]-expected):points=points[::-1]
                    arc=fit_ordered_circle(points,tolerance_mm=1e-6,is_closed=False);rep=dict(kind='arc',arc_parameters=arc.to_parameters());loss=arc.point_at(fractions*arc.length_mm);arcs+=1
                error=float(np.linalg.norm(loss@R.T+t-reference,axis=1).max());assert error<1e-5;maximum=max(maximum,error)
                child.update(representation=rep,original_points_mm=points.tolist(),loss_samples_mm=loss.tolist(),
                    coordinate_frame='workpiece_local',native_segment_index=index,logical_segment_index=child['original_segment_index'])
            parent['endpoints_mm']=[parent['children'][0]['original_points_mm'][0],parent['children'][-1]['original_points_mm'][-1]]
        assert lines==87 and arcs==24;validate_local_pair(binding,target);write(dest/'source_24_targets_local.json',target)
        previous=read(OUT/'run96/receipt.json');bm=read(OUT/'run97/receipt.json');assert previous['status']=='NATIVE_LOCAL_DATASET_READY' and bm['status']=='NATIVE_MIXED_LOCAL_LABELS_RECOVERED'
        cases=[]
        for row in previous['rows']:
            inp=read(OUT/f"run96/source_{row['source']:02d}_input.json")
            cases.append(dict(source_key=str(row['source']),role='DEVELOPMENT' if row['source']==64 else 'TRAIN',
                input_file='run96/'+inp['file'],input_sha256=inp['sha256'],target_file='run96/'+row['target_file'],target_sha256=row['target_sha256'],original_G=row['original_G']))
        cases.append(dict(source_key='24',role='ARC_DIAGNOSTIC_NOT_TRAIN',input_file=str((dest/binding['file']).relative_to(OUT)),
            input_sha256=binding['sha256'],target_file=str((dest/'source_24_targets_local.json').relative_to(OUT)),target_sha256=digest(dest/'source_24_targets_local.json'),original_G=111))
        cases.append(dict(source_key='BM79A_native_local',role='ADDITIONAL_TRAIN_CANDIDATE_GROUP_AUDIT_PENDING',
            input_file='run97/xyz_mm.npy',input_sha256=bm['hashes']['input'],target_file='run97/targets_local.json',target_sha256=bm['hashes']['target'],original_G=4))
        for row in cases:
            assert digest(OUT/row['input_file'])==row['input_sha256'];assert digest(OUT/row['target_file'])==row['target_sha256']
            validate_local_pair(dict(coordinate_frame='workpiece_local',unit='mm'),read(OUT/row['target_file']))
        write(dest/'local_cohort.json',dict(coordinate_frame='workpiece_local',unit='mm',cases=cases,
            original_eight_G=496,including_new_training_candidate_G=500,legacy_world_feature_cache_compatible=False,
            no_new_accuracy_claim=True,point_features_must_be_recomputed=True))
        write(dest/'receipt.json',dict(status='LOCAL_LINE_ARC_COHORT_BOUND',source24_lines=lines,source24_arcs=arcs,
            source24_geometry_roundtrip_maximum_mm=maximum,original_eight_G=496,new_candidate_G=4,
            cohort_sha256=digest(dest/'local_cohort.json'),optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
        print('LOCAL_LINE_ARC_COHORT_BOUND','ORIGINAL_G',496,'NEW_TRAIN_CANDIDATE_G',4,'ROUNDTRIP_MM',maximum,flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
