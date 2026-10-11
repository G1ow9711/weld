"""Offline finite CAD material hypotheses for original point-only candidates."""
import collections
import json
import shutil
import time
from pathlib import Path

import gmsh
import numpy as np

from training.labels import sha256
from weld_pipeline.audit_step_weld_contacts import contacts
from weld_pipeline.curved_cache import load_piece

OUT=Path('weld_pipeline/reports/candidate_material_identity_148_20261001')
NAMES=('224-GR24A1','207-FR79M','203-TT1A')


def classify(found,axes):
    sides=[]
    for axis in axes:
        axis=axis/np.linalg.norm(axis)
        sides.append([face for face in found if abs(np.array(face['unsigned_normal_world'])@axis)>=np.cos(np.deg2rad(3.))])
    pairs={(int(a['volume']),int(b['volume'])) for a in sides[0] for b in sides[1] if a['face']!=b['face']}
    same={pair for pair in pairs if pair[0]==pair[1]};cross=pairs-same
    if same and cross:
        state='SAME_OR_CROSS_BODY_AMBIGUOUS'
    elif cross:
        state='CROSS_BODY_ONLY'
    elif same:
        state='SAME_BODY_ONLY'
    elif sides[0] or sides[1]:
        state='ONE_FINITE_SIDE_MISSING'
    else:
        state='BOTH_FINITE_SIDES_MISSING'
    return dict(state=state,side_face_ids=[[int(face['face']) for face in side] for side in sides],
        side_volume_ids=[sorted({int(face['volume']) for face in side}) for side in sides],
        same_body_pairs=sorted(same),cross_body_pairs=sorted(cross),
        unique_cross_body_pair=(len(cross)==1 and not same),derived_customer_truth_reviewed=False)


def main():
    if OUT.exists():
        raise ValueError('Fixed material-identity audit exists')
    OUT.mkdir(parents=True)
    split_path=Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split=json.loads(split_path.read_text('utf8'))
    sources=[__file__,'weld_pipeline/audit_step_weld_contacts.py','weld_pipeline/curved_cache.py']
    config=dict(role='OFFLINE_FIT_MATERIAL_HYPOTHESIS_AUDIT',pieces=list(NAMES),fractions=[0.,.25,.5,.75,1.],
        normal_unsigned_tolerance_deg=3.,reference_plane_distance_mm=.2,boundary_face_center_nudge_mm=.1,
        classes=['SAME_BODY_ONLY','CROSS_BODY_ONLY','SAME_OR_CROSS_BODY_AMBIGUOUS','ONE_FINITE_SIDE_MISSING','BOTH_FINITE_SIDES_MISSING'],
        sources={path:sha256(path) for path in sources},split_sha256=sha256(split_path),
        STEP_used_offline_only=True,material_IDs_are_not_runtime_features=True,
        no_geometry_labels_modified=True,reviewed_customer_truth=False,full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    snapshot=OUT/'source_snapshot';snapshot.mkdir()
    for source in sources:
        shutil.copy2(source,snapshot/Path(source).name)
    gmsh.initialize([],readConfigFiles=False);gmsh.option.setNumber('General.Terminal',0)
    gmsh.option.setString('Geometry.OCCTargetUnit','MM');results={};started=time.perf_counter()
    try:
        for name in NAMES:
            if name not in split['train'] or name in split['validation']:
                raise ValueError('Only registered FIT sources allowed')
            cad_path=Path('weld_pipeline/reports/step_geometry_20260930')/(name+'.json')
            reference_path=Path('weld_pipeline/reports/step_contact_continuity_candidates_v1')/(name+'_attribute_candidates.json')
            cad=json.loads(cad_path.read_text('utf8'));reference=json.loads(reference_path.read_text('utf8'))
            if not reference['pose_verified_for_reference']:
                raise ValueError('Finite reference pose is not verified')
            step=Path(reference['step'])
            if sha256(step)!=reference['step_sha256'] or sha256(step)!=cad['sha256']:
                raise ValueError('Original STEP changed')
            transform=np.array(reference['alignment']['transform']);rotation,translation=transform[:3,:3],transform[:3,3]
            piece=load_piece(split['piece_roots'][name],verify=True)
            if piece['record']['pcd_sha256']!=reference['pcd_sha256'] or sha256(piece['record']['input'])!=reference['pcd_sha256']:
                raise ValueError('Original point/reference alignment changed')
            gmsh.clear();gmsh.model.occ.importShapes(str(step.resolve()));gmsh.model.occ.synchronize()
            faces=[{**face,'volume':volume['volume'],'board_name':volume['name'],
                'thickness':volume['supported_thickness_mm'],'outward':dict(normal=None,verified=False)}
                for volume in cad['volumes'] for face in volume['planar_faces']]
            rows=[];counts=collections.Counter();positive=collections.Counter();negative=collections.Counter()
            for si,(sequence,target) in enumerate(zip(piece['sequences'],piece['targets'])):
                if sequence.get('curve_kind')=='arc':
                    counts['arc_candidates_not_audited']+=1;continue
                indices=np.unique(np.round(np.array(config['fractions'])*(len(sequence['xyz'])-1)).astype(int))
                samples=[]
                for index in indices:
                    q=sequence['xyz'][index].astype(float)
                    axes=sequence['features'][index,60:66].astype(float).reshape(2,3)
                    found=contacts(q,faces,rotation,translation)
                    hypothesis=classify(found,axes)
                    known=bool(target[index,0]>=0.);weld=bool(target[index,0]>.5) if known else None
                    sample=dict(sample_index=int(index),fraction=float(index/(len(sequence['xyz'])-1)),xyz_mm=q.tolist(),
                        reference_foreground_diagnostic_only=weld,finite_contact_faces=found,material_hypothesis=hypothesis)
                    samples.append(sample);counts[hypothesis['state']]+=1
                    if known:
                        (positive if weld else negative)[hypothesis['state']]+=1
                rows.append(dict(sequence=si,type=sequence['type'],
                    has_any_original_weld_supervision=bool((target[:,0]>.5).any()),samples=samples,
                    identity_role='offline_reference_only_not_a_welding_side_definition'))
            result=dict(piece=name,candidates=rows,total_nonarc_candidates=len(rows),
                sample_class_counts=dict(counts),foreground_sample_class_counts=dict(positive),background_sample_class_counts=dict(negative),
                input_sha256=piece['record']['pcd_sha256'],step_sha256=sha256(step),cad_description_sha256=sha256(cad_path),
                pose_reference_sha256=sha256(reference_path),sample_count_is_not_child_geometry_denominator=True,
                hidden_material_and_customer_truth_reviewed=False,full_contract_pass=False)
            if sha256(step)!=reference['step_sha256'] or sha256(piece['record']['input'])!=reference['pcd_sha256']:
                raise ValueError('Original input changed during offline audit')
            (OUT/(name+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
            results[name]={key:value for key,value in result.items() if key!='candidates'}
            print('MATERIAL_IDENTITY_AUDIT',name,'CANDIDATES',len(rows),'FG',dict(positive),'BG',dict(negative),flush=True)
    finally:
        gmsh.finalize()
    final=dict(config=config,pieces=results,elapsed_seconds=time.perf_counter()-started,
        observed_check_read=False,full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(final,indent=2,ensure_ascii=False),encoding='utf8')
    print('MATERIAL_IDENTITY_AUDIT_COMPLETE',len(results),flush=True)


if __name__=='__main__':
    main()
