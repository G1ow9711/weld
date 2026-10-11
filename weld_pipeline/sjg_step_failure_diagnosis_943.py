"""Offline STEP teacher audit; never writes deployment proposals or train labels."""
import argparse
from collections import Counter
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest, utc


def plane_matches(faces, normal, points):
    normals=np.asarray([f['normal'] for f in faces]);offsets=np.asarray([f['offset'] for f in faces])
    angles=np.degrees(np.arccos(np.clip(np.abs(normals@normal),0,1)))
    residual=np.max(np.abs(points@normals.T-offsets),axis=0)
    eligible=np.flatnonzero((angles<=.1)&(residual<=.2))
    rank=np.argsort(residual+1000*(angles>.1))[:3]
    return dict(matches=eligible.tolist(),nearest=[dict(index=int(i),angle_deg=float(angles[i]),residual_mm=float(residual[i])) for i in rank])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);inventory=read(OUT/'inventory.json')
    cases={c['source_key']:c for c in read(OUT/'run98/local_cohort.json')['cases']};summaries=[]
    for key in (4,10):
        case=cases[str(key)];cached=OUT/f'source_{key:02d}';freeze=read(cached/'input_freeze.json')
        assert digest(cached/'geometry.json')==freeze['geometry_sha256']
        assert digest(ROOT/inventory[key]['step']['STEP_path'])==inventory[key]['step']['STEP_sha256']
        folder=(ROOT/inventory[key]['raw']['original_pcd_path']).parent
        native_path=list(folder.glob('*_weld_welds.json'));assert len(native_path)==1
        transform=np.asarray(read(native_path[0])['exportToWorld']);rot=transform[:3,:3];shift=transform[:3,3]
        assert np.allclose(rot.T@rot,np.eye(3),atol=1e-7)
        geometry=read(cached/'geometry.json');coverage=read(cached/'coverage.json')
        target=read(OUT/case['target_file']);assign={(a['parent'],a['child']):a for a in coverage['assignments']}
        missing={(r['parent'],r['child']) for r in read(OUT/f'run126/source_{key}/TRAIN_only_geometric_compatibility.json')['missing']}
        hypotheses=read(OUT/f'run115/source_{key}/surfaces.json');selected=read(OUT/f'run125/source_{key}/surfaces.json')
        rows=[];face_cache={}
        for pi,parent in enumerate(target['parents']):
            for ci,child in enumerate(parent['children']):
                gt=np.asarray(child['representation']['endpoints_mm']);a=assign.get((pi,ci))
                if a is None:
                    rows.append(dict(parent=pi,child=ci,category='STEP_TEACHER_CONTACT_ABSENT'));continue
                contact=geometry['contacts'][a['contact']];ends=(np.asarray([contact['p0'],contact['p1']])-shift)@rot
                direction=ends[1]-ends[0];length=np.linalg.norm(direction);direction/=length
                q=np.clip((gt-ends[0])@direction,0,length);error=float(np.linalg.norm(ends[0]+q[:,None]*direction-gt,axis=1).max())
                face_rows=[]
                for fi in contact['face_indices']:
                    if fi not in face_cache:
                        f=geometry['faces'][fi];points=(np.concatenate([np.asarray(v) for v in f['lines']])-shift)@rot
                        normal=np.asarray(f['normal'])@rot
                        face_cache[fi]=dict(STEP_face_index=fi,tag=f['tag'],volume=f['volume'],
                            normal=normal.tolist(),center_local_mm=((np.asarray(f['center'])-shift)@rot).tolist(),
                            hypotheses=plane_matches(hypotheses,normal,points),selected=plane_matches(selected,normal,points))
                    face_rows.append(face_cache[fi])
                category='CANDIDATE_PRESENT'
                if (pi,ci) in missing:
                    if any(not f['hypotheses']['matches'] for f in face_rows):category='REQUIRED_STEP_PLANE_NOT_ACCURATE_IN_HYPOTHESES'
                    elif any(not f['selected']['matches'] for f in face_rows):category='REQUIRED_STEP_PLANE_DROPPED_BY_SELECTION'
                    else:category='PLANES_PRESENT_BUT_FINITE_INTERSECTION_MISSING'
                rows.append(dict(parent=pi,child=ci,id=child['original_child_id'],category=category,
                    STEP_contact=a['contact'],STEP_contact_max_endpoint_distance_mm=error,STEP_face_indices=contact['face_indices'],
                    gt_length_mm=float(np.linalg.norm(gt[1]-gt[0])),gt_endpoints_mm=gt.tolist()))
        counts=dict(Counter(r['category'] for r in rows));errors=[r['STEP_contact_max_endpoint_distance_mm'] for r in rows if 'STEP_contact_max_endpoint_distance_mm' in r]
        summary=dict(source=key,G=case['original_G'],categories=counts,STEP_contact_max_endpoint_distance_mm=max(errors),
            STEP_geometry_sha256=digest(cached/'geometry.json'),native_metadata_sha256=digest(native_path[0]),
            registration_provenance=inventory[key]['step']['verified_offline_frames'][0],
            independent_validation=False,teacher_diagnostic_only=True,optimizer_updates=0)
        write(dest/f'source_{key}.json',dict(summary=summary,children=rows,required_faces=list(face_cache.values())))
        summaries.append(summary);print('STEP_FAILURE_DIAGNOSIS',key,counts,'teacher_max_mm',max(errors),flush=True)
    write(dest/'receipt.json',dict(status='OFFLINE_STEP_FAILURE_DIAGNOSIS_COMPLETE',sources=summaries,
        no_runtime_STEP_dependency_added=True,optimizer_updates=0,full_contract_pass=False,end_utc=utc()))


if __name__=='__main__':main()
