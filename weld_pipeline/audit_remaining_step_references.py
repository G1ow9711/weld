"""Reference-only STEP pose and plate attributes; no inference dependency."""
import json
import time
from pathlib import Path

import gmsh
import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.audit_step_weld_contacts import contacts, outward
from weld_pipeline.derive_step_attribute_candidates import pairs
from weld_pipeline.refine_step_reference_pose import refinement
from weld_pipeline.verify_step_cloud_alignment import mesh_step, align
from weld_pipeline.truth import load_truth


def candidate_children(truth,faces,rotation,translation):
    result=[]
    for weld in truth:
        for index,seg in enumerate(weld['segments']):
            if seg['kind']!='line':
                continue
            p=np.array(seg['points']);tangent=p[-1]-p[0];tangent/=np.linalg.norm(tangent)
            samples=[]
            for fraction in [0.,.25,.5,.75,1.]:
                q=p[0]*(1-fraction)+p[-1]*fraction
                found=contacts(q,faces,rotation,translation)
                choices=[pair for pair in pairs(found,tangent) if all(
                    r['thickness_candidate_mm'] is not None and r['thickness_candidate_mm']>0 for r in pair)]
                samples.append({'fraction':fraction,'xyz_mm':q.tolist(),'contact_count':len(found),
                    'candidate_pairs':choices,'unique_pair':len(choices)==1})
            unique=[s for s in samples if s['unique_pair']]
            identities={tuple(p['volume'] for p in s['candidate_pairs'][0]) for s in unique}
            consistent=len(unique)==5 and len(identities)==1
            pair=next((s['candidate_pairs'][0] for s in unique if s['fraction']==.5),None)
            if pair is None and unique:
                pair=unique[0]['candidate_pairs'][0]
            state='geometric_candidate_all_samples' if consistent else 'geometric_candidate_partial_or_ambiguous'
            if pair is None:
                state='no_unique_contact_pair'
            result.append({'weld_id':weld['id'],'weld_number':weld['metadata'].get('WeldNumber (f/v + Name number)'),
                'child_index':index,'type':weld['type'],'state':state,'pair_at_mid_or_nearest_sample':pair,
                'samples':samples,'reviewed_customer_truth':False})
    return result


def main():
    out=Path('weld_pipeline/reports/all_step_references_20260930');out.mkdir(parents=True,exist_ok=True)
    sources={p:sha256(p) for p in [__file__,'weld_pipeline/audit_step_weld_contacts.py',
        'weld_pipeline/derive_step_attribute_candidates.py','weld_pipeline/refine_step_reference_pose.py',
        'weld_pipeline/verify_step_cloud_alignment.py']}
    gmsh.initialize([],readConfigFiles=False);gmsh.option.setNumber('General.Terminal',0)
    gmsh.option.setString('Geometry.OCCTargetUnit','MM')
    summaries=[]
    try:
        for folder in sorted(Path('DataSet/9月20日').iterdir()):
            steps=list(folder.glob('*.stp')) if folder.is_dir() else []
            if not steps:
                continue
            name=folder.name;path=out/(name+'_attribute_candidates.json');start=time.perf_counter()
            if path.exists():
                report=json.loads(path.read_text('utf8'))
                if report.get('sources')!=sources:
                    raise ValueError('Existing reference audit source changed')
            else:
                try:
                    step=steps[0];pcd=next(folder.glob('*AllFaces.pcd'))
                    cad=json.loads((Path('weld_pipeline/reports/step_geometry_20260930')/(name+'.json')).read_text('utf8'))
                    if sha256(step)!=cad['sha256']:
                        raise ValueError('STEP geometry changed')
                    prior=Path('weld_pipeline/reports/step_alignment_20260930')
                    if (prior/(name+'_attribute_candidates.json')).exists():
                        report=json.loads((prior/(name+'_attribute_candidates.json')).read_text('utf8'))
                        report['reused_reference_hash']=sha256(prior/(name+'_attribute_candidates.json'))
                        report['alignment']=json.loads((prior/(name+'_refined.json')).read_text('utf8'))['trials'][0]
                        report['pose_verified_for_reference']=True
                    else:
                        vertices,triangles,owners,face_ids,basis=mesh_step(step)
                        xyz=read_xyz_pcd(pcd)*1000.
                        trials=align(vertices,triangles,basis,xyz)
                        biggest=max(cad['volumes'],key=lambda v:v['volume_mm3'])
                        normal=np.array(biggest['parallel_major_face_pairs'][0]['normal_axis'])
                        rng=np.random.default_rng(132)
                        sample=xyz[rng.choice(len(xyz),min(70000,len(xyz)),replace=False)]
                        refined=[refinement(vertices,triangles,np.array(t['transform']),sample,normal) for t in trials[:2]]
                        refined.sort(key=lambda r:(r['point_to_cad_p95_mm'],r['point_to_cad_median_mm']))
                        pose=refined[0];t=np.array(pose['transform'])
                        np.savez_compressed(out/(name+'_mesh.npz'),vertices=vertices,triangles=triangles,owners=owners,
                            face_ids=face_ids,aligned_vertices=vertices@t[:3,:3].T+t[:3,3],transform=t)
                        trusted=pose['point_to_cad_p95_mm']<=.05 and pose['within_1mm_rate']>=.995
                        faces=[]
                        for volume in cad['volumes']:
                            for face in volume['planar_faces']:
                                faces.append({**face,'volume':volume['volume'],'board_name':volume['name'],
                                    'thickness':volume['supported_thickness_mm'],'outward':outward(volume['volume'],face)})
                        truth,excluded=load_truth(folder)
                        children=candidate_children(truth,faces,t[:3,:3],t[:3,3]) if trusted else []
                        report={'name':name,'alignment':pose,'initial_alignment_trials':trials,
                            'pose_verified_for_reference':trusted,'outward_face_count':sum(f['outward']['verified'] for f in faces),
                            'planar_face_count':len(faces),'children':children,'excluded_truth_geometry':excluded,
                            'child_count':len(children),'all_five_sample_unique_consistent':sum(r['state']=='geometric_candidate_all_samples' for r in children)}
                    report.update(step=str(step),step_sha256=sha256(step),pcd_sha256=sha256(pcd),sources=sources,
                        status='DERIVED_REFERENCE_CANDIDATES_NOT_REVIEWED_TRUTH',
                        side_convention='Distinct solids, perpendicular material-outward faces, positive cross(n1,n2) dot directed tangent; customer side freezing pending.',
                        runtime_step_dependency=False,elapsed_seconds=time.perf_counter()-start)
                except Exception as exc:
                    report={'name':name,'sources':sources,'status':'REFERENCE_UNRESOLVED','error':type(exc).__name__+': '+str(exc),
                            'pose_verified_for_reference':False,'child_count':0,'all_five_sample_unique_consistent':0,
                            'elapsed_seconds':time.perf_counter()-start}
                path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
            summary={k:report.get(k) for k in ['name','status','pose_verified_for_reference','child_count','all_five_sample_unique_consistent','elapsed_seconds']}
            if 'alignment' in report:
                summary.update({k:report['alignment'][k] for k in ['point_to_cad_median_mm','point_to_cad_p95_mm','point_to_cad_max_mm','within_1mm_rate']})
            summaries.append(summary)
            (out/'progress.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf8')
            print('STEP_REFERENCE',json.dumps(summary,ensure_ascii=False),flush=True)
    finally:
        gmsh.finalize()
    (out/'complete.json').write_text(json.dumps({'sources':sources,'pieces':summaries,'full_contract_pass':False,
        'reviewed_customer_truth':False,'runtime_step_dependency':False},ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
