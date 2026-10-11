"""Recover reference poses using plate directions, never weld truth coordinates."""
import itertools
import json
import time
from pathlib import Path

import gmsh
import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.audit_remaining_step_references import candidate_children
from weld_pipeline.audit_step_weld_contacts import outward
from weld_pipeline.point_frame import dominant_xy_frame
from weld_pipeline.refine_step_reference_pose import refinement
from weld_pipeline.truth import load_truth


def main():
    root=Path('weld_pipeline/reports/all_step_references_20260930')
    out=Path('weld_pipeline/reports/step_reference_pose_repair_v1');out.mkdir(parents=True,exist_ok=True)
    gmsh.initialize([],readConfigFiles=False);gmsh.option.setNumber('General.Terminal',0)
    gmsh.option.setString('Geometry.OCCTargetUnit','MM')
    rows=[]
    try:
        for name in ['222-TT1A','232-TT1A']:
            start=time.perf_counter();folder=Path('DataSet/9月20日')/name
            step=next(folder.glob('*.stp'));pcd=next(folder.glob('*AllFaces.pcd'))
            archive=np.load(root/(name+'_mesh.npz'))
            vertices,triangles=archive['vertices'],archive['triangles']
            cad=json.loads((Path('weld_pipeline/reports/step_geometry_20260930')/(name+'.json')).read_text('utf8'))
            normal=np.array(max(cad['volumes'],key=lambda v:v['volume_mm3'])['parallel_major_face_pairs'][0]['normal_axis'])
            candidates=[f for v in cad['volumes'] for f in v['planar_faces']
                        if abs(np.array(f['parametric_normal'])@normal)<.01]
            tangent=np.array(max(candidates,key=lambda f:f['area_mm2'])['parametric_normal'])
            tangent-=normal*(tangent@normal);tangent/=np.linalg.norm(tangent)
            basis=np.column_stack([tangent,np.cross(normal,tangent),normal])
            xyz=read_xyz_pcd(pcd)*1000.;xy_frame=dominant_xy_frame(xyz)
            world=np.eye(3);world[:2,:2]=xy_frame.T
            rng=np.random.default_rng(132);sample=xyz[rng.choice(len(xyz),min(70000,len(xyz)),replace=False)]
            trials=[]
            for swap in [False,True]:
                permute=np.eye(3)[:,[1,0,2]] if swap else np.eye(3)
                for signs in itertools.product([-1.,1.],repeat=3):
                    middle=permute@np.diag(signs)
                    if np.linalg.det(middle)<0:
                        continue
                    r=world@middle@basis.T
                    v=vertices@r.T
                    t=np.eye(4);t[:3,:3]=r
                    t[:3,3]=(xyz.min(0)+xyz.max(0)-v.min(0)-v.max(0))/2.
                    trial=refinement(vertices,triangles,t,sample,normal)
                    trials.append(trial)
            trials.sort(key=lambda p:(p['point_to_cad_p95_mm'],p['point_to_cad_median_mm']))
            pose=trials[0];trusted=pose['point_to_cad_p95_mm']<=.05 and pose['within_1mm_rate']>=.995
            transform=np.array(pose['transform'])
            gmsh.clear();gmsh.model.occ.importShapes(str(step.resolve()));gmsh.model.occ.synchronize()
            faces=[]
            for volume in cad['volumes']:
                for face in volume['planar_faces']:
                    faces.append({**face,'volume':volume['volume'],'board_name':volume['name'],
                        'thickness':volume['supported_thickness_mm'],'outward':outward(volume['volume'],face)})
            truth,excluded=load_truth(folder)
            children=candidate_children(truth,faces,transform[:3,:3],transform[:3,3]) if trusted else []
            report={'name':name,'alignment':pose,'trials':trials,'pose_verified_for_reference':trusted,
                'children':children,'child_count':len(children),'all_five_sample_unique_consistent':sum(r['state']=='geometric_candidate_all_samples' for r in children),
                'status':'DERIVED_REFERENCE_CANDIDATES_NOT_REVIEWED_TRUTH' if trusted else 'REFERENCE_UNRESOLVED',
                'step_sha256':sha256(step),'pcd_sha256':sha256(pcd),'source_sha256':sha256(__file__),
                'method':'Native CAD transverse material face axis matched to PCD dominant plate grid, eight handed gravity-compatible frames, surface-only refinement.',
                'weld_truth_used_for_pose':False,'reviewed_customer_truth':False,'full_contract_pass':False,
                'elapsed_seconds':time.perf_counter()-start}
            (out/(name+'_attribute_candidates.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
            np.savez_compressed(out/(name+'_mesh.npz'),vertices=vertices,triangles=triangles,owners=archive['owners'],
                face_ids=archive['face_ids'],aligned_vertices=vertices@transform[:3,:3].T+transform[:3,3],transform=transform)
            row={k:report[k] for k in ['name','pose_verified_for_reference','child_count','all_five_sample_unique_consistent','elapsed_seconds']}
            row['point_to_cad_p95_mm']=pose['point_to_cad_p95_mm'];rows.append(row)
            print('STEP_REFERENCE_REPAIR',json.dumps(row),flush=True)
    finally:
        gmsh.finalize()
    (out/'complete.json').write_text(json.dumps({'rows':rows,'full_contract_pass':False},indent=2),encoding='utf8')


if __name__=='__main__':
    main()
