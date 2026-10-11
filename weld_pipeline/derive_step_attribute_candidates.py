"""CAD-derived reference candidates with explicit, unreviewed side convention."""
import csv
import json
from pathlib import Path

import gmsh
import numpy as np

from training.labels import sha256
from weld_pipeline.audit_step_weld_contacts import contacts, outward
from weld_pipeline.truth import load_truth


def pairs(found, tangent):
    options = []
    for i,a in enumerate(found):
        for b in found[i+1:]:
            if a['volume']==b['volume'] or not a['outward_verified'] or not b['outward_verified']:
                continue
            n1,n2 = [np.array(p['outward_normal_world']) for p in [a,b]]
            if abs(n1@n2) > .2:
                continue
            first,second = (a,b) if np.cross(n1,n2)@tangent > 0 else (b,a)
            options.append([first,second])
    return options


def main():
    root = Path('weld_pipeline/reports/step_alignment_20260930')
    gmsh.initialize([], readConfigFiles=False)
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setString('Geometry.OCCTargetUnit', 'MM')
    try:
        for name in ['224-GR24A1','527-FR146A']:
            folder = Path('DataSet/9月20日')/name
            step = next(folder.glob('*.stp'))
            pose_file = root/(name+'_refined.json')
            transform = np.array(json.loads(pose_file.read_text('utf8'))['trials'][0]['transform'])
            rotation,translation = transform[:3,:3],transform[:3,3]
            cad = json.loads((Path('weld_pipeline/reports/step_geometry_20260930')/(name+'.json')).read_text('utf8'))
            gmsh.clear()
            gmsh.model.occ.importShapes(str(step.resolve()))
            gmsh.model.occ.synchronize()
            faces = []
            for volume in cad['volumes']:
                for face in volume['planar_faces']:
                    faces.append({**face,'volume':volume['volume'],'board_name':volume['name'],
                        'thickness':volume['supported_thickness_mm'],'outward':outward(volume['volume'],face)})
            truth,excluded = load_truth(folder)
            rows,output = [],[]
            for weld in truth:
                for index,seg in enumerate(weld['segments']):
                    p = np.array(seg['points'])
                    if seg['kind']!='line':
                        continue
                    tangent = p[-1]-p[0]
                    tangent /= np.linalg.norm(tangent)
                    samples = []
                    for fraction in [0.,.25,.5,.75,1.]:
                        q = p[0]*(1.-fraction)+p[-1]*fraction
                        found = contacts(q,faces,rotation,translation)
                        options = pairs(found,tangent)
                        samples.append({'fraction':fraction,'xyz_mm':q.tolist(), 'contact_count':len(found),
                                        'candidate_pairs':options,'unique_pair':len(options)==1})
                    unique = [s for s in samples if s['unique_pair']]
                    identities = {tuple(p['volume'] for p in s['candidate_pairs'][0]) for s in unique}
                    consistent = len(identities)==1 and len(unique)==5
                    pair = next((s['candidate_pairs'][0] for s in unique if s['fraction']==.5),None)
                    if pair is None and unique:
                        pair = unique[0]['candidate_pairs'][0]
                    state = 'geometric_candidate_all_samples' if consistent else 'geometric_candidate_partial_or_ambiguous'
                    if pair is None:
                        state = 'no_unique_contact_pair'
                    number = weld['metadata'].get('WeldNumber (f/v + Name number)')
                    result = {'weld_id':weld['id'],'weld_number':number,'child_index':index,
                              'type':weld['type'],'state':state,'pair_at_mid_or_nearest_sample':pair,'samples':samples,
                              'reviewed_customer_truth':False}
                    output.append(result)
                    row = {'piece':name,'weld_number':number,'parent_id':weld['id'],'child_index':index,
                           'type':weld['type'],'candidate_state':state,'unique_samples':len(unique),
                           'required_samples':5,'reviewed_customer_truth':False}
                    for i in [0,1]:
                        side = pair[i] if pair else {}
                        row.update({f'board_{i+1}_candidate':side.get('board_name'),
                                    f'thickness_{i+1}_candidate_mm':side.get('thickness_candidate_mm'),
                                    f'normal_{i+1}_outward_candidate':json.dumps(side.get('outward_normal_world'))})
                    rows.append(row)
            report = {'name':name,'step':str(step),'step_sha256':sha256(step),
                      'pose_sha256':sha256(pose_file),'source_sha256':sha256(__file__),
                      'status':'DERIVED_REFERENCE_CANDIDATES_NOT_REVIEWED_TRUTH',
                      'side_convention':'Choose distinct solids with perpendicular outward faces; order so cross(n1,n2) dot directed weld tangent is positive. Customer side/material-direction convention remains unconfirmed.',
                      'inference_step_used':False,'children':output,
                      'child_count':len(rows),'all_five_sample_unique_consistent':sum(r['candidate_state']=='geometric_candidate_all_samples' for r in rows)}
            (root/(name+'_attribute_candidates.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
            with (root/(name+'_attribute_candidates.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            print('DERIVED_REFERENCE',name,len(rows),'all_samples_consistent',report['all_five_sample_unique_consistent'],flush=True)
    finally:
        gmsh.finalize()


if __name__=='__main__':
    main()
