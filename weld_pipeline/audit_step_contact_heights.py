"""Compare existing height labels to contacted CAD solids, without relabeling."""
import csv
import json
from pathlib import Path

import numpy as np

from training.labels import sha256


def main():
    source=Path('weld_pipeline/reports/endpoint_height_physics_20260930')
    prior=json.loads((source/'complete.json').read_text('utf8'))
    out=Path('weld_pipeline/reports/step_contact_height_audit_20260930');out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        raise ValueError('Frozen contact-height audit exists')
    summaries=[];all_rows=[]
    for piece in prior['pieces']:
        name=piece['name'];ref_path=Path('weld_pipeline/reports/step_contact_continuity_candidates_v1')/(name+'_attribute_candidates.json')
        if not ref_path.exists():
            summaries.append({'name':name,'status':'NO_THIS_ROUND_CAD_REFERENCE'});continue
        ref=json.loads(ref_path.read_text('utf8'))
        candidates=[Path(root)/(name+'_mesh.npz') for root in ['weld_pipeline/reports/step_reference_pose_repair_v1',
            'weld_pipeline/reports/all_step_references_20260930','weld_pipeline/reports/step_alignment_20260930']]
        mesh=next((p for p in candidates if p.exists()),None)
        if mesh is None:
            summaries.append({'name':name,'status':'NO_CACHED_CAD_MESH'});continue
        with np.load(mesh) as value:
            vertices=value['vertices'].copy();triangles=value['triangles'].copy();owners=value['owners'].copy()
        pose=np.array(ref['alignment']['transform']);vertices=vertices@pose[:3,:3].T+pose[:3,3]
        spans={}
        for volume in np.unique(owners):
            pts=vertices[np.unique(triangles[owners==volume])]
            spans[int(volume)]={'z_min_mm':float(pts[:,2].min()),'z_max_mm':float(pts[:,2].max()),'height_mm':float(np.ptp(pts[:,2]))}
        physics=json.loads((source/(name+'.json')).read_text('utf8'))
        labels={(r['parent'],r['child'],r['field']):r for r in physics['rows']}
        rows=[]
        for child in ref['children']:
            for field,sample_index in [('start_height',0),('end_height',-1)]:
                key=(child['weld_id'],child['child_index'],field)
                if key not in labels:
                    continue
                label=labels[key];pairs=child['samples'][sample_index]['candidate_pairs']
                if len(pairs)!=1:
                    continue
                comparison=[]
                for side,board in enumerate(pairs[0]):
                    volume=int(board['volume']);height=spans[volume]
                    comparison.append({'side':side+1,'volume':volume,'board_name':board['board_name'],
                        'contact_normal_z':float(board['outward_normal_world'][2]),**height,
                        'label_difference_mm':abs(height['height_mm']-label['target_mm'])})
                vertical=[c for c in comparison if abs(c['contact_normal_z'])<.4]
                within=any(c['label_difference_mm']<=3. for c in vertical)
                row={'piece':name,'parent':key[0],'child':key[1],'field':field,'type':label['type'],
                    'height_label_mm':label['target_mm'],'xyz_mm':label['xyz_mm'],'contact_boards':comparison,
                    'vertical_contact_board_height_within_3':within,'reviewed_customer_truth':False,
                    'note':'Solid vertical span is a diagnostic measurement candidate, not a replacement height definition.'}
                rows.append(row);all_rows.append(row)
        summary={'name':name,'comparable_endpoints':len(rows),'contacted_vertical_solid_span_within_3':sum(r['vertical_contact_board_height_within_3'] for r in rows),
            'mesh_sha256':sha256(mesh),'reference_sha256':sha256(ref_path),'measurement_definition_frozen':False}
        summaries.append(summary)
        (out/(name+'.json')).write_text(json.dumps({'summary':summary,'rows':rows},indent=2,ensure_ascii=False),encoding='utf8')
        print('CAD_HEIGHT_COMPARE',name,summary['contacted_vertical_solid_span_within_3'],'/',len(rows),flush=True)
    table=[]
    for r in all_rows:
        for side in r['contact_boards']:
            table.append({'piece':r['piece'],'parent':r['parent'],'child':r['child'],'field':r['field'],'type':r['type'],
                'original_height_mm':r['height_label_mm'],'board_side':side['side'],'board_name':side['board_name'],
                'contact_normal_z':side['contact_normal_z'],'solid_z_min_mm':side['z_min_mm'],'solid_z_max_mm':side['z_max_mm'],
                'solid_vertical_span_mm':side['height_mm'],'difference_mm':side['label_difference_mm'],'review_status':'NOT_REVIEWED_NO_RELABELING'})
    with (out/'原板高标签与实际接触STEP板件_核对.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (out/'complete.json').write_text(json.dumps({'pieces':summaries,'endpoints':len(all_rows),
        'matched_candidate_spans':sum(r['vertical_contact_board_height_within_3'] for r in all_rows),
        'scope':'FIT-only candidate reference audit; no label deletion or value change; not full E acceptance.',
        'source_hashes':{__file__:sha256(__file__)},'full_contract_pass':False},indent=2,ensure_ascii=False),encoding='utf8')


if __name__=='__main__':
    main()
