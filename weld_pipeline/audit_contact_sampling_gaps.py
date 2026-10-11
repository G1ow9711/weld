"""FIT-only: separate geometric endpoint tolerance from sample-point gaps."""
import json
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.finite_joint_base_support import FiniteJointBases
from weld_pipeline.local_base_sampling import sampling_fill_radius


def main():
    out=Path('weld_pipeline/reports/contact_sampling_gaps_20261001')
    if out.exists():
        raise ValueError('Fixed sampling diagnostic exists')
    out.mkdir(parents=True)
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    rows=[]
    for name in ['203-TT1A','224-GR24A1','207-FR79M']:
        failed_path=Path('weld_pipeline/runs/contextual_local_height_seed142')/('FIT_'+name+'_reference_support.json')
        failed=json.loads(failed_path.read_text('utf8'))['rows']
        fixed_path=Path('weld_pipeline/runs/local_height_datum_seed140')/('FIT_datum_'+name+'.json')
        fixed=json.loads(fixed_path.read_text('utf8'))['rows']
        indexed={(r['parent'],r['child'],r['endpoint']):r for r in fixed}
        piece=load_piece(split['piece_roots'][name],verify=True)
        parents={p['id']:p for p in piece['truth']}
        xyz=read_xyz_pcd(piece['record']['input'])*1000.
        planes=FiniteJointBases(xyz)
        for r in failed:
            if r['jitter_mm']!=0. or r['supported']:
                continue
            source=indexed[(r['parent'],r['child'],r['endpoint'])]
            q,n=np.array(source['xyz_mm']),np.array(source['pcd_normal'])
            points=np.array(parents[r['parent']]['segments'][r['child']]['points'])
            inward=points[-1]-points[0] if r['endpoint']=='start' else points[0]-points[-1]
            proposals=[]
            for p in planes.candidates(q,n):
                projection=q+p['normal']*(p['offset']-q@p['normal'])
                uvq=projection@p['basis'].T
                u=inward@p['basis'].T
                if np.linalg.norm(u)<1e-8:
                    continue
                u/=np.linalg.norm(u);v=np.array([-u[1],u[0]])
                radius,sampling=sampling_fill_radius(p['tree'],uvq,u,v)
                distance,index=p['tree'].query(uvq)
                proposals.append(dict(plane_offset_error_mm=float(abs(q@p['normal']-p['offset'])),
                    nearest_sample_in_plane_mm=float(distance),in_plane_fill_radius_mm=radius,
                    sample_statistics=sampling,points=len(p['uv'])))
            proposals.sort(key=lambda p:np.hypot(p['plane_offset_error_mm'],p['nearest_sample_in_plane_mm']))
            cad_n=np.array(source['standing_board']['outward_normal_world'])
            angle=float(np.rad2deg(np.arccos(np.clip(abs(n@cad_n)/(np.linalg.norm(n)*np.linalg.norm(cad_n)),0.,1.))))
            rows.append(dict(piece=name,parent=r['parent'],child=r['child'],endpoint=r['endpoint'],
                rejection_reason=r['diagnostic']['reason'],normal_unsigned_angle_to_offline_candidate_deg=angle,
                fixed_CAD_projection_mm=source['measurement']['section']['projection']['distance_mm'],
                pure_PCD_plane_proposals=proposals,source_review_status='NOT_REVIEWED'))
        print('CONTACT_GAPS',name,sum(r['piece']==name for r in rows),flush=True)
    result=dict(rows=rows,role='FIT_SAMPLING_OPERATOR_DIAGNOSTIC',check_truth_read=False,
        source_sha256=sha256(__file__),endpoint_reference_reviewed=False,
        sample_neighbour_distance_not_equated_to_geometry_error=True,full_contract_pass=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print('CONTACT_GAPS_COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    main()
