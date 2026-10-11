"""FIT endpoint rejection: locate sample gaps relative to the actual child."""
import collections
import json
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates
from weld_pipeline.finite_joint_base_support import FiniteJointBases


def boundary_evidence(plane,q,inward):
    n,basis=plane['normal'],plane['basis']
    projected=q+n*(plane['offset']-q@n)
    centre=projected@basis.T
    u=inward@basis.T;u/=np.linalg.norm(u);v=np.array([-u[1],u[0]])
    ids=plane['tree'].query_ball_point(centre,48.)
    uv=plane['uv'][ids]
    relative=(uv-centre)@np.stack((u,v)).T
    nearest_ids=np.argsort(np.linalg.norm(relative,axis=1))[:12]
    boundary=relative[np.abs(relative[:,1])<=.3]
    along=np.unique(np.round(boundary[:,0],5)) if len(boundary) else np.array([])
    gaps=np.diff(along);gaps=gaps[gaps>1e-4]
    typical=float(np.median(gaps)) if len(gaps)>=3 else None
    before=boundary[boundary[:,0]<=0.];after=boundary[boundary[:,0]>=0.]
    bracket=None
    if len(before) and len(after):
        a=before[np.argmax(before[:,0])];b=after[np.argmin(after[:,0])]
        d=b-a
        ratio=float(np.clip(-a@d/(d@d),0.,1.)) if d@d>1e-12 else 0.
        distance=float(np.linalg.norm(a+ratio*d));length=float(np.linalg.norm(d))
        bracket=dict(before_local_mm=a.tolist(),after_local_mm=b.tolist(),
                     endpoint_to_boundary_segment_mm=distance,boundary_sample_gap_mm=length,
                     typical_local_boundary_gap_mm=typical,
                     gap_ratio_to_local_median=(length/typical if typical else None),
                     hole_continuity_verified=False,material_identity_verified=False)
    covariance=np.linalg.eigvalsh(np.cov(relative.T)).tolist() if len(relative)>2 else []
    return dict(plane_normal=n.tolist(),plane_offset_mm=plane['offset'],query_xyz_mm=q.tolist(),
        query_to_plane_mm=float(abs(q@n-plane['offset'])),nearest_sample_mm=float(plane['tree'].query(centre)[0]),
        nearest_samples_along_child_and_transverse_mm=relative[nearest_ids].tolist(),
        local_points=len(relative),local_covariance_eigenvalues=covariance,
        local_transverse_levels=len(np.unique(np.round(relative[:,1]/.15))),
        local_transverse_range_mm=(float(np.ptp(relative[:,1])) if len(relative) else None),
        boundary_row_points=len(boundary),boundary_bracket=bracket)


def main():
    out=Path('weld_pipeline/reports/contact_boundary_sampling_20261001')
    if out.exists():
        raise ValueError('Frozen boundary diagnostic already exists')
    out.mkdir(parents=True)
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    rows=[];inputs={}
    for name in ['203-TT1A','224-GR24A1','207-FR79M']:
        failures=json.loads((Path('weld_pipeline/runs/contextual_local_height_seed142')/('FIT_'+name+'_reference_support.json')).read_text('utf8'))['rows']
        targets=json.loads((Path('weld_pipeline/runs/local_height_datum_seed140')/('FIT_datum_'+name+'.json')).read_text('utf8'))['rows']
        indexed={(r['parent'],r['child'],r['endpoint']):r for r in targets}
        piece=load_piece(split['piece_roots'][name],verify=True)
        inputs[name]=dict(path=piece['record']['input'],sha256=sha256(piece['record']['input']))
        xyz=read_xyz_pcd(piece['record']['input'])*1000.
        bases=FiniteJointBases(xyz);profile=FiniteContactPlates(xyz)
        parents={p['id']:p for p in piece['truth']}
        for row in failures:
            if row['jitter_mm']!=0. or row['supported']:
                continue
            target=indexed[(row['parent'],row['child'],row['endpoint'])]
            q,n=np.array(target['xyz_mm']),np.array(target['pcd_normal'])
            points=np.array(parents[row['parent']]['segments'][row['child']]['points'])
            inward=points[-1]-points[0] if row['endpoint']=='start' else points[0]-points[-1]
            inward/=np.linalg.norm(inward)
            attempts=[boundary_evidence(p,q,inward) for p in bases.candidates(q,n)]
            attempts.sort(key=lambda item:np.hypot(item['query_to_plane_mm'],item['nearest_sample_mm']))
            f,observation=profile.observe(q,n)
            rows.append(dict(piece=name,parent=row['parent'],child=row['child'],endpoint=row['endpoint'],
                rejection=row['diagnostic']['reason'],standing_face_offline=target['standing_board'],
                offline_local_height_candidate_mm=target['measurement']['height_mm'],
                contact_component_hypothesis=observation['component'],
                component_span_candidates_mm=([c['vertical_span_mm'] for c in [observation['component']] if c] if observation['component'] else []),
                pure_PCD_boundary_evidence=attempts,source_review_status='NOT_REVIEWED'))
        if sha256(piece['record']['input'])!=inputs[name]['sha256']:
            raise ValueError('Original PCD changed during audit')
        print('BOUNDARY_SAMPLING',name,sum(r['piece']==name for r in rows),flush=True)
    nearest=[r['pure_PCD_boundary_evidence'][0] for r in rows if r['pure_PCD_boundary_evidence']]
    brackets=[r['boundary_bracket'] for r in nearest if r['boundary_bracket']]
    plausible=[r for r in brackets if r['endpoint_to_boundary_segment_mm']<=.3 and r['gap_ratio_to_local_median'] is not None and r['gap_ratio_to_local_median']<=1.8]
    result=dict(role='FIT_BOUNDARY_SAMPLING_DIAGNOSTIC',rows=rows,
        counts=dict(failed_original_endpoints=len(rows),endpoints_with_candidate_planes=len(nearest),
            nearest_sample_over_3mm=sum(r['nearest_sample_mm']>3. for r in nearest),
            nearest_local_only_line=sum(r['local_transverse_levels']<3 for r in nearest),
            boundary_bracket_available=len(brackets),regular_boundary_bracket_hypotheses=len(plausible)),
        regular_boundary_brackets_do_not_prove_material_identity_or_hidden_holes=True,
        inputs=inputs,source_sha256=sha256(__file__),observed_check_read=False,full_contract_pass=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print('BOUNDARY_SAMPLING_COMPLETE',json.dumps(result['counts']),flush=True)


if __name__=='__main__':
    main()
