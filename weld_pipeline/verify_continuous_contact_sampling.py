"""Physical checks on sample gaps, finite surfaces, and unchanged 3 mm limits."""
import json
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.continuous_contact_sampling import ContinuousJointBases, boundary_projection, VERSION
from weld_pipeline.finite_joint_base_support import FiniteJointBases
from weld_pipeline.verify_finite_joint_base_support import grid


def main():
    out=Path('weld_pipeline/reports/continuous_sampling_20261001')
    if out.exists():
        raise ValueError('Frozen continuous sampling receipt exists')
    out.mkdir(parents=True)
    ordinary=grid(np.arange(0.,52.,2.),np.arange(-32.,34.,2.))
    angle=np.deg2rad(1.)
    rotation=np.array([[np.cos(angle),0.,-np.sin(angle)],[0.,1.,0.],[np.sin(angle),0.,np.cos(angle)]])
    holed=ordinary[~((ordinary[:,0]>0.)&(ordinary[:,0]<26.)&(np.abs(ordinary[:,1])<24.))]
    islands=grid(np.r_[np.arange(-50.,-8.,2.),np.arange(10.,52.,2.)],np.arange(-32.,34.,2.))
    cases=[
        ('flat_finite_endpoint',ordinary,[0,0,0],[0,0,1],[1,0,0],True),
        ('tilted_endpoint',ordinary@rotation.T,[0,0,0],rotation@[0,0,1],rotation@[1,0,0],True),
        ('translated_tilted',ordinary@rotation.T+[31,-12,103],[31,-12,103],rotation@[0,0,1],rotation@[1,0,0],True),
        ('normal_sign',ordinary,[0,0,0],[0,0,-1],[1,0,0],True),
        ('eight_mm_board',grid(np.arange(0.,52.,2.),np.arange(0.,10.,2.)),[0,0,0],[0,0,1],[1,0,0],True),
        ('wrong_inward',ordinary,[0,0,0],[0,0,1],[-1,0,0],False),
        ('single_line',grid(np.arange(0.,52.,2.),[0]),[0,0,0],[0,0,1],[1,0,0],False),
        ('two_outline_rows',grid(np.arange(0.,52.,2.),[0,8]),[0,0,0],[0,0,1],[1,0,0],False),
        ('opening_both_sides',holed,[0,0,0],[0,0,1],[1,0,0],False),
        ('coplanar_islands',islands,[0,0,0],[0,0,1],[1,0,0],False),
        ('remote_material',ordinary+[40,0,0],[0,0,0],[0,0,1],[1,0,0],False),
        ('parallel_remote',ordinary+[0,0,8],[0,0,0],[0,0,1],[1,0,0],False),
        ('sparse_grid_interior',grid(np.arange(0.,64.,8.),np.arange(-32.,40.,8.)),[4,4,0],[0,0,1],[1,0,0],True),
        ('base_plane_error_2_99',ordinary,[0,0,2.99],[0,0,1],[1,0,0],True),
        ('base_plane_error_3_01',ordinary,[0,0,3.01],[0,0,1],[1,0,0],False),
    ]
    rows=[]
    for name,xyz,q,n,u,expected in cases:
        value,diagnostic=ContinuousJointBases(xyz).support(np.array(q,float),n,u)
        rows.append(dict(name=name,kind='base',expected=expected,actual=value is not None,
                         passed=(value is not None)==expected,value=value,diagnostic=diagnostic))
    contact=np.array([[x,0.,z] for x in np.arange(-48.,56.,8.) for z in [0.,8.,16.]])
    missing=contact[~((contact[:,0]>-16.)&(contact[:,0]<24.))]
    boundary_cases=[
        ('sparse_boundary_gap',contact,[4,0,0],True),
        ('contact_normal_error_2_99',contact,[4,2.99,0],True),
        ('contact_normal_error_3_01',contact,[4,3.01,0],False),
        ('long_missing_boundary',missing,[4,0,0],False),
        ('no_boundary_extrapolation',contact,[60,0,0],False),
        ('translated_boundary',contact+[19,-11,37],[23,-11,37],True),
    ]
    for name,xyz,q,expected in boundary_cases:
        planes=FiniteJointBases(xyz).candidates(np.array(q,float),[0,1,0])
        attempts=[];accepted=None
        for plane in planes:
            value,diagnostic=boundary_projection(plane,np.array(q,float),[1,0,0])
            attempts.append(diagnostic)
            if value is not None:
                accepted=value
        rows.append(dict(name=name,kind='contact_boundary',expected=expected,actual=accepted is not None,
                         passed=(accepted is not None)==expected,projection=None if accepted is None else accepted.tolist(),
                         diagnostics=attempts))
    result=dict(version=VERSION,passed=all(row['passed'] for row in rows),cases=rows,
        source_sha256=sha256('weld_pipeline/continuous_contact_sampling.py'),verifier_sha256=sha256(__file__),
        synthetic_physics_only=True,real_accuracy_not_measured=True,full_contract_pass=False)
    (out/'synthetic_receipt.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('CONTINUOUS_SAMPLING_PHYSICS',result['passed'],len(rows),flush=True)
    if not result['passed']:
        print('FAILED_CASES',[row['name'] for row in rows if not row['passed']],flush=True)
        raise ValueError('Continuous physical check failed; preserve receipt')


if __name__=='__main__':
    main()
