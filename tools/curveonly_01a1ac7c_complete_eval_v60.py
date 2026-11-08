"""Freeze a complete-circle geometry correction and replay all old models."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,types,shutil,time
import numpy as np
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_complete_eval_v36 as base
from tools import curveonly_01a1ac7c_complete_eval_adapter_v37 as adapter
from weld_pipeline import curveonly_01a1ac7c_complete_units_v60 as geometry
R,O=e.REC,e.OUT;DEST=R/'complete_curve_primary_v60'


def prepare():
    assert e.read(R/'closed_coverage_v59/RESULT.json')['complete']
    DEST.mkdir(exist_ok=True);p=e.read(base.DEST/'FROZEN_PROTOCOL.json')
    p.update(version=geometry.VERSION,policy=geometry.POLICY,
        previous_frozen_version=e.bind(base.DEST/'FROZEN_PROTOCOL.json'),
        TRAIN_geometry_evidence=e.bind(R/'closed_coverage_v59/RESULT.json'),
        raw_representation_evidence=e.bind(R/'round25/run/REPRESENTATION_FIRST_FAILURE_ADDENDUM_v58.json'),
        sources=p['sources']+[e.bind(__file__),e.bind(geometry.__file__),e.bind(adapter.__file__)],
        single_correction='Originalanalyticcirclecomplementaryhalves canform onecompleteclosedgeometry whenonly blocker is inferredSTEPcarrierowner. Bothendpointpairs<=.001mm, sameanalyticcircle<=.001mm, single2piturn andoppositetangents, mutuallyunique. Explicitconfirmedphysicaljoint/side/type/processcutsstillblock. No openpath ownerexception or toleranceincrease.',
        rationale='Userconfirmedcompletecurveprimary. TRAINrawCBFcontinuouspositivecirclecoverage andexactends show inferrednativeowner isnot certifiedphysicaljoint. Existingrules sayoneparentpath neednotone materialpair. Allrawowners/cuts andprocessUNKNOWN retained. Geometrynormalizationdoesnotacceptprocessidentity.',
        continuation='V36 nativepath andfallbackrules unchanged except strict complementarycircle inferredowner exception. Alloriginalpoints/corners kept.',
        unit_change_and_duplicate_weight_change_not_model_gain=True,
        no_model_or_inference_change=True,no_validation_fitting_mining_or_calibration=True,time_optimization=False,epoch=time.time())
    e.write(DEST/'FROZEN_PROTOCOL.json',p)
    for name in ('PRIMARY_INPUTS.json','USER_UNIT_CONFIRMATION.json'):shutil.copy2(base.DEST/name,DEST/name)


def check():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']:assert e.sha(b['path'])==b['sha256'],b['path']
    for key in ('user_confirmation','previous_frozen_version','TRAIN_geometry_evidence'):
        b=p[key];assert e.sha(b['path'])==b['sha256']
    return p


def annotate(c,*args,**kwargs):
    p=np.asarray(c['points']);out=base.annotate_one(dict(c,closed=bool(np.linalg.norm(p[0]-p[-1])<.1)),*args,**kwargs)
    return dict(out,_material_identity_source=geometry.OWNER_SOURCE)


def controls():
    check()
    def arc(a,b,owner,side=None):
        t=np.linspace(a,b,201);p=np.c_[50*np.cos(t),50*np.sin(t),np.zeros(len(t))]
        return dict(points=p.tolist(),weld_type='horizontal',weld_side=side,circle=e.core.circle_fit(p),
            _material=owner,_carrier=owner,_material_identity_source=geometry.OWNER_SOURCE)
    a,b=arc(0,np.pi,0),arc(np.pi,2*np.pi,1)
    c=geometry.units([a,b]);assert len(c)==1 and c[0]['closed']
    assert len(geometry.units([dict(a,_material_identity_source='CONFIRMED_CUSTOMER_JOINT'),b]))==2
    assert len(geometry.units([dict(a,_confirmed_joint_id='A'),dict(b,_confirmed_joint_id='B')]))==2
    assert len(geometry.units([a,dict(b,_confirmed_process_cut_ends=[0,1])]))==2
    assert len(geometry.units([a,dict(b,weld_side='B')]))==2
    assert len(geometry.units([a,arc(np.pi+.01,2*np.pi-.01,1)]))==2
    assert len(geometry.units([a,arc(0,np.pi,1)]))==2
    assert len(geometry.units([a,b,dict(b,_material=2,_carrier=2)]))==3
    assert sum(len(v['members']) for v in c)==2
    e.write(DEST/'CONTROLS.json',dict(passed=True,exact_complementary_circle_inferred_owners_only=True,
        confirmed_joints_and_cuts_known_sides_true_gaps_duplicates_and_branches_preserved=True,
        all_original_points_and_members_retained=True,no_geometry_tolerance_increase=True,no_VALlabels_read=True))
    print('COMPLETE_V60_CONTROLS_PASSED',flush=True)


def truths():
    types.FunctionType(base.truths.__code__,dict(base.__dict__,DEST=DEST,geometry=geometry,check=check,annotate_one=annotate))()
    index=e.read(DEST/'TRUTH_INDEX.json');ids={c['source_id'] for c in e.read(DEST/'PRIMARY_INPUTS.json')['cases']}
    e.write(DEST/'PRIMARY_TRUTH_INDEX.json',dict(index,cases=[c for c in index['cases'] if c['source_id'] in ids]))


def evaluate(name):
    assert e.read(DEST/'CONTROLS.json')['passed']
    types.FunctionType(base.evaluate.__code__,dict(base.__dict__,DEST=DEST,geometry=geometry,check=check,annotate_one=annotate))(name)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='controls':controls()
    elif action=='truths':truths()
    elif action=='evaluate':evaluate(sys.argv[2])
    else:raise ValueError(action)
