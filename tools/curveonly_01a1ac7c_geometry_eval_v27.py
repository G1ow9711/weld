"""Frozen complete-geometry research view; original V2 remains authoritative."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,time,multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_score_v2 as old
from tools import curveonly_01a1ac7c_targets_v14 as targets
from weld_pipeline import curveonly_01a1ac7c_geometry_units_v26 as geometry
R,O=e.REC,e.OUT;DEST=R/'hierarchy_v27'


def source_groups(packet):
    if not packet['candidates']:return [],[]
    curves=[]
    for ci,c in enumerate(packet['candidates']):
        material=c.get('material',-1)
        curves.append(dict(c,points=c['points'],weld_type='INPUT_GEOMETRY_ONLY',weld_side=None,
            _material=material,_carrier='material_'+str(material)if material>=0 else'isolated_'+str(ci)))
    groups=geometry.units(curves);lookup={m['curve']:gi for gi,g in enumerate(groups)for m in g['members']}
    return [lookup[i]for i in range(len(curves))],groups


def annotate(curves,packet,assignments=None):
    chains,groups=source_groups(packet);out=[]
    for i,c in enumerate(curves):
        ci=assignments.get(i)if assignments is not None else c.get('source_candidate')
        valid=isinstance(ci,int)and 0<=ci<len(chains)
        out.append(dict(c,_carrier=chains[ci]if valid else None,
            _material=packet['candidates'][ci].get('material')if valid else None,
            _original_source_candidate=ci,source_identity_status='INPUT_CURVE_GEOMETRY_CHAIN_NOT_COMPLETE_JOINT_CERTIFICATE'))
    return out,groups


def prepare():
    DEST.mkdir(exist_ok=True)
    p=dict(version=geometry.VERSION,policy=geometry.POLICY,primary_truth=e.bind(R/'PRIMARY129_TRUTH_INDEX.json'),
        inputs=e.bind(R/'INPUTS.json'),source_inventory=e.bind(R/'SOURCE_INVENTORY.json'),
        source_geometry='Frozen independent R16 packets; input curve continuity may join native edges of same material. Other material identity and process applicability remain unverified except separately recorded proofs.',
        geometry_tolerance_mm=3.,sampling_mm=1.,original_V2_G=dict(TRAIN=211,VAL=85),original_V2_scores_unchanged=True,
        stage='DIAGNOSTIC_ONLY_NOT_CONTRACT_ACCEPTANCE',comparison='Freeze this independent rule then rescore all selected historical outputs; changed units alone are not model gain.',
        no_prediction_merging_using_truth=True,no_model_fit=True,no_scope_roles_or_raw_sources_changed=True,
        sources=[e.bind(__file__),e.bind(geometry.__file__),e.bind(old.__file__),e.bind(targets.__file__),e.bind(e.core.__file__)],
        maximum_workers=4,forecast_bytes_per_worker=1<<30,epoch=time.time())
    e.write(DEST/'FROZEN_PROTOCOL.json',p)


def check():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']:assert e.sha(b['path'])==b['sha256']
    return p


def controls():
    check()
    def arc(a,b):
        t=np.linspace(a,b,251);p=np.column_stack([50*np.cos(t),50*np.sin(t),np.zeros(len(t))])
        return dict(points=p.tolist(),circle=e.core.circle_fit(p),weld_type='horizontal',weld_side=None)
    a,b=arc(0,np.pi),arc(np.pi,2*np.pi);u=geometry.units([a,b]);assert len(u)==1 and u[0]['closed']
    assert len(geometry.units([a,a]))==2
    q=arc(np.pi+2/50,2*np.pi-2/50);assert len(geometry.units([a,q]))==2
    q=arc(np.pi+2/50,2*np.pi);gapped=geometry.units([a,q]);assert len(gapped)==1 and not gapped[0]['closed']
    assert gapped[0]['numerical_bridge_length_mm']<1e-8 and np.linalg.norm(np.array(gapped[0]['points'][0])-gapped[0]['points'][-1])>1.99
    assert len(geometry.units([dict(a,_material=0),dict(b,_material=1)]))==2
    assert len(geometry.units([dict(a,weld_side='A'),dict(b,weld_side='B')]))==2
    assert len(geometry.units([a,b,b]))==3
    assert sum(len(c['members'])for c in u)==2
    e.write(DEST/'CONTROLS.json',dict(passed=True,closed_loop_from_complementary_arcs=True,duplicates_retained=True,
        real2mm_gap_retained=True,different_materials_and_sides_not_joined=True,branch_ambiguity_not_joined=True,
        all_original_parts_and_boundaries_accounted=True,no_prediction_truth_inputs=True,source=e.bind(geometry.__file__)))
    print('CONNECTED_GEOMETRY_CONTROLS_PASSED',flush=True)


def one_truth(case):
    check();sid=case['source_id'];index=next(c for c in e.read(R/'PRIMARY129_TRUTH_INDEX.json')['cases']if c['source_id']==sid)
    truth=e.read(index['truth']['path']);packet=e.read(O/'round16/features'/sid/'geometry.json')
    saved={r['source_id']:r for r in e.read(R/'round09/ALL_TRAIN_TARGETS.json')['cases']}
    if case['split']=='TRAIN':possible=saved[sid]['chosen'];method='Own frozen TRAIN legal-interval assignments; source numerics unchanged'
    else:
        possible=[]
        for ci,c in enumerate(packet['candidates']):
            possible.extend(dict(candidate=ci,**r)for r in targets.legal_intervals(c,truth))
        possible=[min([r for r in possible if r['truth']==ti],key=lambda r:sum(r['error'].values()))for ti in range(len(truth['runs']))if any(r['truth']==ti for r in possible)]
        method='Postfreeze diagnostic full finite-carrier assignment; not fitting'
    assigned={int(r['truth']):int(r['candidate'])for r in possible};parts,chains=annotate(truth['runs'],packet,assigned);units=geometry.units(parts)
    path=DEST/'truth_proposals'/sid/'GEOMETRY_UNITS.json';e.write(path,dict(source_id=sid,split=case['split'],units=units,
        original_truth=index['truth'],original_records=len(parts),geometry_groups=len(units),source_chains=chains,
        original_assignments=possible,assignment_method=method,formal_joint_or_process_identity_accepted=False,
        all_original_curve_records_retained=sum(len(r['members'])for r in units)==len(parts)))
    return dict(source_id=sid,split=case['split'],truth=e.bind(path),original_records=len(parts),geometry_groups=len(units))


def truths():
    p=check();assert e.read(DEST/'CONTROLS.json')['passed'];cases=e.read(R/'INPUTS.json')['cases'];rows=[]
    now=int(Path('/sys/fs/cgroup/memory.current').read_text());limit=int(Path('/sys/fs/cgroup/memory.max').read_text());assert now+4*(1<<30)<.9*limit
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))as pool:
        for f in as_completed([pool.submit(one_truth,c)for c in cases]):
            rows.append(f.result());e.write(DEST/'TRUTH_PROGRESS.json',dict(complete=len(rows),total=len(cases),cases=rows));print('GEOMETRY_TRUTH_PROPOSED',rows[-1]['source_id'],rows[-1]['original_records'],rows[-1]['geometry_groups'],flush=True)
    e.write(DEST/'TRUTH_INDEX.json',dict(cases=rows,diagnostic_only=True,original_labels_unchanged=True,
        counts={split:dict(inputs=sum(c['split']==split for c in rows),original_records=sum(c['original_records']for c in rows if c['split']==split),geometry_groups=sum(c['geometry_groups']for c in rows if c['split']==split))for split in ('TRAIN','VAL')}))


def matching(pred,truth):
    cost=np.full((len(pred),len(truth)),1e9);edges={}
    for i,p in enumerate(pred):
        for j,t in enumerate(truth):
            if t.get('weld_side')is not None and p.get('weld_side')!=t['weld_side']:continue
            err=old.geometry_error(p,t)
            if err is not None:cost[i,j]=sum(err.values());edges[i,j]=err
    matches=[]
    if cost.size:
        pi,ti=old.linear_sum_assignment(cost)
        matches=[dict(prediction=int(i),truth=int(j),**edges[i,j])for i,j in zip(pi,ti)if(i,j)in edges]
    tp=len(matches)
    return dict(G=len(truth),P=len(pred),TP=tp,FP=len(pred)-tp,FN=len(truth)-tp,matches=matches)


def evaluate(name):
    check();root=R/name/'run';frozen=e.read(root/'OUTPUTS_FROZEN.json');index={r['source_id']:r for r in e.read(DEST/'TRUTH_INDEX.json')['cases']};cases=[]
    for f in frozen['cases']:
        assert e.sha(f['prediction']['path'])==f['prediction']['sha256'];value=e.read(f['prediction']['path']);original=[old.exported(p)for p in value['parents']]
        packet=e.read(O/name/'features'/f['source_id']/'geometry.json');parts,_=annotate(original,packet);prediction=geometry.units(parts)
        truth=e.read(index[f['source_id']]['truth']['path'])['units'];result=matching(prediction,truth)
        row=dict(source_id=f['source_id'],split=f['split'],metrics=result,prediction_groups=prediction,truth_binding=index[f['source_id']]['truth'],
            original_prediction_units=len(original),new_fitted_curves=0,input_failure=f.get('input_failure',False))
        e.write(DEST/'scores'/name/'cases'/f['source_id']/'SCORE.json',row);cases.append(row)
    totals={}
    for split in ('TRAIN','VAL'):
        q={k:sum(c['metrics'][k]for c in cases if c['split']==split)for k in ('G','P','TP','FP','FN')};q.update(extraction=q['TP']/q['G']if q['G']else None,FP_per_G=q['FP']/q['G']if q['G']else None);totals[split]=q
    e.write(DEST/'scores'/name/'RESULT.json',dict(version=geometry.VERSION,totals=totals,cases=[dict(source_id=c['source_id'],split=c['split'],metrics=c['metrics'])for c in cases],
        frozen_outputs=e.bind(root/'OUTPUTS_FROZEN.json'),protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),
        original_V2_unchanged=True,score_unit_change_not_model_gain=True,geometry_diagnostic_not_formal_acceptance=True,full_contract_pass=False))
    print('CONNECTED_GEOMETRY_SCORE',name,totals,flush=True)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='controls':controls()
    elif action=='truths':truths()
    elif action=='evaluate':evaluate(sys.argv[2])
    else:raise ValueError(action)
