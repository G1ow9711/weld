"""Fixed FIT physical references, endpoint perturbations, no learned geometry input."""
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.continuous_contextual_height_model import VERSION, FEATURE_COUNT, predict, reference_features
from weld_pipeline.contact_plate_extent_model import physical_span_baseline
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates
from weld_pipeline.continuous_contact_sampling import ContinuousJointBases


def records(split,names,out,role,jitter):
    x,y,rows,expected,coverage=[],[],[],0,{}
    source=Path('weld_pipeline/runs/local_height_datum_seed140')
    for name in names:
        reference=source/('FIT_datum_'+name+'.json' if role=='FIT' else 'OBSERVED_CHECK_datum_'+name+'.json')
        if not reference.exists():
            coverage[name]=dict(state='NO_CURRENT_FIXED_CAD_REFERENCE');continue
        fixed=json.loads(reference.read_text('utf8'))
        piece=load_piece(split['piece_roots'][name],verify=True)
        if sha256(piece['record']['input'])!=piece['record']['pcd_sha256']:
            raise ValueError('Original point cloud changed')
        xyz=read_xyz_pcd(piece['record']['input'])*1000.
        profile,bases=FiniteContactPlates(xyz),ContinuousJointBases(xyz)
        context_path=Path(split['structure_roots'][name])/(name+'.npz')
        with np.load(context_path,allow_pickle=False) as context:
            structural,process=context['structural'],context['process']
        mapping=piece['record']['mapping']['line_mapping']['weld_to_sequence']
        parents={parent['id']:(index,parent) for index,parent in enumerate(piece['truth'])}
        count=Counter();local_rows=[]
        for row in fixed['rows']:
            expected+=1;count['expected_endpoints']+=1
            measurement=row['measurement']
            if measurement['state']!='LOCAL_CONTACT_HEIGHT_CANDIDATE' or 'pcd_normal' not in row:
                count['no_supported_fixed_candidate']+=1;continue
            parent_index,parent=parents[row['parent']]
            if str(parent_index) not in mapping:
                count['no_straight_PCD_contact_mapping']+=1;continue
            sequence_index=int(mapping[str(parent_index)])
            segment=parent['segments'][row['child']]
            if segment['kind']!='line':
                count['curve_role_not_frozen']+=1;continue
            points=np.array(segment['points'],float)
            inward=points[-1]-points[0] if row['endpoint']=='start' else points[0]-points[-1]
            inward/=np.linalg.norm(inward)
            q=np.array(row['xyz_mm']);normal=np.array(row['pcd_normal'])
            for offset in jitter:
                feature,diagnostic=reference_features(profile,bases,q+offset*inward,normal,inward,
                                                      structural[sequence_index],process[sequence_index])
                detail=dict(piece=name,parent=row['parent'],child=row['child'],endpoint=row['endpoint'],
                            jitter_mm=offset,reference_height_mm=measurement['height_mm'],
                            reviewed_customer_truth=False,diagnostic=diagnostic)
                if feature is None:
                    count['finite_reference_not_supported_'+str(offset)]+=1
                    detail['supported']=False
                else:
                    detail.update(supported=True,sample_index=len(x))
                    x.append(feature);y.append(measurement['height_mm']);rows.append(detail)
                    count['samples']+=1
                    if offset==0.:
                        count['supported_original_endpoints']+=1
                local_rows.append(detail)
        coverage[name]=dict(counts=dict(count),reference_sha256=sha256(reference),
                            context_sha256=sha256(context_path),pcd_sha256=piece['record']['pcd_sha256'])
        (out/(role+'_'+name+'_reference_support.json')).write_text(json.dumps(dict(coverage=coverage[name],rows=local_rows),indent=2,ensure_ascii=False),encoding='utf8')
        print(role,'CONTEXT_HEIGHT',name,count['samples'],'ORIGINAL',count['supported_original_endpoints'],'/',count['expected_endpoints'],flush=True)
    return np.asarray(x,np.float32).reshape(-1,FEATURE_COUNT),np.asarray(y,float),rows,expected,coverage


def score(bundle,x,y,rows,expected,coverage,out,role):
    predicted,baseline=predict(bundle,x)
    error=np.abs(predicted-y)
    original=np.array([row['jitter_mm']==0. for row in rows])
    endpoint_counts={key:dict(expected=sum(v.get('counts',{}).get('expected_endpoints',0) for v in coverage.values())//2,
                              supported=0,within_3=0) for key in ('start','end')}
    for row,accepted in zip(rows,error<=3.):
        if row['jitter_mm']==0.:
            endpoint_counts[row['endpoint']]['supported']+=1
            endpoint_counts[row['endpoint']]['within_3']+=int(accepted)
    result=dict(role=role,expected_original_endpoints=expected,supported_original_endpoints=int(original.sum()),
        original_within_3=int(((error<=3.)&original).sum()),endpoints=endpoint_counts,
        total_jitter_samples=len(x),jitter_within_3=int((error<=3.).sum()),
        original_physical_baseline_within_3=int(((np.abs(baseline-y)<=3.)&original).sum()),
        coverage=coverage,geometry_misses_included=False,customer_reviewed_truth=False,full_contract_pass=False)
    np.savez(out/(role+'_predictions.npz'),targets=y,predictions=predicted,errors=error,original=original)
    (out/(role+'_summary.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(role,'CONTEXT_HEIGHT_SCORE',result['original_within_3'],'/',expected,'ENDPOINTS',endpoint_counts,flush=True)
    return result


def main():
    started=time.perf_counter();out=Path('weld_pipeline/runs/continuous_local_height_seed146')
    if out.exists():
        raise ValueError('Fixed height trial exists')
    out.mkdir(parents=True)
    check=json.loads(Path('weld_pipeline/reports/continuous_sampling_20261001/synthetic_receipt.json').read_text('utf8'))
    if not check['passed'] or check['source_sha256']!=sha256('weld_pipeline/continuous_contact_sampling.py'):
        raise ValueError('Current continuous-sampling physical check required')
    split_path=Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split=json.loads(split_path.read_text('utf8'));names=[name for name in split['train'] if '__' not in name]
    sources=[__file__,'weld_pipeline/contextual_local_height_model.py','weld_pipeline/continuous_contact_sampling.py',
             'weld_pipeline/finite_joint_base_support.py','weld_pipeline/finite_contact_plate_profiles.py','weld_pipeline/point_plate_context.py']
    config=dict(version=VERSION,seed=146,features=FEATURE_COUNT,trees=128,max_features=.75,min_leaf=1,
        train=names,observed_check=split['validation'],jitter_mm=[-3.,-1.,0.,1.,3.],
        sources={path:sha256(path) for path in sources},split_sha256=sha256(split_path),
        targets='fixed140_same_joint_offline_CAD_candidates_NOT_REVIEWED',source_balance=True,
        input='39 physical + 50 point structure + 6 physical sizes + 3 unsigned canonical normal + 3 inward tangent',
        exclusions=['old140_predictions','old142_predictions','learned_geometry_predictions','part_names','absolute_xyz','runtime_CAD'],
        selection='Single fixed trial. Both original endpoint groups strictly >95% before observed check read.',
        continuous_sampling_geometry_tolerance_mm=3.,material_identity_verified=False,old_labels_and_weights_replaced=False,full_contract_pass=False)
    (out/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for path in sources:
        shutil.copy2(path,snapshot/Path(path).name)
    x,y,rows,expected,coverage=records(split,names,out,'FIT',config['jitter_mm'])
    if not len(y):
        raise ValueError('No supported physical supervision')
    np.savez(out/'FIT_training.npz',features=x,targets=y)
    counts=Counter(row['piece'] for row in rows)
    weights=np.array([1./counts[row['piece']] for row in rows])
    head=ExtraTreesRegressor(n_estimators=128,max_features=.75,min_samples_leaf=1,n_jobs=2,random_state=146)
    head.fit(x,y-physical_span_baseline(x[:,:39]),sample_weight=weights)
    bundle=dict(span_residual=head,config=config);joblib.dump(bundle,out/'heads.joblib')
    fit=score(bundle,x,y,rows,expected,coverage,out,'FIT')
    eligible=all(value['within_3']>.95*value['expected'] for value in fit['endpoints'].values())
    (out/'FIT_decision.json').write_text(json.dumps(dict(eligible_on_reference_FIT=eligible,fit=fit,
        check_used_for_decision=False,not_contract_acceptance=True),indent=2,ensure_ascii=False),encoding='utf8')
    print('CONTEXT_HEIGHT_FIT_DECISION',eligible,flush=True)
    check=None
    if eligible:
        x,y,rows,expected,coverage=records(split,split['validation'],out,'OBSERVED_CHECK',[0.])
        check=score(bundle,x,y,rows,expected,coverage,out,'OBSERVED_CHECK')
    result=dict(fit_summary=fit,observed_check_summary=check,
        check_status='POST_FIT_OBSERVED_NOT_BLIND' if eligible else 'NOT_RUN_FIT_REFERENCE_GATE_FAILED',
        heads_sha256=sha256(out/'heads.joblib'),elapsed_seconds=time.perf_counter()-started,
        original140_and_v4_unchanged=True,full_contract_pass=False)
    (out/'complete.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print('CONTEXT_HEIGHT_COMPLETE_FULL_CONTRACT_NOT_ACHIEVED',flush=True)


if __name__=='__main__':
    main()
