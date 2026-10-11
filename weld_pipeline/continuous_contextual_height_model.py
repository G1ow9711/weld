"""Fixed contextual height model with continuous sampled contact association."""
import numpy as np

from weld_pipeline.contact_plate_extent_model import physical_span_baseline
from weld_pipeline.contextual_local_height_model import combine_features, FEATURE_COUNT
from weld_pipeline.continuous_contact_sampling import project_contact

VERSION='continuous_contextual_local_height_v1'


def reference_features(profile,bases,point,normal,inward,structural,process):
    bound,diagnostic=project_contact(profile,bases,point,normal,inward)
    if bound is None:
        return None,diagnostic
    features,observation,projected=bound
    return combine_features(features,structural,process,normal,inward),diagnostic


def endpoint_inputs(profile,bases,point,normals,inward):
    q=np.asarray(point,float);normals=np.asarray(normals,float).reshape(2,3).copy()
    lengths=np.linalg.norm(normals,axis=1)
    if not np.isfinite(normals).all() or np.any(lengths<1e-8):
        return None,dict(state='unknown',value=None,reason='INVALID_PCD_CONTACT_NORMALS')
    normals/=lengths[:,None]
    standing=normals[np.abs(normals[:,2])<.4]
    bottom=normals[np.abs(normals[:,2])>=np.cos(np.deg2rad(3.))]
    if len(standing)!=1 or len(bottom)!=1:
        return None,dict(state='unknown',value=None,reason='LOCAL_HEIGHT_BOARD_ROLE_UNRESOLVED')
    bound,contact_diagnostic=project_contact(profile,bases,q,standing[0],inward)
    if bound is None:
        return None,dict(state='unknown',value=None,**contact_diagnostic)
    features,observation,projection=bound
    base,base_diagnostic=bases.support(q,bottom[0],inward)
    if base is None:
        return None,dict(state='unknown',value=None,reason='FINITE_JOINT_BASE_NOT_RESOLVED',
                         base_diagnostic=base_diagnostic,contact_diagnostic=contact_diagnostic)
    return features,dict(state='candidate_not_reviewed',value=None,version=VERSION,
        source='fresh_PCD_continuous_contact_and_same_joint_base_plus_frozen146_head',
        base=base,contact_diagnostic=contact_diagnostic,height_definition='实际接触板及局部底板基准',
        reason='CUSTOMER_TRUTH_REVIEW_AND_COMPLETE_GEOMETRY_ACCEPTANCE_PENDING',
        reviewed_customer_truth=False,finite_material_identity_independently_verified=False,
        full_contract_pass=False)


def predict(bundle,features):
    x=np.asarray(features,np.float32)
    if x.ndim!=2 or x.shape[1]!=FEATURE_COUNT or bundle['config']['version']!=VERSION:
        raise ValueError('Fixed continuous contextual height head required')
    baseline=physical_span_baseline(x[:,:39])
    return baseline+bundle['span_residual'].predict(x),baseline


def estimate_children(bundle,profile,bases,sequences,predictions,structural,process):
    results,features,pending=[],[],[]
    for weld in predictions:
        for index,segment in enumerate(weld['segments']):
            result=dict(parent_id=weld['id'],child_index=index,fields={})
            if weld['type']!='horizontal' or segment['kind']!='line':
                result['fields']={key:dict(state='unknown',value=None,reason='VERTICAL_OR_CURVE_HEIGHT_ROLE_NOT_FROZEN')
                                  for key in ('start_height','end_height')}
                results.append(result);continue
            points=np.array(segment['points'],float);options=[]
            for si,sequence in enumerate(sequences):
                if sequence['type']!='horizontal' or sequence.get('curve_kind')=='arc':
                    continue
                origin,direction=sequence['xyz'][0].astype(float),sequence['direction']
                positions=(points-origin)@direction
                error=float(np.linalg.norm(np.cross(points-origin,direction),axis=1).max())
                if error<=.25 and positions.min()>=-.01 and positions.max()<=sequence['sample_step_mm']*(len(sequence['xyz'])-1)+.01:
                    options.append((error,si))
            if not options:
                result['fields']={key:dict(state='unknown',value=None,reason='PREDICTED_LINE_NOT_ASSOCIATED_WITH_PCD_CONTACT')
                                  for key in ('start_height','end_height')}
                results.append(result);continue
            si=min(options)[1];sequence=sequences[si];tangent=points[-1]-points[0]
            for key,q,inward in [('start_height',points[0],tangent),('end_height',points[-1],-tangent)]:
                sample=int(np.clip(round(float((q-sequence['xyz'][0])@sequence['direction'])/sequence['sample_step_mm']),0,len(sequence['features'])-1))
                normals=sequence['features'][sample,60:66].reshape(2,3)
                local,record=endpoint_inputs(profile,bases,q,normals,inward)
                result['fields'][key]=record
                if local is not None:
                    standing=normals[np.abs(normals[:,2])<.4]
                    features.append(combine_features(local,structural[si],process[si],standing[0],inward));pending.append(record)
            results.append(result)
    if features:
        x=np.asarray(features,np.float32);values,baseline=predict(bundle,x)
        residual=bundle['span_residual'].predict(x)
        trees=np.stack([tree.predict(x) for tree in bundle['span_residual'].estimators_])
        agreement=np.mean(np.abs(trees-residual[None,:])<=3.,axis=0)
        for record,value,base,confidence in zip(pending,values,baseline,agreement):
            record.update(physical_span_baseline_mm=float(base),learned_residual_mm=float(value-base),
                confidence=float(confidence),confidence_definition='tree_agreement_within_3mm_not_acceptance_probability',
                confidence_calibrated=False)
            if value>0 and np.isfinite(value):
                record['value']=float(value)
            else:
                record.update(state='unknown',value=None,reason='INVALID_CONTEXTUAL_LOCAL_HEIGHT')
    return results
