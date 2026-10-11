"""Shared height residual with point-derived contact context and inward direction."""
import numpy as np

from weld_pipeline.contact_plate_extent_model import physical_span_baseline
from weld_pipeline.local_height_datum_model_v4 import endpoint_inputs, project_contact

VERSION = 'contextual_finite_local_height_v1'
FEATURE_COUNT = 101


def combine_features(profile, structural, process, normal, inward):
    n, u = np.asarray(normal,float).copy(), np.asarray(inward,float).copy()
    if np.linalg.norm(n)<1e-8 or np.linalg.norm(u)<1e-8:
        raise ValueError('Nondegenerate height contact and inward directions required')
    n/=np.linalg.norm(n); u/=np.linalg.norm(u)
    if n[np.argmax(np.abs(n))]<0:
        n*=-1
    result=np.r_[profile,structural,process,n,u].astype(np.float32)
    if result.shape!=(FEATURE_COUNT,) or not np.isfinite(result).all():
        raise ValueError('Finite 101-dimensional local height input required')
    return result


def reference_features(profile,bases,point,normal,inward,structural,process):
    bound,diagnostic=project_contact(profile,bases,point,normal)
    if bound is None:
        return None,diagnostic
    features,observation,projected=bound
    return combine_features(features,structural,process,normal,inward),diagnostic


def predict(bundle,features):
    x=np.asarray(features,np.float32)
    if x.ndim!=2 or x.shape[1]!=FEATURE_COUNT or bundle['config']['version']!=VERSION:
        raise ValueError('Fitted contextual height head required')
    baseline=physical_span_baseline(x[:,:39])
    values=baseline+bundle['span_residual'].predict(x)
    return values,baseline


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
                position=(points-origin)@direction
                error=float(np.linalg.norm(np.cross(points-origin,direction),axis=1).max())
                if error<=.25 and position.min()>=-.01 and position.max()<=sequence['sample_step_mm']*(len(sequence['xyz'])-1)+.01:
                    options.append((error,si))
            if not options:
                result['fields']={key:dict(state='unknown',value=None,reason='PREDICTED_LINE_NOT_ASSOCIATED_WITH_PCD_CONTACT')
                                  for key in ('start_height','end_height')}
                results.append(result);continue
            si=min(options)[1];sequence=sequences[si];tangent=points[-1]-points[0]
            for field,q,inward in [('start_height',points[0],tangent),('end_height',points[-1],-tangent)]:
                sample=int(np.clip(round(float((q-sequence['xyz'][0])@sequence['direction'])/sequence['sample_step_mm']),0,len(sequence['features'])-1))
                normals=sequence['features'][sample,60:66].reshape(2,3)
                local,record=endpoint_inputs(profile,bases,q,normals,inward)
                result['fields'][field]=record
                if local is not None:
                    standing=normals[np.abs(normals[:,2])<.4]
                    features.append(combine_features(local,structural[si],process[si],standing[0],inward))
                    pending.append(record)
            results.append(result)
    if features:
        x=np.asarray(features,np.float32)
        values,baseline=predict(bundle,x)
        residual_predictions=np.stack([tree.predict(x) for tree in bundle['span_residual'].estimators_])
        agreement=np.mean(np.abs(residual_predictions-bundle['span_residual'].predict(x)[None,:])<=3.,axis=0)
        for record,value,base,confidence in zip(pending,values,baseline,agreement):
            record.update(version=VERSION,physical_span_baseline_mm=float(base),
                          learned_residual_mm=float(value-base),confidence=float(confidence),
                          confidence_definition='tree_agreement_within_3mm_not_calibrated_acceptance_probability',
                          confidence_calibrated=False)
            if value>0 and np.isfinite(value):
                record['value']=float(value)
            else:
                record.update(state='unknown',value=None,reason='INVALID_CONTEXTUAL_LOCAL_HEIGHT')
    return results
