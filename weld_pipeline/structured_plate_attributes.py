"""Learn thickness from contact distances and plate structure, not coordinates."""
import numpy as np

from weld_pipeline.learned_plate_attributes import at_positions, features as direction_features, CONFIDENCE_THRESHOLD

VERSION='structured_point_plate_attributes_v2'


def thickness_features(row,structure):
    row=np.asarray(row,float)
    return np.r_[row[np.r_[0:49,70:84]],structure,abs(row[59]),row[62],row[65]].astype(np.float32)


def predict_samples(bundle,sequence,points,tangent):
    rows=at_positions(sequence,points)
    reg=np.stack([thickness_features(r,sequence['plate_context']) for r in rows])
    directions=np.stack([direction_features(r) for r in rows])
    thickness=bundle['thickness'].predict(reg);signs=bundle['material_sign'].predict(directions)
    confidence=np.min(np.stack([p.max(1) for p in bundle['material_sign'].predict_proba(directions)]),axis=0)
    result=[]
    for i,row in enumerate(rows):
        normals=row[60:66].reshape(2,3).astype(float);normals/=np.linalg.norm(normals,axis=1,keepdims=True)
        normals*=np.where(signs[i][:,None]>0,1.,-1.);t=thickness[i].copy()
        if np.cross(normals[0],normals[1])@tangent<0:
            normals=normals[::-1];t=t[::-1]
        valid=confidence[i]>=CONFIDENCE_THRESHOLD
        result.append({'state':'estimated_candidate' if valid else 'unknown',
            'reason':None if valid else 'LOW_MATERIAL_DIRECTION_CLASS_CONFIDENCE',
            'thickness_mm':t.tolist() if valid else None,'material_outward_normals':normals.tolist() if valid else None,
            'confidence':float(confidence[i]),'source':'Point_contact_and_plate_structure_supervised_by_unreviewed_CAD_candidates',
            'reviewed_truth_training':False,'customer_side_convention_confirmed':False})
    return result


def estimate_children(bundle,sequences,predictions):
    results=[]
    for w in predictions:
        for child,seg in enumerate(w['segments']):
            if seg['kind']!='line':
                results.append({'parent_id':w['id'],'child_index':child,'state':'unknown','reason':'CURVED_ATTRIBUTE_TRAINING_TRUTH_NOT_AVAILABLE'});continue
            p=np.array(seg['points'],float);tangent=p[-1]-p[0];tangent/=np.linalg.norm(tangent)
            options=[]
            for i,seq in enumerate(sequences):
                if seq['type']!=w['type'] or seq.get('curve_kind')=='arc':
                    continue
                q=seq['xyz'][0].astype(float);u=seq['direction'];t=(p-q)@u
                error=float(np.linalg.norm(np.cross(p-q,u),axis=1).max())
                if error<=.25 and t.min()>=-.01 and t.max()<=seq['sample_step_mm']*(len(seq['xyz'])-1)+.01:
                    options.append((error,i))
            if not options:
                results.append({'parent_id':w['id'],'child_index':child,'state':'unknown','reason':'CONTACT_CANDIDATE_NOT_ASSOCIATED'});continue
            _,i=min(options);points=np.array([p[0]*(1-f)+p[-1]*f for f in [0.,.25,.5,.75,1.]])
            samples=predict_samples(bundle,sequences[i],points,tangent)
            results.append({'parent_id':w['id'],'child_index':child,'state':'estimated_candidate' if all(s['state']=='estimated_candidate' for s in samples) else 'partial_or_unknown',
                'sample_fractions':[0.,.25,.5,.75,1.],'samples':samples,
                'side_convention':'cross(normal_1,normal_2) dot directed weld tangent positive','scope':'Development candidate, not E2 acceptance'})
    return results
