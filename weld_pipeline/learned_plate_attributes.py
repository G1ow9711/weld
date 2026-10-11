"""Shared point-profile attribute head; no CAD/reference access at inference."""
import numpy as np

VERSION='learned_cad_candidate_plate_attributes_v1'
CONFIDENCE_THRESHOLD=.8


def features(row):
    row=np.asarray(row,float)
    return np.r_[row[np.r_[0:49,70:84]],row[55:57],abs(row[59]),row[62],row[65]].astype(np.float32)


def at_positions(sequence,points):
    feature=np.asarray(sequence['features'])
    step=sequence['sample_step_mm']
    t=(np.asarray(points)-sequence['xyz'][0])@sequence['direction']/step
    if np.any(t<-.01) or np.any(t>len(feature)-1+.01):
        raise ValueError('Attribute sample lies outside point-derived candidate')
    t=np.clip(t,0.,len(feature)-1.)
    lo=np.floor(t).astype(int);hi=np.minimum(lo+1,len(feature)-1)
    w=t-lo
    return feature[lo]*(1-w[:,None])+feature[hi]*w[:,None]


def predict_samples(bundle,sequence,points,tangent):
    rows=at_positions(sequence,points)
    x=np.stack([features(row) for row in rows])
    thickness=bundle['thickness'].predict(x)
    signs=bundle['material_sign'].predict(x)
    probabilities=bundle['material_sign'].predict_proba(x)
    confidence=np.min(np.stack([p.max(1) for p in probabilities]),axis=0)
    output=[]
    for i,row in enumerate(rows):
        normals=row[60:66].reshape(2,3).astype(float)
        normals/=np.linalg.norm(normals,axis=1,keepdims=True)
        normals*=np.where(signs[i][:,None]>0,1.,-1.)
        side_thickness=thickness[i].copy()
        if np.cross(normals[0],normals[1])@tangent<0:
            normals=normals[::-1];side_thickness=side_thickness[::-1]
        valid=confidence[i]>=CONFIDENCE_THRESHOLD
        output.append({'state':'estimated_candidate' if valid else 'unknown',
                       'reason':None if valid else 'LOW_MATERIAL_DIRECTION_CLASS_CONFIDENCE',
                       'thickness_mm':side_thickness.tolist() if valid else None,
                       'material_outward_normals':normals.tolist() if valid else None,
                       'confidence':float(confidence[i]),'source':'Point_profiles_supervised_by_unreviewed_CAD_derived_TRAIN_candidates',
                       'reviewed_truth_training':False,'customer_side_convention_confirmed':False})
    return output


def estimate_children(bundle,sequences,predictions):
    results=[]
    for w in predictions:
        for child,seg in enumerate(w['segments']):
            if seg['kind']!='line':
                results.append({'parent_id':w['id'],'child_index':child,'state':'unknown','reason':'CURVED_ATTRIBUTE_TRAINING_TRUTH_NOT_AVAILABLE'})
                continue
            p=np.array(seg['points'],float)
            tangent=p[-1]-p[0];tangent/=np.linalg.norm(tangent)
            options=[]
            for i,seq in enumerate(sequences):
                if seq['type']!=w['type'] or seq.get('curve_kind')=='arc':
                    continue
                direction=seq['direction'];q=seq['xyz'][0].astype(float)
                length=seq['sample_step_mm']*(len(seq['xyz'])-1)
                t=(p-q)@direction
                error=float(np.linalg.norm(np.cross(p-q,direction),axis=1).max())
                if error<=.25 and t.min()>=-.01 and t.max()<=length+.01:
                    options.append((error,i))
            if not options:
                results.append({'parent_id':w['id'],'child_index':child,'state':'unknown','reason':'CONTACT_CANDIDATE_NOT_ASSOCIATED'})
                continue
            _,index=min(options)
            points=np.array([p[0]*(1-f)+p[-1]*f for f in [0.,.25,.5,.75,1.]])
            samples=predict_samples(bundle,sequences[index],points,tangent)
            results.append({'parent_id':w['id'],'child_index':child,'state':'estimated_candidate' if all(s['state']=='estimated_candidate' for s in samples) else 'partial_or_unknown',
                            'sample_fractions':[0.,.25,.5,.75,1.],'samples':samples,
                            'side_convention':'cross(normal_1,normal_2) dot directed weld tangent positive',
                            'scope':'Development candidate; not E2 acceptance'})
    return results
