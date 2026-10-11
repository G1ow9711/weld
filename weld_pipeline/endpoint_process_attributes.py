"""Candidate endpoint-only process estimates from point arrays, never labels."""
import numpy as np

from weld_pipeline.horizontal_patch_tree import equivalent_frame, patch_features

VERSION = 'point_endpoint_process_candidates_v1'
FIELD_FAMILY = {'front_wall':'wall', 'back_wall':'wall', 'start_height':'height', 'end_height':'height',
    'start_hole_height':'hole', 'end_hole_height':'hole', 'start_base_edge':'edge', 'end_base_edge':'edge',
    'has_eave':'eave', 'side_plate_split':'side_split'}


def endpoint_profile(sequence, point, toward_interior, structural, mean, covariance):
    q = np.asarray(point, float)
    direction = np.asarray(sequence['direction'], float)
    reverse = float(direction @ np.asarray(toward_interior)) < 0.
    dummy = np.zeros((len(sequence['features']),7),np.float32)
    seq, _ = equivalent_frame(sequence, dummy, reverse=reverse)
    t = float((q-np.asarray(seq['xyz'][0])) @ seq['direction'] / seq['sample_step_mm'])
    if t < -.01 or t > len(seq['features'])-1+.01:
        raise ValueError('Endpoint outside point-derived candidate')
    t = float(np.clip(t, 0., len(seq['features'])-1))
    a = int(np.floor(t)); b = min(a+1, len(seq['features'])-1)
    rows = patch_features(seq, [a,b], structural, mean, covariance)
    return rows[0]*(1.-(t-a))+rows[1]*(t-a)


def child_profiles(sequence, points, structural, mean, covariance):
    p = np.asarray(points,float)
    tangent = p[-1]-p[0]
    if np.linalg.norm(tangent) <= 1e-9:
        raise ValueError('Degenerate child')
    a = endpoint_profile(sequence,p[0],tangent,structural,mean,covariance)
    b = endpoint_profile(sequence,p[-1],-tangent,structural,mean,covariance)
    middle = (a+b)*.5
    return {field:(middle if family in ['eave','side_split'] else b if field.startswith(('end_','back_')) else a)
        for field,family in FIELD_FAMILY.items()}


def predict_field(bundle, field, x):
    family = FIELD_FAMILY[field]
    if family not in bundle['heads']:
        return {'state':'unknown','value':None,'reason':'NO_UNAMBIGUOUS_TRAINING_LABELS'}
    head = bundle['heads'][family]
    proba = head['state'].predict_proba(np.asarray(x)[None,:])[0]
    confidence = float(proba.max()); state = str(head['state'].classes_[proba.argmax()])
    if confidence < .8:
        return {'state':'unknown','value':None,'reason':'LOW_PROCESS_STATE_CONFIDENCE','confidence':confidence}
    if state == 'exact':
        value = float(head['value'].predict(np.asarray(x)[None,:])[0])
    elif state == 'no_hole':
        value = 0.
    elif state == 'over_50':
        value = 51.
    elif state in ['false','true']:
        value = state == 'true'
    else:
        raise ValueError('Unexpected learned process state')
    return {'state':state,'value':value,'confidence':confidence,'source':'PCD_profiles_supervised_by_explicit_FIT_Excel_endpoints',
        'scope':'child_endpoint_only_not_along_line_acceptance','side_schema_frozen':False}
