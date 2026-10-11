"""Free E16 estimates on each predicted path; no reference or cache reads."""
import csv,json,math
from pathlib import Path
import numpy as np
import torch
from weld_pipeline.contract_field_acceptance_297 import FIELDS
from weld_pipeline.legacy_frozen_incremental_model_392 import ATTRIBUTE_STATES
from weld_pipeline.legacy_incremental_correspondence_414 import canonical_child_fields

def values_at(sequence,position_mm):
    f=np.asarray(sequence['features'],np.float32)
    t=np.asarray(position_mm,np.float64)/float(sequence['sample_step_mm'])
    assert np.isfinite(t).all() and np.all(t>=-1e-7) and np.all(t<=len(f)-1+1e-7)
    t=np.clip(t,0.,len(f)-1);a=np.floor(t).astype(int);z=np.minimum(a+1,len(f)-1);w=(t-a).astype(np.float32)
    return (f[a]*(1.-w[:,None])+f[z]*w[:,None]).astype(np.float32)

def positions_on_own_candidate(sequence,points):
    p=np.asarray(points,np.float64)
    if sequence.get('curve_kind')=='arc':
        from weld_pipeline.curved_sequences import arc_positions
        return arc_positions(sequence,p)
    origin=np.asarray(sequence['xyz'][0],np.float64);direction=np.asarray(sequence['direction'],np.float64)
    t=(p-origin)@direction;error=np.linalg.norm(p-origin-t[:,None]*direction,axis=1)
    return t,error

def feature_path_for_own_prediction(segment,sequences,sequence_index=None):
    original=np.asarray(segment['points'],np.float64)
    index=np.linspace(0,len(original)-1,17)
    # Interpolate only our own ordered output; never change its endpoints.
    p=np.column_stack([np.interp(index,np.arange(len(original)),original[:,k]) for k in range(3)])
    choices=[]
    candidates=range(len(sequences)) if sequence_index is None else [sequence_index]
    for si in candidates:
        seq=sequences[si]
        if (seq.get('curve_kind')=='arc')!=(segment['kind']=='arc'):continue
        t,error=positions_on_own_candidate(seq,p);span=(len(seq['xyz'])-1)*float(seq['sample_step_mm'])
        if error.max()>.25 or t.min()<-1e-7 or t.max()>span+1e-7:continue
        choices.append((float(error.max()),si,t))
    if not choices:return None
    _,si,t=min(choices,key=lambda row:(row[0],row[1]))
    return dict(features=values_at(sequences[si],t),world_positions_mm=p,own_candidate_index=si,
        negative_chart_direction=bool(t[-1]<t[0]),
        source='CURRENT_PCD_FEATURES_AT_OWN_PREDICTED_PATH')

def decode_fields(output,row,scope='POINTWISE',whole_known_fields=None):
    logits=output['state_logits'][row].detach().cpu();prob=logits.softmax(-1).numpy()
    scalar=output['scalar_values'][row].detach().cpu().numpy()
    normal=output['signed_unit_normals'][row].detach().cpu().numpy();fields={}
    for i,name in enumerate(FIELDS):
        state=ATTRIBUTE_STATES[int(prob[i].argmax())];value=None;censor=None;reason=None
        if scope=='WHOLE_PATH' and whole_known_fields is not None and name not in whole_known_fields:
            state='unknown';reason='NO_WHOLE_LINE_SUPERVISION'
        if state=='value':
            if name.startswith('normal_'):
                n=normal[int(name[-1])-1]
                if np.isfinite(n).all() and abs(float(np.linalg.norm(n))-1.)<=.001:value=n.tolist()
                else:state='unknown';reason='PREDICTED_SIGNED_NORMAL_NOT_A_UNIT_VECTOR'
            elif name in ('hole_edge','segment_reason'):
                state='unknown';reason='STRUCTURAL_CATEGORY_DEFINITION_UNFROZEN'
            elif name in ('has_eave','side_plate_split'):value=bool(scalar[i]>=0.)
            elif math.isfinite(float(scalar[i])) and float(scalar[i])>=0.:value=float(scalar[i])
            else:state='unknown';reason='PREDICTED_DISTANCE_NOT_FINITE_NONNEGATIVE'
        elif state=='out_of_range':
            if name in ('front_wall','back_wall','start_base_edge','end_base_edge'):censor=dict(threshold_mm=50.,relation='gt',units='mm')
            else:state='unknown';reason='NO_BOUND_CENSOR_SEMANTICS_FOR_FIELD'
        elif state=='no_object' and name not in ('start_hole_height','end_hole_height'):
            state='unknown';reason='NO_BOUND_NO_OBJECT_SEMANTICS_FOR_FIELD'
        if state=='unknown' and reason is None:reason='LEARNED_UNKNOWN_STATE'
        fields[name]=dict(status=state,value=value,censor=censor,confidence=float(prob[i].max()),
            prediction_status='UNKNOWN' if state=='unknown' else 'MODEL_ESTIMATE',reference_review_status='UNCONFIRMED',
            source='LEARNED_FROZEN_PARENT_INCREMENTAL_HEAD',reason=reason,
            whole_curve_or_signed_side_acceptance_certified=False)
    return fields

@torch.no_grad()
def estimate_predicted_attributes(model,predictions,H_sequences,V_sequences):
    records=[]
    for parent in predictions:
        seqs=V_sequences if parent['type']=='vertical' else H_sequences
        forced=parent.get('original_candidate_sequence_index') if parent['type']=='vertical' else None
        for ci,segment in enumerate(parent['segments']):
            data=feature_path_for_own_prediction(segment,seqs,forced)
            record=dict(parent_id=parent['id'],child_index=ci,weld_side='unknown',formal_E_truth_certified=False,
                original_GT_or_STEP_reads=0,local_scope_not_expanded_to_whole_curve=True)
            if data is None:
                record.update(fields={name:dict(status='unknown',value=None,source=None,reason='OWN_PREDICTION_HAS_NO_ORIGINAL_FEATURE_CHART') for name in FIELDS},local_estimates=[])
            else:
                device=next(model.events.parameters()).device
                local=torch.from_numpy(data['features']).to(device)
                canonical=local.flip(0) if data['negative_chart_direction'] else local
                whole=model.path_attributes(canonical.reshape(1,-1));pointwise=model.attributes(local)
                record.update(fields=canonical_child_fields(decode_fields(whole,0,scope='WHOLE_PATH',whole_known_fields=model.config.get('whole_path_known_supervision_fields')),data['negative_chart_direction']),own_candidate_index=data['own_candidate_index'],
                    local_estimates=[dict(predicted_child_fraction=float(i/16),world_xyz_mm=data['world_positions_mm'][i].tolist(),
                        fields=decode_fields(pointwise,i),scope='POINTWISE_ESTIMATE_ONLY') for i in range(17)])
            records.append(record)
    return records

def export_predictions(predictions,attributes,out,workpiece_metadata):
    out=Path(out);by_key={(r['parent_id'],r['child_index']):r for r in attributes};rows=[];curves={}
    columns=['workpiece_id','weld_id','parent_id','segment_index','weld_type','geometry_type','weld_side',
        'is_closed','start_x','start_y','start_z','end_x','end_y','end_z','length_mm','ordered_points_mm','arc_parameters','curve_reference','confidence']
    columns.extend(name+'_'+part for name in FIELDS for part in ('status','value','censor','confidence','source','reason'))
    for parent in predictions:
        for ci,segment in enumerate(parent['segments']):
            p=np.asarray(segment['points'],np.float64);assert p.ndim==2 and p.shape[1]==3 and np.isfinite(p).all()
            sid=parent['id']+'_'+str(ci+1);attribute=by_key[(parent['id'],ci)]
            row=dict(workpiece_id=workpiece_metadata,weld_id=sid,parent_id=parent['id'],segment_index=ci,
                weld_type=parent['type'],geometry_type=segment['kind'],weld_side='unknown',
                is_closed=bool(np.linalg.norm(p[0]-p[-1])<=1e-6),
                **dict(zip(('start_x','start_y','start_z','end_x','end_y','end_z'),[*p[0],*p[-1]])),
                length_mm=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),
                ordered_points_mm=json.dumps(p.tolist()),arc_parameters=json.dumps(segment.get('arc_parameters')),
                curve_reference='child_curves.json#'+sid,confidence=parent.get('confidence'))
            for name in FIELDS:
                field=attribute['fields'][name]
                for part in ('status','value','censor','confidence','source','reason'):
                    value=field.get(part);row[name+'_'+part]=json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value
            rows.append(row);curves[sid]=dict(kind=segment['kind'],ordered_points_mm=p.tolist(),
                arc_parameters=segment.get('arc_parameters'),local_estimates=attribute['local_estimates'])
    with (out/'predicted_children.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    with (out/'child_curves.json').open('x',encoding='utf-8') as f:json.dump(curves,f,ensure_ascii=False,allow_nan=False)
    return len(rows)
