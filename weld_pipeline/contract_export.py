"""Child-row development export. Unknown attributes stay explicit, never zero."""
import csv
import json
from pathlib import Path
import numpy as np

ATTRIBUTES=['thickness_1','thickness_2','normal_1','normal_2','front_wall','back_wall',
    'start_height','end_height','start_hole_height','end_hole_height','start_base_edge','end_base_edge',
    'has_eave','side_plate_split']


def save_children(welds,outdir,workpiece_id):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    rows=[];curves={}
    for parent in welds:
        for index,segment in enumerate(parent['segments']):
            points=np.asarray(segment['points'],float)
            if not np.isfinite(points).all() or len(points)<2:raise ValueError('Invalid output coordinates')
            sid=str(parent['id'])+'_'+str(index+1)
            curves[sid]={'kind':segment['kind'],'ordered_points_mm':points.tolist()}
            row={'workpiece_id':workpiece_id,'weld_id':sid,'parent_id':parent['id'],'segment_index':index,
                'weld_type':parent['type'],'geometry_type':segment['kind'],'weld_side':'unknown',
                'is_closed':bool(np.linalg.norm(points[0]-points[-1])<1e-6),
                **dict(zip(['start_x','start_y','start_z','end_x','end_y','end_z'],[*points[0],*points[-1]])),
                'length_mm':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                'curve_reference':'child_curves.json#'+sid,'confidence':parent.get('confidence'),
                'status':'geometry_prediction_attributes_incomplete','reason_code':'ATTRIBUTE_ESTIMATOR_AND_SIDE_SCHEMA_PENDING'}
            for name in ATTRIBUTES:
                for field,value in [('value',None),('status','unknown'),('source',None),('confidence',None),
                    ('reason','ESTIMATOR_NOT_IMPLEMENTED')]:row[name+'_'+field]=value
            rows.append(row)
    fields=['workpiece_id','weld_id','parent_id','segment_index','weld_type','geometry_type','weld_side','is_closed',
        'start_x','start_y','start_z','end_x','end_y','end_z','length_mm','curve_reference','confidence','status','reason_code']
    fields += [name+'_'+field for name in ATTRIBUTES for field in ['value','status','source','confidence','reason']]
    with (out/'predicted_children.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    (out/'child_curves.json').write_text(json.dumps(curves,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'output_schema.json').write_text(json.dumps({'units':'mm','count_unit':'one true predicted geometric child per row',
        'attributes':ATTRIBUTES,'unknown_values':'Empty CSV value plus explicit unknown status; never valid or not-applicable.',
        'full_contract_output_compliant':False,'missing_features':['weld_side','reviewed segmentation reasons','attribute estimators'],
        'columns':fields},indent=2),encoding='utf8')
    return len(rows)
