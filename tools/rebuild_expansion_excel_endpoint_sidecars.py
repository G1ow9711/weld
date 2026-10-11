"""TRAIN-only Excel endpoint attribute evidence, separate from frozen geometry.

Internal child endpoints remain unknown. Parent-wide Any flags never become
per-child labels. Exported frames are preserved without inventing plate normals.
"""
from pathlib import Path
import math,collections
import numpy as np
import openpyxl
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.contract_curve_geometry import CircularArc
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';DATA=ROOT.parent/'data/G_Data';DEST=BASE/'excel_endpoint_sidecars_v1'
FIELDS={'start':{'blocker_distance':'FrontWallDistance','plate_height':'startHeight','hole_height':'StartViaHoleHeight','outer_edge_distance':'StartBasePlateEdgeDistance'},'end':{'blocker_distance':'BackWallDistance','plate_height':'endHeight','hole_height':'EndViaHoleHeight','outer_edge_distance':'EndBasePlateEdgeDistance'}}
def numeric(raw,field):
 if isinstance(raw,bool) or not isinstance(raw,(int,float)) or not math.isfinite(raw):return dict(raw_value=raw,known=False,reason='MISSING_OR_NONNUMERIC_EXPORT')
 if field=='blocker_distance' and raw==-1:return dict(raw_value=raw,known=True,encoding='NO_BLOCKER_OR_DISTANCE_GT_50',exact_distance_known=False)
 if field=='outer_edge_distance' and raw==51:return dict(raw_value=raw,known=True,encoding='DISTANCE_GT_50',exact_distance_known=False)
 if raw<0:return dict(raw_value=raw,known=False,reason='NEGATIVE_PLACEHOLDER_NOT_DISTANCE')
 return dict(raw_value=raw,known=True,encoding='EXACT_MM',exact_distance_known=True)
def endpoint(child,side):
 r=child['representation']
 if r['kind']=='line':return np.asarray(r['endpoints_mm'][0 if side=='start' else -1])
 a=CircularArc.from_parameters(r['arc_parameters']);return a.point_at(np.array([0. if side=='start' else a.length_mm]))[0]
def main():
 DEST.mkdir(exist_ok=False);groups={g['key']:g for g in read(BASE/'group_manifest.json')['groups']};summary=[]
 write(DEST/'protocol.json',dict(scope='Accepted expansion TRAIN only, source Excel evidence sidecars; existing geometry labels unchanged',internal_endpoint_policy='UNKNOWN, never copy parent endpoint values to interior children',parent_flags='Raw parent scope; Any does not label individual children',vertical_required_nongeometric_attributes=['end.blocker_distance'],wall_minus_one='No blocker OR >50, cannot disambiguate',outer_edge_51='>50, not an exact 51mm distance',frames='Exported endpoint frames retained and algebraically audited; no plate-normal truth inferred',DEV_reads=0,fit_calls=0))
 for accepted in read(BASE/'training_label_rebuild_summary.json')['successes']:
  key=accepted['source_key'];assert groups[key]['role']=='EXPANSION_TRAIN';label=BASE/'labels'/key;receipt=read(label/'receipt.json');target=read(label/'targets.json');assert digest(label/'targets.json')==receipt['targets_sha256'];xlsx=[(p,sha) for p,sha in receipt['source_files'].items() if p.endswith('.xlsx')];assert len(xlsx)==1;name,sha=xlsx[0];path=DATA/name;assert digest(path)==sha;book=openpyxl.load_workbook(path,read_only=True,data_only=False)
  try:
   it=book['焊缝信息'].iter_rows(values_only=True);headers=[str(v or '').split('\n')[-1] for v in next(it)];rows={str(r['ObjectId']):r for v in it if (r:=dict(zip(headers,v))).get('ObjectId')}
  finally:book.close()
  assert set(rows)=={p['original_parent_id'] for p in target['parents']};parents=[];counts=collections.Counter()
  for parent in target['parents']:
   pid=parent['original_parent_id'];row=rows[pid];assert int(row['SegmentCount'])==len(parent['children']);children=[]
   for i,child in enumerate(parent['children']):
    values={};frames={}
    for side,position in [('start',0),('end',len(parent['children'])-1)]:
     outer=i==position;prefix=side.title();values[side]={}
     for field,col in FIELDS[side].items():
      required=child['weld_type']=='horizontal' or (side=='end' and field=='blocker_distance');v=numeric(row.get(col),field) if outer else dict(known=False,reason='NO_CHILD_ENDPOINT_EXPORT');v.update(required_by_current_customer_rule=required,source_column=col,source_scope='PARENT_OUTER_ENDPOINT' if outer else 'UNAVAILABLE');values[side][field]=v;counts['known_required' if v['known'] and required else 'unknown_required' if required else 'optional']+=1
     if outer:
      t=np.array([row[prefix+'Matrix.Translation.'+axis] for axis in 'XYZ'],float);error=float(np.linalg.norm(t-endpoint(child,side)));assert error<1e-4;raw=[[row.get(prefix+'.UpperLeft.'+axis+'.'+coord) for coord in 'XYZ'] for axis in 'XYZ']
      # Correct export prefix is StartMatrix/EndMatrix, not Start/End.
      raw=[[row.get(prefix+'Matrix.UpperLeft.'+axis+'.'+coord) for coord in 'XYZ'] for axis in 'XYZ'];matrix=np.asarray(raw,float);finite=bool(np.isfinite(matrix).all());orth=float(np.max(abs(matrix@matrix.T-np.eye(3)))) if finite else None;det=float(np.linalg.det(matrix)) if finite else None;frames[side]=dict(raw_export_axes=raw,translation=t.tolist(),matched_endpoint_error_mm=error,orthonormal=bool(finite and orth<1e-6 and abs(det-1)<1e-6),max_orthogonality_error=orth,determinant=det,axis_sign_not_reinterpreted=True,plate_normals_not_inferred=True);counts['valid_frame' if frames[side]['orthonormal'] else 'invalid_frame']+=1
     else:frames[side]=dict(known=False,reason='NO_INTERIOR_CHILD_FRAME_EXPORT')
    children.append(dict(child_id=child['original_child_id'],index=i,weld_type=child['weld_type'],endpoint_attributes=values,endpoint_frames=frames))
   flags={k:row.get(k) for k in ('HasEave (Any)','IsWorkRangeSplit (Any)','WorkRangeFrontSplit','WorkRangeBackSplit','IsSplitedForDualRobot (Any)','HasMid (Any)')};parents.append(dict(parent_id=pid,raw_parent_scope_flags=flags,parent_flags_not_broadcast=True,children=children))
  dest=DEST/key;dest.mkdir();write(dest/'evidence.json',dict(key=key,parents=parents));out=dict(key=key,source_excel_sha256=sha,geometry_receipt_sha256=digest(label/'receipt.json'),evidence_sha256=digest(dest/'evidence.json'),counts=dict(counts),geometry_targets_unchanged=True,required_endpoint_numeric_attributes_complete=counts['unknown_required']==0,fit_calls=0,DEV_reads=0);write(dest/'receipt.json',out);summary.append(out);print(key,dict(counts),flush=True)
 write(DEST/'summary.json',dict(cases=summary,case_count=len(summary),fit_calls=0,DEV_reads=0,geometry_labels_not_mutated=True,sidecars_not_full_attribute_model=True))
if __name__=='__main__':main()
