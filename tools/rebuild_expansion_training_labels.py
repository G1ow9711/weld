"""Rebuild only frozen expansion TRAIN labels; JSON geometry checked against PLY/Excel.

Cases with unresolved representation or metadata conflicts are quarantined, never
silently simplified. Validation labels are not opened by this tool.
"""
from pathlib import Path
import collections,time
import numpy as np
import openpyxl
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.contract_curve_geometry import fit_ordered_circle
from weld_pipeline.cbf_weld_label_reader_332_v2 import extract_cbf_weld_json
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';DATA=ROOT.parent/'data/G_Data'
def run_case(group):
 folder=DATA/group['representative_directory'];jsonfiles=list(folder.glob('*weld_welds.json'));plyfiles=list(folder.glob('*weld_welds.ply'));assert len(jsonfiles)==len(plyfiles)==1,'Expected one geometry JSON and one weld PLY';jsonpath=jsonfiles[0];plypath=plyfiles[0];bookpath=next(folder.glob('*.xlsx'));export=read(jsonpath);assert export['unit']=='millimeter' and export['coordinateFrame']=='reference_workpiece';w=openpyxl.load_workbook(bookpath,read_only=True,data_only=False)
 try:
  it=w['焊缝信息'].iter_rows(values_only=True);headers=[str(v or '').split('\n')[-1] for v in next(it)];excel={str(r['ObjectId']):r for vals in it if (r:=dict(zip(headers,vals))).get('ObjectId')}
 finally:w.close()
 assert set(excel)=={str(x['id']) for x in export['welds']}
 with plypath.open() as f:
  header=[]
  for line in f:
   header.append(line.strip())
   if line.strip()=='end_header':break
  assert 'format ascii 1.0' in header and 'comment unit millimeter; frame reference_workpiece' in header;count=int(next(s.split()[-1] for s in header if s.startswith('element vertex')));points=np.loadtxt(f,ndmin=2)[:,:3];assert len(points)==count
 cbfs=list(folder.glob('*.cbf'));aux={};sources=[jsonpath,plypath,bookpath]
 if len(cbfs)==1:
  objects,_=extract_cbf_weld_json(cbfs[0]);aux={str(r.get('ObjectId')):r for r in objects};sources+=cbfs
 allpoints=[];parents=[];audits=[];kinds=collections.Counter()
 for weld in export['welds']:
  pid=str(weld['id']);row=excel[pid];segments=weld['segments'];assert len(segments)==int(row['SegmentCount']);children=[];ends=[];length=0
  kindtext=str(row['WeldType (isVerticalWeldLine)']);assert 'Vertical' in kindtext or 'Horizontal' in kindtext;wt='vertical' if 'Vertical' in kindtext else 'horizontal'
  for i,s in enumerate(segments):
   assert s['index']==i;p=np.asarray(s['points'],float);assert len(p)>=2 and p.shape[1]==3 and np.isfinite(p).all();allpoints.append(p);ends.append(p[[0,-1]]);kind=s['kind'];declared=float(s['lengthMm']);length+=declared
   if kind=='line':
    delta=p[-1]-p[0];ln=np.linalg.norm(delta);assert ln>0 and abs(ln-declared)<1e-4;assert np.linalg.norm(np.cross(p-p[0],delta/ln),axis=1).max()<1e-5;representation=dict(kind='line',endpoints_mm=p[[0,-1]].tolist())
   elif kind=='arc':
    arc=fit_ordered_circle(p,tolerance_mm=1e-5,is_closed=np.linalg.norm(p[0]-p[-1])<1e-6);assert abs(arc.length_mm-declared)<1e-3;representation=dict(kind='arc',arc_parameters=arc.to_parameters())
   else:raise ValueError('Unsupported source curve '+kind)
   children.append(dict(original_child_id=pid+':'+str(i),original_parent_id=pid,original_segment_index=i,original_kind=kind,weld_type=wt,representation=representation,original_attributes={},attributes_status='NOT_YET_RECONSTRUCTED'))
   kinds[wt+'/'+kind]+=1
  assert abs(length-float(row['Length']))<1e-3
  for side,p in [('Start',ends[0][0]),('End',ends[-1][1])]:
   expected=np.array([row[side+'Matrix.Translation.'+a] for a in 'XYZ'],float);assert np.linalg.norm(expected-p)<1e-4
  geometric_closed=bool(np.linalg.norm(ends[0][0]-ends[-1][1])<=1e-6);closed_flag=aux.get(pid,{}).get('IsClosedWeldLine')
  if geometric_closed and closed_flag is None:raise ValueError('Closed parent flag unresolved in authoritative/auxiliary exports')
  closed=bool(closed_flag) if closed_flag is not None else False
  if closed:assert geometric_closed
  gaps=['SHARED' if np.linalg.norm(a[1]-b[0])<=1e-6 else 'POSITIVE_GAP' for a,b in zip(ends[:-1],ends[1:])];parents.append(dict(original_parent_id=pid,original_child_count=len(children),children=children,gaps=gaps,is_closed=closed));audits.append(dict(parent=pid,json_excel_geometry_agrees=True,closed_flag_source='CBF_auxiliary' if closed_flag is not None else 'Open endpoint geometry',closed_endpoint_geometry=geometric_closed))
 samples=np.concatenate(allpoints);d1=float(cKDTree(points).query(samples,workers=2)[0].max());d2=float(cKDTree(samples).query(points,workers=2)[0].max());assert max(d1,d2)<1e-5,(d1,d2)
 target=dict(coordinate_frame='workpiece_local',unit='mm',role='TRAIN',original_parents=len(parents),original_children=sum(len(p['children']) for p in parents),parents=parents,unresolved_children=0);dest=BASE/'labels'/group['key'];dest.mkdir(parents=True,exist_ok=False);write(dest/'targets.json',target);receipt=dict(source_key=group['key'],family_id=group['family_id'],role='EXPANSION_TRAIN',directory=group['representative_directory'],parents=len(parents),children=target['original_children'],categories=dict(kinds),json_to_ply_max_mm=d1,ply_to_json_max_mm=d2,parent_audits=audits,targets_sha256=digest(dest/'targets.json'),source_files={str(p.relative_to(DATA)):digest(p) for p in sources},group_manifest_sha256=digest(BASE/'group_manifest.json'),source_sha256=digest(__file__),model_score=False,attributes_complete=False);write(dest/'receipt.json',receipt);return receipt

def main():
 manifest=read(BASE/'group_manifest.json');result=[];failures=[]
 for group in manifest['groups']:
  if group['role']!='EXPANSION_TRAIN':continue
  dest=BASE/'labels'/group['key']/'receipt.json'
  if dest.exists():result.append(read(dest));continue
  try:r=run_case(group);result.append(r);print('PASS',group['key'],r['directory'],r['children'],r['categories'],flush=True)
  except Exception as exc:
   failure=dict(key=group['key'],directory=group['representative_directory'],type=type(exc).__name__,reason=str(exc));failures.append(failure);print('QUARANTINE',failure,flush=True)
 write(BASE/'training_label_rebuild_summary.json',dict(successes=result,quarantined=failures,validation_labels_opened=False,no_model_fit=True,all_training_cases_resolved=not failures));print('LABEL_REBUILD_COMPLETE',len(result),len(failures),flush=True)
if __name__=='__main__':main()
