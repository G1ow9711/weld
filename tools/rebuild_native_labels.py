"""Offline reconstruction of the frozen five TRAIN pieces from original exports.

Excel supplies parent identities/counts, PLY verifies geometry; CBF only recovers
ordered straight child boundaries. Labels are never runtime point inputs.
"""
import hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import openpyxl
from scipy.spatial import cKDTree
from weld_pipeline.cbf_weld_label_reader_332_v2 import extract_cbf_weld_json,JsonReferenceIndex
from weld_pipeline.contract_curve_geometry import fit_ordered_circle
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.world_set_evaluation_285 import reference_segment
DATA=ROOT.parent/'data/G_Data';DEST=ROOT/'server_rebuild/labels_v1'
CASES={'21':('9月20日/GR768A',1,1),'BM79A_native_local':('9月20日/BM79A',1,4),'10':('9月20日/224-GR24A1',124,208),'25':('标注数据20261004/真值/1-161-DK2B-0928',16,24),'36':('标注数据20261004/真值/2-161-FR62C-0928',20,32)}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def ply(p):
 with p.open() as f:
  header=[]
  for line in f:
   header.append(line.strip())
   if line.strip()=='end_header':break
  assert 'comment unit millimeter; frame reference_workpiece' in header
  n=int(next(s.split()[-1] for s in header if s.startswith('element vertex')))
  v=np.loadtxt(f,ndmin=2);assert len(v)==n
 return v[:,:3]
def excel(p):
 w=openpyxl.load_workbook(p,read_only=True,data_only=False);it=w['焊缝信息'].iter_rows(values_only=True)
 keys=[str(x or '').split('\n')[-1] for x in next(it)];r=[dict(zip(keys,row)) for row in it];w.close()
 return [x for x in r if x.get('ObjectId')]
def endpoint(row,side):return np.array([row[side+'Matrix.Translation.'+a] for a in 'XYZ'],float)
def child(pid,i,kind,rep,vertical):
 return dict(original_child_id=str(pid)+':'+str(i),original_parent_id=pid,original_segment_index=i,original_kind=kind,weld_type='vertical' if vertical else 'horizontal',representation=rep,original_attributes={},attributes_status='UNCONFIRMED_CHILD_ATTRIBUTES_NOT_FABRICATED')
DEST.mkdir(parents=True,exist_ok=True);receipts=[]
for key,(path,P,G) in CASES.items():
 d=DATA/path;book=next(d.glob('*.xlsx'));label=next(d.glob('*_weld_welds.ply'));surface=next(d.glob('*_weld_workpiece.ply'))
 rows=excel(book);assert len(rows)==P and sum(int(x['SegmentCount']) for x in rows)==G
 points=ply(label);tree=cKDTree(points);parents=[];audit=[];sources=[book,label,surface]
 if key=='BM79A_native_local':
  row=rows[0];pid=str(row['ObjectId']);steps=np.linalg.norm(np.diff(points,axis=0),axis=1)
  blocks=np.split(points,np.flatnonzero(steps<1e-7)+1);assert len(blocks)==G
  assert np.linalg.norm(points[0]-endpoint(row,'Start'))<1e-7 and np.linalg.norm(points[-1]-endpoint(row,'End'))<1e-7
  children=[];length=0
  for i,block in enumerate(blocks):
   v=block[-1]-block[0];l=np.linalg.norm(v);e=np.linalg.norm(np.cross(block-block[0],v/l),axis=1).max()
   if e<=1e-6:kind='line';rep=dict(kind=kind,endpoints_mm=block[[0,-1]].tolist());length+=l
   else:
    arc=fit_ordered_circle(block,tolerance_mm=1e-6,is_closed=False);kind='arc';rep=dict(kind=kind,arc_parameters=arc.to_parameters());length+=arc.length_mm
   children.append(child(pid,i,kind,rep,False))
  assert abs(length-float(row['Length']))<1e-6
  parents=[dict(original_parent_id=pid,original_child_count=G,children=children,gaps=['SHARED']*(G-1),is_closed=False)]
 else:
  cbf=next(d.glob('*.cbf'));sources.append(cbf);objects,meta=extract_cbf_weld_json(cbf);index=JsonReferenceIndex(objects)
  byid={str(x['ObjectId']):x for x in objects};assert set(byid)=={str(x['ObjectId']) for x in rows}
  for row in rows:
   pid=str(row['ObjectId']);raw=byid[pid];parts=index.line_children(raw);assert len(parts)==int(row['SegmentCount'])
   endpoints=np.array([parts[0]['points_reference_mm'][0],parts[-1]['points_reference_mm'][1]])
   assert np.max(np.linalg.norm(endpoints-np.array([endpoint(row,'Start'),endpoint(row,'End')]),axis=1))<1e-5
   assert abs(sum(x['length_mm'] for x in parts)-float(row['Length']))<1e-4
   children=[];gap=[]
   for i,c in enumerate(parts):
    a,b=np.array(c['points_reference_mm']);sample=np.linspace(a,b,int(np.ceil(c['length_mm']))+1)
    error=float(tree.query(sample,workers=2)[0].max());assert error<=0.501,(key,pid,i,error)
    children.append(child(pid,i,'line',dict(kind='line',endpoints_mm=[a.tolist(),b.tolist()]),c['original_is_vertical']))
    audit.append(dict(parent=pid,child=i,reference_to_label_ply_max_mm=error))
    if i:gap.append('SHARED' if np.linalg.norm(a-np.array(parts[i-1]['points_reference_mm'][1]))<=1e-6 else 'POSITIVE_GAP')
   parents.append(dict(original_parent_id=pid,original_child_count=len(children),children=children,gaps=gap,is_closed=bool(raw['IsClosedWeldLine'])))
 targets=dict(coordinate_frame='workpiece_local',unit='mm',role='TRAIN',original_parents=P,original_children=G,parents=parents,unresolved_children=0)
 decoded=dict(parents=[dict(child_count=len(p['children']),is_closed=p['is_closed'],gaps=p['gaps'],segments=[dict(reference_segment(dict(c,world_representation=c['representation'])),weld_type=c['weld_type']) for c in p['children']]) for p in parents],invalid_predictions=[],total_prediction_children=G)
 score=score_native_parents(decoded,targets);assert score['categories']['overall']['TP']==G
 out=DEST/key;out.mkdir(exist_ok=True);(out/'targets.json').write_text(json.dumps(targets,ensure_ascii=False,indent=2))
 receipt=dict(source_key=key,role='TRAIN',parents=P,children=G,files={str(f.relative_to(DATA)):sha(f) for f in sources},targets_sha256=sha(out/'targets.json'),ply_checks=audit,label_roundtrip_TP=G,model_test=False,fit_calls=0,full_contract_pass=False)
 (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));receipts.append(receipt)
 print('LABEL_REBUILT',key,P,G,flush=True)
(DEST/'receipt.json').write_text(json.dumps(dict(status='FIVE_TRAIN_NATIVE_LABELS_REBUILT',children=sum(x['children'] for x in receipts),parents=sum(x['parents'] for x in receipts),sources=receipts),ensure_ascii=False,indent=2))
