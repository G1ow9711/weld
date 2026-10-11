"""Geometry-only native XYZ preparation; no label or STEP reads."""
import hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from weld_pipeline.sjg_local_input_897 import read_workpiece_local_surface
DATA=ROOT.parent/'data/G_Data';DEST=ROOT/'server_rebuild/inputs_v1'
CASES={'21':'9月20日/GR768A','BM79A_native_local':'9月20日/BM79A','10':'9月20日/224-GR24A1','25':'标注数据20261004/真值/1-161-DK2B-0928','36':'标注数据20261004/真值/2-161-FR62C-0928'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
for key,d in CASES.items():
 start=time.perf_counter();source=next((DATA/d).glob('*_weld_workpiece.ply'))
 xyz,meta=read_workpiece_local_surface(source);out=DEST/key;out.mkdir(parents=True,exist_ok=True)
 np.save(out/'xyz_mm.npy',xyz)
 receipt=dict(meta,source_file=str(source.relative_to(DATA)),source_sha256=sha(source),xyz_sha256=sha(out/'xyz_mm.npy'),bbox_mm=[xyz.min(0).tolist(),xyz.max(0).tolist()],seconds=time.perf_counter()-start,GT_reads=0,STEP_reads=0,all_source_xyz_preserved=True,inference_speed_acceptance=False)
 (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));print(key,len(xyz),receipt['seconds'],flush=True)
