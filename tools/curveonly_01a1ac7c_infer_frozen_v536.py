"""Frozen complete-curve geometry inference: raw mm XYZ, optional material STEP."""
import argparse,csv,hashlib,importlib,json,math,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'experiment_records/CURVEONLY_01a1ac7c_20261106_v1/final_frozen_20261108/MODEL_SPEC.json'
RUNTIME='weld_pipeline.curveonly_01a1ac7c_native_boundary_runtime_v528'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def bound(b):
 p=Path(b['path']);p=p if p.is_absolute() else ROOT/p
 if p.stat().st_size!=b['bytes'] or sha(p)!=b['sha256']:raise ValueError('Frozen file changed: '+str(p))
 return p

def verify(path):
 s=json.loads(Path(path).read_text())
 if s['schema']!='CURVEONLY_FROZEN_COMPLETE_GEOMETRY_V1' or s['runtime_module']!=RUNTIME:raise ValueError('Unexpected frozen model')
 if s['input_contract']!='Pure mm XYZ PLY/NPY; optional material-only STEP and independently qualified STEP-to-XYZ pose':raise ValueError('Unexpected inputs')
 bound(s['weights'])
 for b in s['runtime_sources']:bound(b)
 return s

def save(path,value):
 with Path(path).open('w',encoding='utf8') as f:
  json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def export_complete(out,workpiece_id,decoded):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);save(out/'curves.json',decoded)
 fields=['workpiece_id','weld_id','weld_type','weld_side','is_closed','start_x','start_y','start_z','end_x','end_y','end_z','length_mm','confidence','curve_points_file','geometry_locator','status']
 with (out/'complete_curves.csv').open('w',encoding='utf8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for i,p in enumerate(decoded['parents']):
   a,b=p['segments'][0]['points'][0],p['segments'][-1]['points'][-1]
   writer.writerow(dict(workpiece_id=workpiece_id,weld_id=p['parent_id'],weld_type=p.get('weld_type','UNKNOWN'),weld_side=p.get('weld_side') or 'UNKNOWN',is_closed=p['is_closed'],start_x=a[0],start_y=a[1],start_z=a[2],end_x=b[0],end_y=b[1],end_z=b[2],length_mm=math.fsum(float(s['length_mm']) for s in p['segments']),confidence=p['confidence'],curve_points_file='curves.json',geometry_locator=f'parents[{i}].segments',status='COMPLETE_CURVE_GEOMETRY'))
  f.flush();os.fsync(f.fileno())

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--spec',type=Path,default=DEFAULT);ap.add_argument('--verify-only',action='store_true');ap.add_argument('--xyz',type=Path);ap.add_argument('--step',type=Path);ap.add_argument('--pose-config',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--workpiece-id',default='workpiece');a=ap.parse_args();s=verify(a.spec)
 if a.verify_only:
  print(json.dumps(dict(status='FROZEN_FILES_VERIFIED',runtime=RUNTIME,weights_sha256=s['weights']['sha256'],sources=len(s['runtime_sources']))));return
 if a.xyz is None or a.out is None:ap.error('--xyz and --out are required for inference')
 for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
 from tools.curveonly_01a1ac7c_experiment import fence
 from threadpoolctl import threadpool_limits
 sys.addaudithook(fence);runtime=importlib.import_module(RUNTIME)
 with bound(s['weights']).open('rb') as f:model=pickle.load(f)
 pose=None
 if a.pose_config:
  config=json.loads(a.pose_config.read_text())
  if set(config)-{'transform_STEP_to_XYZ','units','coordinate_frame','multiplication'} or config.get('units')!='mm':raise ValueError('Pose config may contain only agreed mm coordinates')
  pose=config['transform_STEP_to_XYZ']
  if len(pose)!=4 or any(len(row)!=4 or not all(math.isfinite(float(v)) for v in row) for row in pose):raise ValueError('Pose must be a finite4x4 STEP-to-XYZ transform')
 raw=dict(path=str(a.xyz.resolve()),sha256=sha(a.xyz));step=dict(path=str(a.step.resolve()),sha256=sha(a.step)) if a.step else None;start=time.perf_counter()
 with threadpool_limits(limits=2):decoded=runtime.extract(dict(raw=raw,STEP=step,transform=pose),model)
 export_complete(a.out,a.workpiece_id,decoded)
 save(a.out/'run.json',dict(status='SUCCESS' if decoded['parents'] else 'SUCCESS_NO_CURVES',raw=raw,STEP=step,weights=s['weights'],runtime=RUNTIME,model_spec_sha256=sha(a.spec),complete_curve_parents=len(decoded['parents']),segmentation_is_not_evaluated=True,seconds=time.perf_counter()-start,formal_performance_or_independent_test_claim=False))
 print(json.dumps(dict(status='SUCCESS' if decoded['parents'] else 'SUCCESS_NO_CURVES',complete_curve_parents=len(decoded['parents']),out=str(a.out.resolve()))))

if __name__=='__main__':main()
