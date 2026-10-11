"""Label-free bounded raw-PCD moments and samples for duplicate screening.

Moments only retrieve possible duplicates; they cannot certify independence or
handle cropped variants on their own. No train/validation assignment is made.
"""
from pathlib import Path
import time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'data/G_Data';BASE=ROOT/'migration_checks';DEST=BASE/'input_geometry_fingerprints_v1'
def run():
 DEST.mkdir(exist_ok=True);audit=read(BASE/'dataset_audit.json');inventory={r['directory']:r for r in read(BASE/'data_inventory.json')['cases']};seen=set();reports=[]
 for row in audit['cases']:
  sha=row['pcd_sha256']
  if sha in seen:continue
  seen.add(sha);out=DEST/(sha+'.json');samplefile=DEST/(sha+'.npz')
  if out.exists():
   prior=read(out);assert prior['pcd_sha256']==sha and digest(samplefile)==prior['sample_sha256'];reports.append(prior);continue
  start=time.perf_counter();path=DATA/inventory[row['directory']]['pointcloud'];assert path.stat().st_size==row['pcd_bytes'] and digest(path)==sha
  with path.open('rb') as stream:
   header=[]
   while True:
    line=stream.readline().decode('ascii').strip();header.append(line)
    if line.startswith('DATA '):break
    if len(header)>40:raise ValueError('Invalid PCD header')
   assert header==row['pcd_header'] and header[-1]=='DATA ascii'
   fields=next(x.split()[1:] for x in header if x.startswith('FIELDS '));assert fields==['x','y','z'];n=int(next(x.split()[1] for x in header if x.startswith('POINTS ')));ids=np.unique(np.linspace(0,n-1,min(8192,n),dtype=np.int64));samples=[];count=0;mean=np.zeros(3);m2=np.zeros((3,3));bounds=np.array([[np.inf]*3,[-np.inf]*3])
   while count<n:
    block=np.loadtxt(stream,dtype=np.float64,max_rows=min(131072,n-count),ndmin=2);assert block.shape[1]==3 and len(block)>0 and np.isfinite(block).all();start_count=count;nb=len(block);bm=block.mean(0);bc=block-bm;delta=bm-mean;new=count+nb;m2+=bc.T@bc+np.outer(delta,delta)*(count*nb/new);mean+=delta*(nb/new);count=new;bounds[0]=np.minimum(bounds[0],block.min(0));bounds[1]=np.maximum(bounds[1],block.max(0));wanted=ids[(ids>=start_count)&(ids<count)]-start_count;samples.append(block[wanted])
   assert count==n and not stream.read().strip()
  sample=np.concatenate(samples);assert len(sample)==len(ids);cov=m2/count;values,vectors=np.linalg.eigh(cov);radius=np.linalg.norm(sample-mean,axis=1);np.savez_compressed(samplefile,points=sample,indices=ids,covariance=cov,center=mean,principal_axes=vectors)
  report=dict(directory=row['directory'],pcd_path=str(path.relative_to(DATA)),pcd_sha256=sha,points=n,sample_points=len(sample),centroid_raw_units=mean.tolist(),covariance_eigenvalues_raw_units_squared=values.tolist(),sample_radius_quantiles_raw_units=np.quantile(radius,np.linspace(0,1,33)).tolist(),axis_aligned_bbox_raw_units=bounds.tolist(),sample_sha256=digest(samplefile),GT_reads=0,STEP_reads=0,unit_conversion_not_assumed=True,no_split_assignment=True,geometric_duplicate_proof=False,source_sha256=digest(__file__),seconds=time.perf_counter()-start);write(out,report);reports.append(report);print(len(reports),row['directory'],n,round(report['seconds'],2),flush=True)
 write(DEST/'summary.json',dict(inputs=reports,count=len(reports),source_audit_sha256=digest(BASE/'dataset_audit.json'),no_split_assignment=True,not_an_independence_certificate=True));print('FINGERPRINTS_COMPLETE',len(reports),flush=True)
if __name__=='__main__':run()
