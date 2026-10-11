"""Offline TRAIN check: can original point queries support endpoint detection?"""
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';DEST=BASE/'raw_point_endpoint_support_v1'
def main():
 DEST.mkdir(exist_ok=False);reports=[]
 for case in read(BASE/'coverage_cohort_protocol.json')['cases']:
  start=time.perf_counter();key=case['key'];path=BASE/'inputs'/key/'xyz_mm.npy';ir=read(path.parent/'receipt.json');assert digest(path)==ir['xyz_sha256'];xyz=np.load(path,mmap_mode='r');tree=cKDTree(xyz);targetpath=BASE/'labels'/key/'targets.json';target=read(targetpath);points=[];rows=[]
  for pi,p in enumerate(target['parents']):
   for ci,c in enumerate(p['children']):
    curve=CurveCarrier(c['representation']);ends=curve.at(np.array([0.,curve.length]));points.extend(ends);rows.append(dict(parent=pi,child=ci,weld_type=c['weld_type'],kind=c['original_kind']))
  dist,ids=tree.query(np.asarray(points),workers=2);dist=dist.reshape(-1,2);ids=ids.reshape(-1,2)
  for i,row in enumerate(rows):row.update(nearest_distances_mm=dist[i].tolist(),nearest_original_point_ids=ids[i].tolist())
  result=dict(key=key,children=len(rows),both_endpoints_within_3mm=int(np.all(dist<=3.,axis=1).sum()),both_endpoints_within_1mm=int(np.all(dist<=1.,axis=1).sum()),endpoint_distance_quantiles_mm=np.quantile(dist,[0,.5,.9,.99,1]).tolist(),rows=rows,input_sha256=digest(path),target_sha256=digest(targetpath),scope='Reference endpoint proximity to raw XYZ queries only; does not prove a raw-point pair passes original parent D3 or weld/contact existence',fit_calls=0,DEV_reads=0,seconds=time.perf_counter()-start);write(DEST/(key+'.json'),result);reports.append({k:v for k,v in result.items() if k!='rows'});print(reports[-1],flush=True);del tree;xyz._mmap.close()
 write(DEST/'summary.json',dict(cases=reports,fit_calls=0,DEV_reads=0))
if __name__=='__main__':main()
