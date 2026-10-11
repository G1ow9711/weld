"""Locate new TRAIN failures before learned screening; never creates candidates."""
import argparse,collections,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def run(key):
 start=time.perf_counter();entry=next(c for c in read(BASE/'coverage_cohort_protocol.json')['cases'] if c['key']==key);target=BASE/'labels'/key/'targets.json';assert digest(target.parent/'receipt.json')==entry['labels_receipt_sha256'];truth=read(target)
 folder=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);binding=read(folder/'input_freeze.json');profiles=read(folder/'profiles.json');assert digest(folder/'profiles.json')==binding['files']['profiles.json']
 line_ids=np.array([i for i,p in enumerate(profiles) if p['representation']['kind']=='line']);lines=np.asarray([profiles[i]['representation']['endpoints_mm'] for i in line_ids]);origin=lines[:,0];delta=lines[:,1]-origin;length2=np.sum(delta*delta,axis=1);assert np.all(length2>0)
 arcs=[(i,CurveCarrier(p['representation'])) for i,p in enumerate(profiles) if p['representation']['kind']=='arc']
 xyzpath=BASE/'inputs'/key/'xyz_mm.npy';assert digest(xyzpath)==binding['input_sha256'];xyz=np.load(xyzpath,mmap_mode='r');tree=cKDTree(xyz);rows=[]
 for pi,p in enumerate(truth['parents']):
  for ci,c in enumerate(p['children']):
   curve=CurveCarrier(c['representation']);samples=curve.at(np.linspace(0,curve.length,17));cloud_dist=tree.query(samples,workers=2)[0]
   if curve.kind=='line':
    distances=[]
    for q in samples[[0,-1]]:
     rel=q-origin;u=np.clip(np.sum(rel*delta,axis=1)/length2,0,1);distances.append(np.linalg.norm(rel-u[:,None]*delta,axis=1))
    errors=np.maximum(*distances);order=np.argsort(errors)[:3];near=[dict(profile=int(line_ids[i]),max_distance_mm=float(errors[i])) for i in order];count=int(sum(errors<=3.));claim='Exact whole-target-line inclusion in 3mm tube around finite source line, by convex endpoint distances; no parent/type/contact validity claim.'
   else:
    values=[]
    for i,carrier in arcs:
     q,_=carrier.project(samples);distance=np.linalg.norm(samples-carrier.at(np.clip(q,0,carrier.length)),axis=1);values.append((float(distance.max()),i))
    values.sort();near=[dict(profile=i,max_distance_mm=d) for d,i in values[:3]];count=sum(d<=3. for d,i in values);claim='17 analytic samples only; not continuous arc/contact coverage proof.'
   rows.append(dict(parent=pi,child=ci,kind=curve.kind,weld_type=c['weld_type'],same_kind_carriers_within_3mm=count,nearest=near,workpiece_sample_nearest_max_mm=float(cloud_dist.max()),coverage_scope=claim))
 report=dict(key=key,rows=rows,children=len(rows),children_with_same_kind_carrier=sum(r['same_kind_carriers_within_3mm']>0 for r in rows),breakdown={kind:dict(G=sum(r['kind']==kind for r in rows),covered=sum(r['kind']==kind and r['same_kind_carriers_within_3mm']>0 for r in rows)) for kind in ('line','arc')},target_sha256=digest(target),profiles_sha256=digest(folder/'profiles.json'),input_xyz_sha256=digest(xyzpath),diagnostic_only=True,GT_after_candidate_freeze=True,no_GT_geometry_injected=True,fit_calls=0,DEV_reads=0,seconds=time.perf_counter()-start)
 out=BASE/'native_capacity'/('source_'+key)/'raw_carrier_coverage.json';assert not out.exists();write(out,report);print({k:v for k,v in report.items() if k!='rows'},flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);a=ap.parse_args()
 with threadpool_limits(limits=2):run(a.key)
