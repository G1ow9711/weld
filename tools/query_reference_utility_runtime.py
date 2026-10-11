"""Input-only features and singleton export for an experimental query selector.

Scores mean reference-match utility, never physical weld existence. Parent
assembly and certified finite contact are not supplied by this prototype.
"""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';GRID=np.array([.125,.375,.625,.875])
class Inputs:
 def __init__(self,key):
  self.folder=BASE/'positive_query_runtime_v2'/key;r=read(self.folder/'receipt.json');self.receipt=r
  for n,sha in r['files'].items():assert digest(self.folder/n)==sha
  src=BASE/'frozen_features/legacy_runtime'/('source_'+key);f=src/'run329'/('source_'+key)/'features.npy';assert digest(f)==r['feature_sha256'];self.x=np.load(f,mmap_mode='r')
  binding=next(v for v in read(src/'run285/all_geometry_frozen_before_labels.json')['cases'] if v['source']==key);g=src/'run285'/binding['geometry_file'];assert digest(g)==r['geometry_sha256']
  with np.load(g) as z:self.offsets=z['offsets'];self.spans=z['spans']
  self.canonical=np.load(self.folder/'canonical_fragments.npy');self.pid=np.load(self.folder/'profile_ids.npy');self.length=self.spans[self.offsets[self.pid+1]-1,1];self.ratio=np.load(self.folder/'interval_ratios.npy').reshape(-1,2);self.role=np.load(self.folder/'weld_type.npy').ravel();self.valid=np.load(self.folder/'valid.npy').ravel();self.profiles=read(src/'run185'/('source_'+key)/'profiles.json')
 def features(self,ids):
  ids=np.asarray(ids);f=ids//4;qi=ids%4;xx=np.asarray(self.x[np.column_stack((2*f,2*f+1))],np.float64);span=self.spans[self.canonical[f],:2];length=self.length[f];q=(span[:,0]+GRID[qi]*(span[:,1]-span[:,0]))/length;bounds=self.ratio[ids];extra=np.column_stack((q,bounds,self.role[ids],np.log1p(length),np.log1p((bounds[:,1]-bounds[:,0])*length)))
  out=np.column_stack((xx.mean(1),xx.min(1),xx.max(1),extra)).astype(np.float32);assert out.shape==(len(ids),462) and np.isfinite(out).all();return out
 def decode(self,ids,scores):
  parents=[];seen=set()
  for i,score in zip(ids,scores):
   i=int(i);pid=int(self.pid[i//4]);signature=(pid,self.ratio[i].tobytes(),int(self.role[i]))
   if signature in seen:continue
   seen.add(signature);curve=CurveCarrier(self.profiles[pid]['representation']);lo,hi=self.ratio[i]*curve.length;rep=curve.segment(float(lo),float(hi));s=line_segment(*rep['endpoints_mm']) if curve.kind=='line' else sample_arc(CircularArc.from_parameters(rep['arc_parameters']));s.update(weld_type='horizontal' if self.role[i]==0 else 'vertical',predicted_side='UNKNOWN');parents.append(dict(child_count=1,is_closed=bool(s.get('is_closed',False)),gaps=[],segments=[s],query_id=i,reference_match_score=float(score)))
  return dict(parents=parents,invalid_predictions=[],total_prediction_children=len(parents),GT_reads=0,STEP_reads=0,prototype_only=True,parent_assembly_implemented=False,finite_material_contact_certified=False)
