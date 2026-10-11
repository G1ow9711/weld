"""Deterministic equal-budget sampling over known TRAIN parent candidates.

Identity is only used to deduplicate within one family, never as a model input.
Exact geometry and contact references are retained; no rounding or averaging.
"""
import hashlib,json
from collections import deque
FIELDS=('kind','weld_type','points','length_mm','profile','predicted_side','material_identity','is_closed')
def exact_segment_record(s,reverse=False):
 record={k:s[k] for k in FIELDS if k in s}
 record['points']=s['points'][::-1] if reverse else s['points']
 if s['kind']=='arc':
  a=s['arc_parameters'];record['arc_geometry']={k:a[k] for k in ('center_mm','radius_mm','plane_normal','is_closed')}
  record['arc_geometry']['absolute_sweep_rad']=abs(a['sweep_rad'])
 return record
def canonical_parent_signature(parent):
 variants=[];n=len(parent['segments'])
 for reverse in (False,True):
  segments=parent['segments'][::-1] if reverse else parent['segments']
  gaps=parent['gaps'][::-1] if reverse else parent['gaps'];parts=[]
  for i,s in enumerate(segments):
   edge=gaps[i] if i<n-1 else ('SHARED' if parent['is_closed'] else None)
   parts.append((exact_segment_record(s,reverse),edge))
  for shift in (range(n) if parent['is_closed'] else [0]):
   variants.append(json.dumps((bool(parent['is_closed']),parts[shift:]+parts[:shift]),sort_keys=True,separators=(',',':'),allow_nan=False))
 return hashlib.sha256(min(variants).encode()).hexdigest()
def eligible_rows(parents,rows,family,seed=1031):
 assert len(parents)==len(rows);seen={};unique=[];duplicates=[]
 for i,(p,r) in enumerate(zip(parents,rows)):
  if not r['parent_loss_valid']:continue
  signature=canonical_parent_signature(p);outcome=(r['children'],r['D3_TP'],r['D3_FP'])
  if signature in seen:
   first,prior=seen[signature];assert prior==outcome,'Exact parent identity disagrees in supervision';duplicates.append((i,first));continue
  seen[signature]=(i,outcome);order=hashlib.sha256(f'{seed}|{family}|{signature}'.encode()).hexdigest();unique.append(dict(row=i,signature=signature,order=order))
 return sorted(unique,key=lambda r:r['order']),duplicates

def round_robin(eligible,budget):
 assert 0<=budget<=sum(len(v) for v in eligible.values());queues={k:deque(v) for k,v in sorted(eligible.items())};chosen=[]
 while len(chosen)<budget:
  for k,q in queues.items():
   if q:chosen.append(dict(family=k,**q.popleft()))
   if len(chosen)==budget:break
 return chosen
