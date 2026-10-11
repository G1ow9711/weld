"""Zero-fit bounded parent hypotheses from real unscreened native children.

No targets are read. Native parents and arcs are retained. This first capacity
experiment adds shared-point line connections; new gap evidence is not invented.
"""
import argparse,collections,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_child_parent_supervision_1255 import singleton,state_geometry,link_features
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment

ROOT=Path(__file__).resolve().parents[1];INPUT=ROOT/'server_rebuild/child_residual_v1';BASE=ROOT/'server_rebuild/best_baseline_restore';DEST=ROOT/'server_rebuild/parent_hypotheses_v1'
CONFIG=dict(new_outgoing_per_port_kind=8,port_budget_scope='Initial proposal per directed port; reverse completion and native edges retained additionally',beam_per_component=32,new_parents_per_component=32,shared_endpoint_gate_mm=6.00001,new_connections='SHARED between line children of same weld type with shared or unknown face evidence',new_gap_connections=False,GT_reads=0,STEP_reads=0,fit_calls=0,native_parents_retained=True,requires_kept_parent_anchor=False,ranker='Frozen446 link log ratio mean + native node probability logit mean; beam heuristic only')
def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def run(key):
 started=time.perf_counter();dest=DEST/('source_'+key);dest.mkdir(parents=True,exist_ok=False);write(dest/'registered_protocol.json',CONFIG)
 src=INPUT/('source_'+key);receipt=read(src/'receipt.json')
 for name,sha in receipt['files'].items():assert digest(src/name)==sha
 original=read(src/'parents.json')['parents'];metadata=read(src/'metadata.json');origins=read(src/'origins.json')
 atoms=[];amd=[];aliases=[];lookup={};row_state=[];keys=[]
 for row,(p,m) in enumerate(zip(original,metadata)):
  s=p['segments'][0];forward=np.asarray(s['points']);backward=forward[::-1]
  rev=tuple(backward.ravel())<tuple(forward.ravel());canonical=dict(s,**reverse_segment(s)) if rev else s
  geometry=sorted([s['endpoints_mm'],s['endpoints_mm'][::-1]])[0] if s['kind']=='line' else min(forward.tolist(),backward.tolist())
  # No rounding and no coordinate averaging. Different contact references remain distinct.
  identity=dict(kind=s['kind'],weld_type=s['weld_type'],closed=p['is_closed'],geometry=geometry,faces=sorted(set(tuple(v) for v in m['face_pairs'])),side=s.get('predicted_side'),material=s.get('material_identity'))
  h=signature(identity)
  if h not in lookup:
   lookup[h]=len(atoms);atoms.append(singleton(canonical));amd.append(dict(m));aliases.append([]);keys.append(h)
  i=lookup[h];aliases[i].append(row);row_state.append(2*i+int(rev))
  amd[i]['probability_mean']=max(amd[i]['probability_mean'],m['probability_mean'])
 state=state_geometry(ParentRelations(atoms,amd));count=len(atoms);ends=state['ends'];tree=cKDTree(ends[:,0]);native=read(BASE/'legacy_runtime'/('source_'+key)/'run379'/('test_'+key)/'decoded.json')
 owner={(v['parent'],v['child']):row_state[i] for i,v in enumerate(origins)}
 native_paths=[];native_edges=set()
 for pi,p in enumerate(native['parents']):
  chain=tuple(owner[pi,ci] for ci in range(p['child_count']));kinds=tuple(1 if g=='SHARED' else 2 for g in p['gaps']);native_paths.append((chain,kinds,p['is_closed']))
  for a,b,kind in zip(chain,chain[1:],kinds):native_edges.add((a,b,kind));native_edges.add((b^1,a^1,kind))
  if p['is_closed'] and len(chain)>1:native_edges.add((chain[-1],chain[0],1));native_edges.add((chain[0]^1,chain[-1]^1,1))
 candidate=set();discarded=[]
 for a in range(2*count):
  if state['closed'][a] or not state['line'][a]:continue
  eligible=[]
  for b in tree.query_ball_point(ends[a,1],CONFIG['shared_endpoint_gate_mm']):
   if a//2==b//2 or state['closed'][b] or not state['line'][b] or state['horizontal'][a]!=state['horizontal'][b]:continue
   fa,fb=state['faces'][a],state['faces'][b]
   if fa and fb and not fa.intersection(fb):continue
   distance=float(np.linalg.norm(ends[a,1]-ends[b,0]));same_pair=bool(state['face_pairs'][a]&state['face_pairs'][b])
   eligible.append((not same_pair,distance,keys[b//2],b))
  eligible.sort()
  for _,distance,_,b in eligible[:CONFIG['new_outgoing_per_port_kind']]:candidate.add((a,b,1))
  discarded.extend(dict(a=a,b=b,kind=1,reason='outgoing_port_budget',distance_mm=d) for _,d,_,b in eligible[CONFIG['new_outgoing_per_port_kind']:])
 candidate.update(native_edges)
 # Preserve reverse representations even when the opposite port was truncated.
 for a,b,kind in list(candidate):candidate.add((b^1,a^1,kind))
 pairs=list(sorted(candidate));mp=ROOT/'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776/run446/models.pkl'
 assert digest(mp)==read(mp.parent/'training_receipt.json')['model_sha256']
 with mp.open('rb') as f:model=pickle.load(f)['models']['link']
 probability=np.zeros((len(pairs),3))
 if pairs:probability[:,model.classes_.astype(int)]=model.predict_proba(link_features(state,[(a,b) for a,b,k in pairs]))
 weights={edge:float(np.log(max(probability[j,edge[2]],1e-6))-np.log(max(probability[j,0],1e-6))) for j,edge in enumerate(pairs)}
 adjacency=collections.defaultdict(list);dsu=list(range(count))
 def find(a):
  while dsu[a]!=a:dsu[a]=dsu[dsu[a]];a=dsu[a]
  return a
 for a,b,kind in pairs:
  adjacency[a].append((b,kind));ra,rb=find(a//2),find(b//2)
  if ra!=rb:dsu[rb]=ra
 components=collections.defaultdict(list)
 for i in range(count):components[find(i)].append(i)
 conflict_cache={}
 def conflict(a,b):
  if a==b:return True
  pair=tuple(sorted((a,b)))
  if pair in conflict_cache:return conflict_cache[pair]
  sa,sb=atoms[a]['segments'][0],atoms[b]['segments'][0];bad=False
  if sa['kind']==sb['kind']=='line' and state['face_pairs'][2*a]&state['face_pairs'][2*b]:
   pa,pb=np.asarray(sa['points'])[[0,-1]],np.asarray(sb['points'])[[0,-1]];v=pa[1]-pa[0];length=np.linalg.norm(v);v=v/length;q=(pb-pa[0])@v;perp=np.linalg.norm(pb-pa[0]-q[:,None]*v,axis=1)
   bad=bool(perp.max()<1e-6 and min(length,q.max())-max(0.,q.min())>1e-6)
  conflict_cache[pair]=bad;return bad
 def canonical(chain,kinds):
  reverse=tuple(i^1 for i in chain[::-1]);rk=kinds[::-1]
  a=tuple((keys[i//2],i%2) for i in chain);b=tuple((keys[i//2],i%2) for i in reverse)
  return (chain,kinds) if a<=b else (reverse,rk)
 def rank(path):
  chain,kinds=path;score=np.mean([weights[a,b,k] for a,b,k in zip(chain,chain[1:],kinds)]) if len(chain)>1 else 0.
  p=np.clip([amd[i//2]['probability_mean'] for i in chain],1e-6,1-1e-6);score+=float(np.mean(np.log(p)-np.log1p(-p)))
  return (-score,tuple((keys[i//2],i%2) for i in chain),kinds)
 native_set={canonical(c,k) for c,k,closed in native_paths};new=[];component_reports=[]
 for component,members in components.items():
  member_set=set(members);seeds={(tuple([2*i+o]),tuple()) for i in members for o in (0,1)}
  for c,k,closed in native_paths:
   if c and c[0]//2 in member_set and not closed:seeds.add((c,k));seeds.add((tuple(i^1 for i in c[::-1]),k[::-1]))
  max_children=max([len(c) for c,k in seeds]+[1])+1;frontier=seeds;retained={};seen=set();dropped=0
  for depth in range(max_children):
   proposed={}
   for chain,kinds in frontier:
    if len(chain)>=max_children:continue
    for b,kind in adjacency.get(chain[-1],[]):
     if any(conflict(b//2,a//2) for a in chain):continue
     c=chain+(b,);k=kinds+(kind,);path=canonical(c,k)
     if path in native_set or path in seen:continue
     proposed[path]=rank(path)
   if not proposed:break
   ordered=sorted(proposed,key=proposed.get);keep=ordered[:CONFIG['beam_per_component']];dropped+=max(0,len(ordered)-len(keep));seen.update(proposed)
   for path in keep:retained[path]=proposed[path]
   frontier=set(keep)|{(tuple(i^1 for i in c[::-1]),k[::-1]) for c,k in keep}
  chosen=sorted(retained,key=retained.get)[:CONFIG['new_parents_per_component']];new.extend(chosen)
  component_reports.append(dict(component=component,atoms=len(members),native_seeds=len(seeds),max_children=max_children,new_parents=len(chosen),beam_discarded=dropped,parent_budget_discarded=max(0,len(retained)-len(chosen))))
 additions=[]
 for chain,kinds in new:additions.append(dict(child_count=len(chain),is_closed=False,gaps=['SHARED' if k==1 else 'POSITIVE_GAP' for k in kinds],segments=[state['segments'][i] for i in chain]))
 decoded=dict(native);decoded['parents']=native['parents']+additions;decoded['total_prediction_children']=native['total_prediction_children']+sum(p['child_count'] for p in additions)
 write(dest/'hypotheses.json',decoded);write(dest/'atoms.json',dict(parents=atoms,metadata=amd,aliases=aliases,row_states=row_state,signatures=keys));write(dest/'paths.json',dict(native=[dict(states=c,kinds=k,closed=closed) for c,k,closed in native_paths],added=[dict(states=c,kinds=k) for c,k in new]));write(dest/'truncation.json',dict(port_discarded=discarded,components=component_reports));write(dest/'edges.json',dict(edges=[dict(a=a,b=b,kind=k,frozen_margin=weights[a,b,k],native=(a,b,k) in native_edges) for a,b,k in pairs]))
 report=dict(source=key,original_children=len(original),atoms=count,native_parents=len(native['parents']),new_parents=len(additions),hypothesis_children=decoded['total_prediction_children'],components=component_reports,GT_reads=0,STEP_reads=0,DEV_reads=0,fit_calls=0,requires_kept_parent_anchor=False,not_an_extraction_result=True,all_native_atom_rows_retained=len(row_state)==len(original),model_sha256=digest(mp),input_receipt_sha256=digest(src/'receipt.json'),source_sha256=digest(__file__),files={name:digest(dest/name) for name in ['hypotheses.json','atoms.json','paths.json','truncation.json','edges.json']},seconds=time.perf_counter()-started)
 write(dest/'receipt.json',report);print(report,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);run(ap.parse_args().key)
