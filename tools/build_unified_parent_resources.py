"""Input-only resource constraints extending the frozen native pool to B0 defaults.

Inherited atom resources are retained. New interval children conflict only on the
same carrier or proven collinear overlap of the same contact faces. No GT reads.
"""
import collections,time
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def run(key):
 start=time.perf_counter();f=BASE/('source_'+key);r=read(f/'receipt.json')
 for name,sha in r['files'].items():assert digest(f/name)==sha
 old=ROOT/'server_rebuild/parent_hypotheses_v1'/('source_'+key);res=read(old/'resource_constraints.json');paths=read(old/'paths.json');paths=paths['native']+paths['added'];pc=read(f/'parent_children.json');seg=read(f/'child_segments.json');records=read(f/'child_records.json');metadata=read(f/'child_metadata.json')
 assert digest(old/'hypotheses.json')==r['hypothesis_sha256'];atom_children=collections.defaultdict(set);child_atoms=collections.defaultdict(set)
 for p,ids in zip(paths,pc):
  assert len(ids)==len(p['states'])
  for s,c in zip(p['states'],ids):atom_children[s//2].add(c);child_atoms[c].add(s//2)
 groups={tuple(sorted({c for a in g for c in atom_children[a]})) for g in res['groups']}
 inherited=len(groups);new_ids=sorted(set(range(len(seg)))-set(child_atoms));by_profile=collections.defaultdict(set);by_faces=collections.defaultdict(set)
 for i,(s,m) in enumerate(zip(seg,metadata)):
  by_profile[s['profile']].add(i)
  if s['kind']=='line':
   for face in m['face_pairs']:
    if len(face)==2 and min(face)>=0:by_faces[tuple(sorted(face))].add(i)
 pairs=set()
 for a in new_ids:
  sa=seg[a];candidates=set(by_profile[sa['profile']])
  if sa['kind']=='line':
   for face in metadata[a]['face_pairs']:
    if len(face)==2 and min(face)>=0:candidates.update(by_faces[tuple(sorted(face))])
  for b in candidates:
   if b==a:continue
   pair=tuple(sorted((a,b)))
   if pair in pairs:continue
   sb=seg[b]
   if sa['weld_type']!=sb['weld_type'] or sa['kind']!=sb['kind']:continue
   sidea=sa.get('predicted_side','UNKNOWN');sideb=sb.get('predicted_side','UNKNOWN')
   if sidea not in (None,'UNKNOWN') and sideb not in (None,'UNKNOWN') and sidea!=sideb:continue
   overlap=False
   if sa['profile']==sb['profile']:
    overlap=any(min(y,v)-max(x,u)>1e-6 for x,y in records[a]['projected_intervals_mm'] for u,v in records[b]['projected_intervals_mm'])
   elif sa['kind']=='line':
    pa=np.asarray(sa['points'])[[0,-1]];pb=np.asarray(sb['points'])[[0,-1]];v=pa[1]-pa[0];length=np.linalg.norm(v);v/=length;q=(pb-pa[0])@v;perp=np.linalg.norm(pb-pa[0]-q[:,None]*v,axis=1)
    overlap=perp.max()<1e-6 and min(length,q.max())-max(0.,q.min())>1e-6
   if overlap:pairs.add(pair)
 groups.update(pairs);groups.update((c,) for c in range(len(seg)));groups=sorted(groups)
 childgroups=collections.defaultdict(set)
 for gi,g in enumerate(groups):
  for c in g:childgroups[c].add(gi)
 parentresources=[];invalid=[]
 for pi,ids in enumerate(pc):
  counts=collections.Counter()
  for c in ids:counts.update(childgroups[c])
  parentresources.append(sorted(counts))
  if any(n>1 for n in counts.values()):invalid.append(pi)
 default=read(f/'default_configuration.json')['parent_indices'];counts=collections.Counter(g for p in default for g in parentresources[p]);badgroups=[g for g,n in counts.items() if n>1]
 # Frozen native constraints must be preserved exactly up to resource naming.
 old_invalid=set(res['internally_conflicting_hypotheses']);assert old_invalid<=set(invalid)
 report=dict(source=key,groups=groups,parent_resources=parentresources,internally_conflicting_parents=invalid,default_parent_indices=default,default_feasible=not badgroups and not(set(default)&set(invalid)),default_conflicting_groups=badgroups,inherited_groups=inherited,new_interval_children=len(new_ids),new_overlap_pairs=len(pairs),parents_sha256=digest(f/'parents.json'),input_receipt_sha256=digest(f/'receipt.json'),native_resources_sha256=digest(old/'resource_constraints.json'),source_sha256=digest(__file__),GT_reads=0,DEV_reads=0,coordinate_merging=False,same_request_not_used_as_material_identity=True,seconds=time.perf_counter()-start)
 write(f/'resource_constraints.json',report);print({k:report[k] for k in ['source','default_feasible','new_interval_children','new_overlap_pairs','seconds']},'invalid',len(invalid),'default_conflicts',len(badgroups),flush=True)
if __name__=='__main__':
 for k in ['21','BM79A_native_local','10','25','36']:run(k)
