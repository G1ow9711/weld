"""Audit geometric annotation compatibility before rebuilding connection supervision.

No runtime labels, endpoint overrides, physical-negative labels, or fitted models.
Line containment is only necessary geometric evidence, not physical side identity.
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT, read, write, digest
from weld_pipeline.sjg_stream_profile_geometry_1149 import read_profile_geometry
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.sjg_joint_link_runtime_1107 import node_geometry

EXP=ROOT/'server_rebuild/expansion_v1'
DEST=EXP/'connection_fragment_ownership_audit_v1'
roles={v['key']:v['role'] for v in read(EXP/'group_manifest.json')['groups']}
KEYS=tuple(sorted(p.parent.name[5:] for p in (EXP/'frozen_features/legacy_runtime').glob('source_*/run379/test_*/selected_graph.json') if roles.get(p.parent.name[5:])=='EXPANSION_TRAIN'))

def main():
 DEST.mkdir(exist_ok=False)
 protocol=dict(keys=list(KEYS),role='TRAIN',tolerance_mm=3.,line_rule='Both finite candidate endpoints within 3 mm of the same finite reference line segment; nonzero projected extent. All compatible owners retained.',arcs='Excluded and counted: endpoint containment cannot certify arc containment.',unknown='No owner, multiple owners and physical side identity remain unresolved. No negative labels inferred from absence.',purpose='Supervision feasibility audit only; not extraction evaluation or physical identity proof',fit_calls=0)
 protocol['group_manifest_sha256']=digest(EXP/'group_manifest.json')
 write(DEST/'protocol.json',protocol)
 summaries=[]
 for key in KEYS:
  base=EXP/'frozen_features/legacy_runtime'/('source_'+key)
  folder=base/'run379'/('test_'+key)
  tp=EXP/'labels'/key/'targets.json';target=read(tp)
  assert target['role']=='TRAIN' and digest(tp)==read(tp.parent/'receipt.json')['targets_sha256']
  pp=base/'run185'/('source_'+key)/'profiles.json';gp=base/'run285'/('source_'+key+'_geometry.npz')
  profiles=read_profile_geometry(pp,retain_fields=('support_intervals_mm','unobserved_gaps_mm'))
  with np.load(gp) as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
  nodes=read(folder/'selected_graph.json')['nodes']; refs=[]
  for pi,p in enumerate(target['parents']):
   for ci,c in enumerate(p['children']):
    if c['representation']['kind']=='line':refs.append((pi,ci,c,np.asarray(c['representation']['endpoints_mm'],float)))
  ends=np.array([r[3] for r in refs]).reshape(-1,2,3)
  vec=ends[:,1]-ends[:,0];length=np.linalg.norm(vec,axis=1);assert np.all(length>0)
  direction=vec/length[:,None];rows=[];counts=Counter();covered=set()
  for ni,node in enumerate(nodes):
   if graph.curve(graph.profile(node['fragment'])).kind!='line':
    rows.append(dict(node=ni,status='ARC_UNAUDITED',owners=[]));counts['arc_unaudited']+=1;continue
   e=node_geometry(node,graph)[0];delta=e[None,:,:]-ends[:,None,0,:]
   t=np.einsum('nek,nk->ne',delta,direction)
   nearest=ends[:,None,0,:]+np.clip(t,0,length[:,None])[:,:,None]*direction[:,None,:]
   error=np.linalg.norm(e[None,:,:]-nearest,axis=2).max(axis=1)
   ix=np.flatnonzero((error<=3.)&(np.abs(t[:,1]-t[:,0])>1e-9))
   owners=[dict(parent=refs[j][0],child=refs[j][1],child_id=refs[j][2]['original_child_id'],projected_endpoints_mm=t[j].tolist(),reference_length_mm=float(length[j]),finite_containment_error_mm=float(error[j])) for j in ix]
   status='UNIQUE_GEOMETRIC_OWNER' if len(owners)==1 else ('AMBIGUOUS_GEOMETRIC_OWNERS' if owners else 'NO_GEOMETRIC_OWNER')
   counts[status]+=1;covered.update((v['parent'],v['child']) for v in owners)
   rows.append(dict(node=ni,status=status,owners=owners))
  manifest=read(folder/'edge_manifest.json');m=manifest['edge_count']
  edges=np.memmap(folder/'edges.i32',dtype='<i4',mode='r',shape=(m,2)) if m else np.empty((0,2),np.int32)
  unique=np.array([len(r['owners'])==1 for r in rows]);parents=np.array([r['owners'][0]['parent'] if unique[i] else -1 for i,r in enumerate(rows)])
  children=np.array([r['owners'][0]['child'] if unique[i] else -1 for i,r in enumerate(rows)])
  ec=Counter()
  for first in range(0,m,65536):
   a,b=edges[first:first+65536].T;known=unique[a]&unique[b];samep=parents[a]==parents[b];samec=children[a]==children[b]
   ec['unresolved_endpoint_owner']+=int((~known).sum())
   ec['unique_owners_different_parents']+=int((known&~samep).sum())
   ec['unique_owners_same_child']+=int((known&samep&samec).sum())
   ec['unique_owners_same_parent_different_children']+=int((known&samep&~samec).sum())
  if m:edges._mmap.close()
  summary=dict(source=key,nodes=len(nodes),node_counts=dict(counts),line_reference_children=len(refs),line_children_with_compatible_nodes=len(covered),total_reference_children=sum(len(p['children']) for p in target['parents']),edges=m,edge_counts=dict(ec))
  write(DEST/(key+'.json'),dict(summary=summary,rows=rows,bindings={str(p.relative_to(ROOT)):digest(p) for p in (tp,pp,gp,folder/'selected_graph.json',folder/'edge_manifest.json',folder/'edges.i32')},not_training_labels=True))
  summaries.append(summary);print(json.dumps(summary),flush=True)
 write(DEST/'receipt.json',dict(status='GEOMETRIC_COMPATIBILITY_AUDIT_COMPLETE',cases=summaries,fit_calls=0,protected_DEV_reads=0,physical_identity_proven=False,promoted=False))

if __name__=='__main__':main()
