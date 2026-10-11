"""Input-only resource groups for finite unscreened-child parent hypotheses."""
import collections
import numpy as np

def build_resources(atoms,metadata,aliases,origins,node_resource_rows):
 node_atoms=collections.defaultdict(set)
 for ai,rows in enumerate(aliases):
  for row in rows:
   for node in origins[row]['graph_nodes']:node_atoms[node].add(ai)
 groups={tuple(sorted(v)) for v in node_atoms.values() if len(v)>1}
 for nodes in node_resource_rows:
  members=set()
  for node in nodes:members.update(node_atoms[int(node)])
  if len(members)>1:groups.add(tuple(sorted(members)))
 inherited=len(groups);by_contact=collections.defaultdict(set)
 for i,(p,m) in enumerate(zip(atoms,metadata)):
  if p['segments'][0]['kind']!='line':continue
  for pair in m['face_pairs']:
   if len(pair)==2 and min(pair)>=0:by_contact[tuple(sorted(pair))].add(i)
 pairs=set()
 for members in by_contact.values():
  members=sorted(members)
  for pos,a in enumerate(members):
   pa=np.asarray(atoms[a]['segments'][0]['points'])[[0,-1]];v=pa[1]-pa[0];length=np.linalg.norm(v);v=v/length
   for b in members[pos+1:]:
    if (a,b) in pairs:continue
    pb=np.asarray(atoms[b]['segments'][0]['points'])[[0,-1]];q=(pb-pa[0])@v;perp=np.linalg.norm(pb-pa[0]-q[:,None]*v,axis=1)
    if perp.max()<1e-6 and min(length,q.max())-max(0.,q.min())>1e-6:pairs.add((a,b))
 groups.update(pairs)
 # Every exact geometry/contact atom is also a one-use resource.
 groups.update((i,) for i in range(len(atoms)))
 return sorted(groups),dict(inherited_conflict_groups=inherited,collinear_contact_overlap_pairs=len(pairs),exact_atom_resources=len(atoms),overlap_tolerance_mm=1e-6,shared_endpoints_not_conflicts=True,GT_reads=0,new_arc_overlap_rule_not_added=True)

def hypothesis_resources(paths,groups):
 atom_groups=collections.defaultdict(set)
 for ri,members in enumerate(groups):
  for i in members:atom_groups[i].add(ri)
 rows=[];invalid=[]
 for pi,path in enumerate(paths):
  counts=collections.Counter()
  for state in path['states']:counts.update(atom_groups[state//2])
  rows.append(sorted(counts))
  if any(v>1 for v in counts.values()):invalid.append(pi)
 return rows,invalid
