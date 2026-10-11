"""Conservative input-byte/name grouping for training expansion planning.

No split assignment: rotation/resampling equivalence still requires checking.
"""
from pathlib import Path
import collections,re
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'migration_checks'
def main():
 audit=read(BASE/'dataset_audit.json');inventory={r['directory']:r for r in read(BASE/'data_inventory.json')['cases']};rows=audit['cases'];parent=list(range(len(rows)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b):parent[find(b)]=find(a)
 seen={}
 for i,r in enumerate(rows):
  path=Path(inventory[r['directory']]['pointcloud']);name=re.sub(r'_GW1_AllFaces$', '',path.stem,flags=re.I).strip('_').casefold();name=re.sub(r'\s+','',name)
  r=dict(r,normalized_input_name=name);rows[i]=r
  for token in [('sha',r['pcd_sha256']),('name',name)]:
   if token in seen:union(i,seen[token])
   else:seen[token]=i
 groups=collections.defaultdict(list)
 for i,r in enumerate(rows):groups[find(i)].append(r)
 known={}
 for k in ['21','BM79A_native_local','10','25','36']:
  r=read(ROOT/'server_rebuild/inputs_v1'/k/'receipt.json');known[str(Path(r['source_file']).parent)]='EXISTING_TRAIN_'+k
 output=[]
 for group in groups.values():
  directories=[r['directory'] for r in group];dev=any('442-DK1A' in d for d in directories);existing=sorted({known[d] for d in directories if d in known});legacy=any(not d.startswith('129/') for d in directories)
  role='PROTECTED_DEV_FAMILY' if dev else 'EXISTING_TRAIN_FAMILY' if existing else 'LEGACY_ROLE_UNRESOLVED' if legacy else 'NEW_UNASSIGNED_NEEDS_GEOMETRIC_DEDUP'
  output.append(dict(directories=directories,names=sorted({r['normalized_input_name'] for r in group}),pcd_sha256=sorted({r['pcd_sha256'] for r in group}),status=role,existing_train_roles=existing,versions=[dict(directory=r['directory'],workbooks=[{k:b[k] for k in ['file','sha256','parents','children','types','curved_parents']} for b in r['workbooks']]) for r in group]))
 counts=collections.Counter(r['status'] for r in output);report=dict(groups=output,group_counts=dict(counts),split_assigned=False,geometric_equivalence_not_proved=True,no_new_fit=True,DEV_labels_not_loaded_by_this_tool=True,uses_existing_global_metadata_audit=True,audit_sha256=digest(BASE/'dataset_audit.json'),inventory_sha256=digest(BASE/'data_inventory.json'),source_sha256=digest(__file__))
 write(BASE/'expansion_groups_v1.json',report);print(dict(counts),flush=True)
 for g in output:
  if g['status']=='NEW_UNASSIGNED_NEEDS_GEOMETRIC_DEDUP' and any(b['curved_parents'] for v in g['versions'] for b in v['workbooks']):print('NEW_CURVE_GROUP',g['directories'],flush=True)
if __name__=='__main__':main()
