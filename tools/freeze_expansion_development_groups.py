"""Freeze conservative family-grouped DEVELOPMENT expansion roles, no fitting.

This is not the contractual acceptance split. Residual crop/derivative similarity
risk remains explicit; suspected similar families are deliberately over-grouped.
"""
from pathlib import Path
import hashlib,collections
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'migration_checks';DEST=ROOT/'server_rebuild/expansion_v1'
def main():
 DEST.mkdir(exist_ok=False);old=read(B/'expansion_groups_v1.json')['groups'];pairs=read(B/'geometry_duplicate_pair_candidates_v1.json')['pairs'];parent=list(range(len(old)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b):parent[find(b)]=find(a)
 index={d:i for i,g in enumerate(old) for d in g['directories']};links=[]
 for p in pairs:
  if p['within_conservative_retrieval_band']:
   a=index[p['a_directory']];b=index[p['b_directory']];union(a,b);links.append(dict(a=p['a_directory'],b=p['b_directory'],reason='Conservative suspected derivative grouping, not physical identity'))
 # Known numbered variants of the same design family stay together.
 family=[i for i,g in enumerate(old) if any('c1fr83a' in n for n in g['names'])]
 for i in family[1:]:union(family[0],i)
 clusters=collections.defaultdict(list)
 for i,g in enumerate(old):clusters[find(i)].append(g)
 output=[]
 for members in clusters.values():
  dirs=sorted({d for g in members for d in g['directories']});hashes=sorted({h for g in members for h in g['pcd_sha256']});fid=hashlib.sha256(('EXPANSION_FAMILY_V1\n'+'\n'.join(hashes)).encode()).hexdigest();statuses={g['status'] for g in members}
  if 'PROTECTED_DEV_FAMILY' in statuses:role='PROTECTED_ORIGINAL_DEV'
  elif 'EXISTING_TRAIN_FAMILY' in statuses:role='EXISTING_TRAIN_ONLY_NO_NEW_VERSION_ADMISSION'
  elif 'LEGACY_ROLE_UNRESOLVED' in statuses:role='EXCLUDED_LEGACY_ROLE_UNRESOLVED'
  else:role='EXPANSION_DEV_VALID' if int(hashlib.sha256(('SPLIT_1031_'+fid).encode()).hexdigest(),16)%5==0 else 'EXPANSION_TRAIN'
  output.append(dict(family_id=fid,key='new_'+fid[:12],role=role,directories=dirs,pcd_sha256=hashes,representative_directory=dirs[0] if role.startswith('EXPANSION_') else None,all_versions_kept_in_same_role=True))
 counts=dict(collections.Counter(g['role'] for g in output));write(DEST/'group_manifest.json',dict(groups=output,counts=counts,assignment_before_new_model_training=True,assignment_not_based_on_model_scores=True,split_seed=1031,validation_hash_modulus=5,formal_contract_split=False,residual_unknown_cropped_derivative_leakage_risk=True,links=links,source_bindings={n:digest(B/n) for n in ['expansion_groups_v1.json','geometry_duplicate_pair_candidates_v1.json','duplicate_full_cloud_v1/summary.json']},source_sha256=digest(__file__)));print(counts,flush=True)
if __name__=='__main__':main()
