"""Offline diagnostic: preserve distinct face pairs on the same solid pair.

Sample agreement is not continuous coverage or annotation completeness.
"""
from pathlib import Path
import itertools
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/step_contact_scope_audit_v1'
def main():
 cases=[]
 for key in ['10','25','36']:
  path=BASE/('source_'+key)/'sampled_local_contact_scope.json';source=read(path);rows=[]
  for row in source['records']:
   if row['kind']!='line':continue
   choices=[]
   for sample in row['interior_samples']:
    pairs=set()
    for a,b in itertools.combinations(sample['faces'],2):
     va=set(a['volumes']);vb=set(b['volumes'])
     if len(va)!=1 or len(vb)!=1 or va&vb:continue
     na=np.asarray(a['normal']);nb=np.asarray(b['normal']);sine=np.linalg.norm(np.cross(na,nb))/(np.linalg.norm(na)*np.linalg.norm(nb))
     if sine<=np.sin(np.deg2rad(.02)):continue
     pairs.add(tuple(sorted([a['face'],b['face']])))
    choices.append(pairs)
   common=set.intersection(*choices) if choices else set()
   rows.append(dict(parent=row['parent'],child=row['child'],volume_pair=row['interior_volume_pair'],face_pair_choices_per_sample=[sorted(x) for x in choices],common_face_pairs=sorted(common),unique_common_sampled_face_pair=len(common)==1,all_samples_locally_unambiguous=all(len(x)==1 for x in choices),continuous_contact_proved=False))
  ambiguous=[r for r in rows if not r['unique_common_sampled_face_pair']];out=dict(source=key,rows=rows,total=len(rows),unique_common_pairs=len(rows)-len(ambiguous),every_sample_unique=sum(r['all_samples_locally_unambiguous'] for r in rows),unresolved_parent_children=[[r['parent'],r['child']] for r in ambiguous],sampled_scope_sha256=digest(path),normal_parallel_tolerance_deg=.02,negative_labels_released=False,continuous_contact_proved=False,annotation_closure_proved=False,source_sha256=digest(__file__))
  write(path.parent/'sampled_contact_face_pairs.json',out);cases.append({k:v for k,v in out.items() if k!='rows'});print(cases[-1],flush=True)
 write(BASE/'sampled_contact_face_pairs_summary.json',dict(cases=cases,offline_only=True))
if __name__=='__main__':main()
