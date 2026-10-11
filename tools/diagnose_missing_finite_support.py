import json,pathlib,numpy as np
from scipy.spatial import cKDTree
root=pathlib.Path(__file__).resolve().parents[1]/'server_rebuild'
for key in ['10','25','36','BM79A_native_local']:
 fs=json.loads((root/'candidates_v5'/key/'faces.json').read_text()); ns=np.array([f['normal'] for f in fs]); off=np.array([f['offset'] for f in fs]); targets=json.loads((root/'labels_v1'/key/'targets.json').read_text()); audit=json.loads((root/'candidates_v5'/key/'coverage_audit.json').read_text()); xyz=np.load(root/'inputs_v1'/key/'xyz_mm.npy',mmap_mode='r'); tree=cKDTree(xyz)
 rows=[]
 for r in audit['rows']:
  c=targets['parents'][r['parent']]['children'][r['child']]
  if c['original_kind']!='line':continue
  q=np.array(c['representation']['endpoints_mm']); res=np.abs(q@ns.T-off).max(0);ids=np.flatnonzero(res<.05);near=[(int(i),round(float(res[i]),6)) for i in np.argsort(res)[:3]]
  if not r['union_support_3mm']:
   samples=np.linspace(q[0],q[1],101); dist=tree.query(samples)[0]
   rows.append(dict(parent=r['parent'],child=r['child'],type=c['weld_type'],matching_planes=ids.tolist(),nearest_planes=near,nearest_xyz_max=float(dist.max()),endpoints=q.tolist(),union_error=r['union_support_max_mm']))
 (root/'candidates_v5'/key/'missing_support_diagnosis.json').write_text(json.dumps(rows,indent=2))
 print(key,'missed',len(rows),'with_two_planes',sum(len(r['matching_planes'])>=2 for r in rows),'rows',json.dumps(rows[:4]))
