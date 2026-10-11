"""Original TRAIN multi-child event representation, no inference accuracy claim."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def main():
 key='10';data=BASE/'positive_query_head_v1'/key;records=read(data/'records.json');sup=BASE/'original_boundary_anchor_supervision_v1'/key;r=read(sup/'receipt.json');assert digest(sup/'targets.npz')==r['supervision_sha256'];assert digest(data/'receipt.json')==r['query_receipt_sha256'];truthfile=ROOT/'server_rebuild/labels_v1/10/targets.json';assert digest(truthfile)==r['target_sha256'];truth=read(truthfile);z=np.load(sup/'targets.npz');rows=z['query_rows'];left=np.diff(z['left_offsets'])>0;right=np.diff(z['right_offsets'])>0;both=left&right;represented={tuple(o) for i in rows for o in records[i]['owners']};available={tuple(o) for j,i in enumerate(rows) if both[j] for o in records[i]['owners']};parents=[]
 for i,p in enumerate(truth['parents']):
  if len(p['children'])!=3 or not all(c['weld_type']=='vertical' for c in p['children']):continue
  children=[]
  for j,c in enumerate(p['children']):
   ends=np.asarray(c['representation']['endpoints_mm']);children.append(dict(child=j,length_mm=float(np.linalg.norm(ends[1]-ends[0])),fixed_query_represented=(i,j) in represented,both_input_endpoints_available=(i,j) in available))
  parents.append(dict(parent=i,children=children,all_children_available=all(c['both_input_endpoints_available'] for c in children)))
 report=dict(parents=parents,vertical_three_child_parents=len(parents),V3_children_represented=sum(c['fixed_query_represented'] for p in parents for c in p['children']),V3_children_both_events=sum(c['both_input_endpoints_available'] for p in parents for c in p['children']),V3_parents_all_children_events=sum(p['all_children_available'] for p in parents),scope='Offline TRAIN endpoint event availability on previous witnessed profiles and fixed positive queries; not parent assembly, complete event upper bound or runtime accuracy',supervision_receipt_sha256=digest(sup/'receipt.json'),fit_calls=0,DEV_reads=0);write(sup/'vertical_three_child_capacity.json',report);print({k:v for k,v in report.items() if k!='parents'})
if __name__=='__main__':main()
