"""TRAIN structural coverage, excluding existing heldout cohorts and DEV families."""
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 current=set(read(BASE/'reference_query_task_v1/protocol.json')['training_keys']);cohort={c['key'] for c in read(BASE/'coverage_cohort_protocol.json')['cases']};excluded=cohort-current;rows=[]
 for group in read(BASE/'group_manifest.json')['groups']:
  key=group['key'];folder=BASE/'labels'/key;inputs=BASE/'inputs'/key
  if group['role']!='EXPANSION_TRAIN' or key in excluded or not (folder/'receipt.json').exists() or not (inputs/'receipt.json').exists():continue
  receipt=read(folder/'receipt.json');assert receipt['group_manifest_sha256']==digest(BASE/'group_manifest.json');assert digest(folder/'targets.json')==receipt['targets_sha256'];target=read(folder/'targets.json');r=read(inputs/'receipt.json');assert r['family_id']==group['family_id'];children=[c for p in target['parents'] for c in p['children']];v3=sum(len(p['children'])==3 and all(c['weld_type']=='vertical' for c in p['children']) for p in target['parents']);rows.append(dict(key=key,family_id=group['family_id'],directory=receipt['directory'],input_points=r['original_points'],children=len(children),vertical_three_child_parents=v3,arc_children=sum(c['representation']['kind']=='arc' for c in children),in_current_six_TRAIN=key in current,labels_receipt_sha256=digest(folder/'receipt.json'),input_receipt_sha256=digest(inputs/'receipt.json')))
 rows.sort(key=lambda r:r['key']);out=BASE/'remaining_training_structure_coverage_v1';out.mkdir(exist_ok=False);write(out/'audit.json',dict(cases=rows,excluded_cohort_keys=sorted(excluded),group_manifest_sha256=digest(BASE/'group_manifest.json'),selection_not_based_on_model_scores=True,heldout_label_reads=0,DEV_reads=0,scope='Structural coverage in accepted TRAIN labels only, not geometric feasibility or runtime accuracy'))
 current_rows=[r for r in rows if r['in_current_six_TRAIN']];print('CURRENT',[(r['key'],r['vertical_three_child_parents'],r['arc_children']) for r in current_rows]);remaining=[r for r in rows if not r['in_current_six_TRAIN']];print('REMAINING V3',[(r['key'],r['input_points'],r['vertical_three_child_parents'],r['arc_children']) for r in remaining if r['vertical_three_child_parents']]);print('REMAINING ARC',[(r['key'],r['input_points'],r['arc_children']) for r in remaining if r['arc_children']]);print('remaining',len(remaining))
if __name__=='__main__':main()
