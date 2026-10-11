"""Read original TRAIN Excel parent split metadata; no runtime rule inferred."""
from pathlib import Path
import collections
import numpy as np
import openpyxl
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';DATA=ROOT.parent/'data/G_Data'
def main():
 labels=ROOT/'server_rebuild/labels_v1/10';r=read(labels/'receipt.json');target=read(labels/'targets.json');assert digest(labels/'targets.json')==r['targets_sha256'];name,sha=next((n,s) for n,s in r['files'].items() if n.endswith('.xlsx'));path=DATA/name;assert digest(path)==sha;book=openpyxl.load_workbook(path,read_only=True,data_only=False)
 try:
  it=book['焊缝信息'].iter_rows(values_only=True);headers=[str(v or '').split('\n')[-1] for v in next(it)];rows={str(row['ObjectId']):row for vals in it if (row:=dict(zip(headers,vals))).get('ObjectId')}
 finally:book.close()
 names=['HasEave (Any)','IsWorkRangeSplit (Any)','WorkRangeFrontSplit','WorkRangeBackSplit','IsSplitedForDualRobot (Any)','HasMid (Any)'];cases=[]
 for i,p in enumerate(target['parents']):
  if len(p['children'])!=3 or not all(c['weld_type']=='vertical' for c in p['children']):continue
  row=rows[p['original_parent_id']];assert int(row['SegmentCount'])==3;lengths=[float(np.linalg.norm(np.diff(np.asarray(c['representation']['endpoints_mm']),axis=0))) for c in p['children']];cases.append(dict(parent=i,original_parent_id=p['original_parent_id'],child_lengths_mm=lengths,child_length_fractions=(np.array(lengths)/sum(lengths)).tolist(),parent_scope_flags={n:row.get(n) for n in names}))
 counts={n:dict(collections.Counter(str(c['parent_scope_flags'][n]) for c in cases)) for n in names};out=BASE/'original10_missing_endpoint_diagnostic_v1';write(out/'excel_split_metadata.json',dict(cases=cases,counts=counts,source_excel_sha256=sha,scope='Original TRAIN parent-scope Excel metadata only; no flags broadcast to children, no operational split algorithm inferred from length fractions, no runtime Excel lookup',fit_calls=0,DEV_reads=0));print(counts)
if __name__=='__main__':main()
