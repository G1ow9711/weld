"""Bind TRAIN-only complete-parent compatibility after freezing real inputs.

Nonmatching candidates remain unknown: this is not negative supervision.
"""
from pathlib import Path
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_parent_compatibility_1161 import compile_compatibility
from weld_pipeline.sjg_shared_prediction_reader_1236 import read_shared_predictions

ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'server_rebuild/parent_residual_v1'
def main():
 manifest=DEST/'inputs_frozen.json';receipt=read(manifest);cases=[]
 for case in receipt['cases']:
  key=case['source'];assert case['role']=='TRAIN' and key!='64'
  for f in case['files'].values():assert digest(ROOT/f['path'])==f['sha256']
  target=ROOT/'server_rebuild/labels_v1'/key/'targets.json';tr=read(target.parent/'receipt.json')
  assert tr['role']=='TRAIN' and digest(target)==tr['targets_sha256']
  result=compile_compatibility(read_shared_predictions(ROOT/case['files']['parents']['path']),read(target))
  result.update(source=key,target_sha256=digest(target),input_manifest_sha256=digest(manifest),nonmatching_candidate_label='UNKNOWN_NOT_CONFIRMED_NEGATIVE',fit_calls=0,DEV_reads=0)
  folder=DEST/('source_'+key);folder.mkdir(exist_ok=False);write(folder/'compatibility.json',result)
  row=dict(source=key,positive_candidate_count=sum(bool(x) for x in result['compatible_targets']),unknown_candidate_count=sum(not x for x in result['compatible_targets']),maximum_matched_children=result['maximum_matched_children'],uncovered_target_parents=result['uncovered_target_parents'],compatibility_sha256=digest(folder/'compatibility.json'));cases.append(row);print(row,flush=True)
 write(DEST/'compatibility_receipt.json',dict(cases=cases,fit_calls=0,DEV_reads=0,not_an_extraction_result=True,negative_labels_not_inferred_from_absence=True))
if __name__=='__main__':main()
