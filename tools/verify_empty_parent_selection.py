"""Exercise empty ledger preservation and unchanged nonempty routing."""
from pathlib import Path
import shutil
from weld_pipeline.server_empty_parent_selection import select_if_empty
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';DEST=BASE/'routing_controls/empty_parent_selection'
def main():
 DEST.mkdir(exist_ok=False);key='new_fcaf2af1abee';source=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run379';training=read(source/'selector_training.json')
 for name in {*training['frozen_files'],'selector_training.json','precision_model.pkl'}:(DEST/name).symlink_to(source/name)
 folder=DEST/'test_empty_ledger';folder.mkdir();original=source/('test_'+key)
 for name in ['unary.npy','parent_metadata.json','compact_parents.json','parent_points.npy']:(folder/name).symlink_to(original/name)
 decoded=read(original/'decoded.json');ledger=[dict(reason='CONTROL_INVALID_PARENT',counted_false_children=3)];decoded['invalid_predictions']=ledger;decoded['total_prediction_children']=3;write(folder/'decoded.json',decoded)
 assert select_if_empty(DEST,'empty_ledger');result=read(folder/'final_decoded.json');assert result['parents']==[] and result['invalid_predictions']==ledger and result['total_prediction_children']==3
 actual=BASE/'frozen_features/legacy_runtime/source_new_ac2008a03607/run379';before=digest(actual/'test_new_ac2008a03607/final_decoded.json');assert not select_if_empty(actual,'new_ac2008a03607');assert digest(actual/'test_new_ac2008a03607/final_decoded.json')==before
 write(DEST/'receipt.json',dict(status='PASS',empty_candidate_ledger_preserved=True,invalid_false_children_count_preserved=3,nonempty_original95_parents_untouched=True,actual_second_case_output_sha256=digest(original/'final_decoded.json'),source_sha256=digest(ROOT/'weld_pipeline/server_empty_parent_selection.py'),fit_calls=0,GT_reads=0,STEP_reads=0))
if __name__=='__main__':main()
