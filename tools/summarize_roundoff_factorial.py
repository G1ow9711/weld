"""Collect every registered arm without choosing a winner on one TRAIN piece."""
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
BASE=ROOT/'server_rebuild'

def main():
 paths={'old_old':BASE/'best_baseline_restore/legacy_runtime/source_10/run379/test_10/evaluation.json','new_new':BASE/'roundoff_gap_baseline_probe_v1/evaluation.json','new_legal_old_probability':BASE/'roundoff_factorial_v1/new_legal_old_probability/evaluation.json','old_legal_new_probability':BASE/'roundoff_factorial_v1/old_legal_new_probability/evaluation.json'}
 scores={k:read(p) for k,p in paths.items()};old=scores['old_old'];oldids={m['original_child_id'] for m in old['matches']};rows={}
 for k,r in scores.items():
  score=r if k=='old_old' else r['score'];value=score['categories']['overall'];ids={m['original_child_id'] for m in score['matches']}
  rows[k]=dict(**value,FN=value['G']-value['TP'],new_target_ids=sorted(ids-oldids),lost_target_ids=sorted(oldids-ids),score_sha256=digest(paths[k]))
 write(BASE/'roundoff_factorial_v1/comparison.json',dict(status='FOUR_ARMS_COMPLETE',arms=rows,registered_protocol_sha256=digest(BASE/'roundoff_gap_baseline_probe_v1/factorial_followup_protocol.json'),scope='One frozen TRAIN source. Separates legality/probability changes within this pipeline, not generalization or deployable mixed-model validation.',promoted=False));print(rows,flush=True)

if __name__=='__main__':main()
