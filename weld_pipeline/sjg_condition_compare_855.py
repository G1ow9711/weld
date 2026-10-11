"""Freeze both endpoint arms and fixed learned interval/parent outputs before GT."""
import argparse
import sys
import traceback
from weld_pipeline.sjg_prepare_776 import ROOT,OUT
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc
from weld_pipeline.sjg_adjacency_model_840 import AdjacencySequenceModel
from weld_pipeline.sjg_adjacency_eval_842 import ThirdSurfaceContext
from weld_pipeline.sjg_event_pairs_824 import decode_event_pairs
from weld_pipeline.sjg_free_sequence_823 import main as infer_sequence
from weld_pipeline.sjg_all_interval_data_844 import main as prepare_pairs
from weld_pipeline.sjg_evaluate_pairs_847 import main as infer_pairs,evaluate_frozen


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);arms=['control','condition_dropout'];saved=sys.argv[:]
    assert read(OUT/'run63/receipt.json')['status']=='TWO_CONDITION_ARMS_TRAINED'
    write(dest/'registered_protocol.json',dict(stage='CONDITION_DROPOUT_FREE_OUTPUT_COMPARISON',training_run='run63',
        arms=arms,original_G=496,optimizer_updates=0,fixed_pairing_checkpoint='run56',fixed_parent_link_checkpoint='run47',
        no_new_pair_training=True,no_threshold_tuning=True,all_arms_frozen_before_any_GT=True,
        source_sha256=digest(__file__),candidate_generator_sha256=digest(ROOT/'weld_pipeline/sjg_all_interval_data_844.py'),
        pair_driver_sha256=digest(ROOT/'weld_pipeline/sjg_evaluate_pairs_847.py'),
        improvement_rule='development TP strictly increases and FP does not increase; all8 TP increases and FP does not increase versus matched control',
        full_contract_pass=False,start_utc=utc()))
    try:
        frozen=[]
        for arm in arms:
            folder=dest/arm;folder.mkdir()
            sys.argv=[saved[0],'--run',str((folder/'sequence').relative_to(OUT)),'--checkpoint-run','run63/'+arm]
            infer_sequence(decode_event_pairs,score_outputs=False,model_factory=AdjacencySequenceModel,context_factory=ThirdSurfaceContext)
            sys.argv=[saved[0],'--run',str((folder/'candidates').relative_to(OUT))]
            prepare_pairs(base=folder/'sequence',create_labels=False,expected_candidates=None)
            sys.argv=[saved[0],'--run',str((folder/'output').relative_to(OUT))]
            infer_pairs(data=folder/'candidates',score_outputs=False)
            frozen.append(dict(arm=arm,sequence_sha256=digest(folder/'sequence/all_predictions_frozen.json'),
                candidates_sha256=digest(folder/'candidates/all_inputs_frozen.json'),outputs_sha256=digest(folder/'output/all_outputs_frozen_before_GT.json')))
        write(dest/'all_arms_frozen_before_GT.json',dict(arms=frozen,all_before_GT=True))
        results={arm:evaluate_frozen(dest/arm/'output','run56') for arm in arms}
        write(dest/'receipt.json',dict(status='CONDITION_DROPOUT_COMPARISON_EVALUATED',arms=results,original_G=496,
            optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise
    finally:sys.argv=saved


if __name__=='__main__':main()
