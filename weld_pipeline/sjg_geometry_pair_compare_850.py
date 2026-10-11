"""Freeze both preregistered geometry pairing arms before scoring either."""
import argparse
import sys
import traceback
from weld_pipeline.sjg_prepare_776 import OUT
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc
from weld_pipeline.sjg_geometry_pair_849 import GeometryPairModel
from weld_pipeline.sjg_evaluate_pairs_847 import main as predict,evaluate_frozen


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    arms=[('raw_geometry',False),('invariant_geometry',True)]
    write(dest/'registered_protocol.json',dict(stage='MATCHED_GEOMETRY_PAIR_COMPARISON',
        training_run='run59',baseline='run57',arms=[a for a,_ in arms],original_G=496,
        source_sha256=digest(__file__),no_threshold_tuning=True,all_arms_frozen_before_any_scoring=True,
        full_contract_pass=False,start_utc=utc()))
    try:
        frozen=[]
        for arm,invariant in arms:
            sys.argv=[__file__,'--run',args.run+'/'+arm]
            predict(checkpoint_run='run59/'+arm,model_factory=lambda flag=invariant:GeometryPairModel(flag),score_outputs=False)
            frozen.append(dict(arm=arm,sha256=digest(dest/arm/'all_outputs_frozen_before_GT.json')))
        write(dest/'all_arms_frozen_before_GT.json',dict(arms=frozen,all_before_GT=True))
        results={}
        for arm,_ in arms:results[arm]=evaluate_frozen(dest/arm,'run59/'+arm)
        baseline=read(OUT/'run57/receipt.json')
        write(dest/'receipt.json',dict(status='GEOMETRY_PAIR_COMPARISON_EVALUATED',baseline=baseline,
            arms=results,optimizer_updates=0,original_G=496,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
