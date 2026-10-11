"""Freeze both material-supervision arms before reading any evaluation teachers."""
import argparse
import sys
import traceback
from weld_pipeline.sjg_prepare_776 import ROOT,OUT
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc
from weld_pipeline.sjg_material_evaluate_862 import main as infer,score_frozen


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);saved=sys.argv[:];arms=['sampled_control','dense_material']
    assert read(OUT/'run70/receipt.json')['status']=='TWO_DENSE_MATERIAL_ARMS_TRAINED'
    write(dest/'registered_protocol.json',dict(stage='DENSE_MATERIAL_PURE_XYZ_COMPARISON',training_run='run70',arms=arms,
        source_sha256=digest(__file__),inference_driver_sha256=digest(ROOT/'weld_pipeline/sjg_material_evaluate_862.py'),
        all_arms_before_reference_reads=True,fixed_distance_threshold=.5,no_development_selection=True,
        weld_model_not_replaced=True,full_contract_G=496,full_contract_pass=False,start_utc=utc()))
    try:
        frozen=[]
        for arm in arms:
            sys.argv=[saved[0],'--run',str((dest/arm).relative_to(OUT))]
            infer(checkpoint_run='run70/'+arm,score_outputs=False)
            frozen.append(dict(arm=arm,sha256=digest(dest/arm/'all_predictions_frozen.json')))
        write(dest/'all_arms_frozen_before_reference.json',dict(arms=frozen,all_before_reference=True))
        metrics={}
        for arm in arms:
            metrics[arm]=score_frozen(dest/arm);r=read(dest/arm/'receipt.json')
            r.update(status='EIGHT_RAW_MATERIAL_RELATIONS_FROZEN_AND_EVALUATED',rows=metrics[arm],end_utc=utc());write(dest/arm/'receipt.json',r)
        write(dest/'receipt.json',dict(status='DENSE_MATERIAL_COMPARISON_COMPLETE',metrics=metrics,
            optimizer_updates=0,full_contract_G=496,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise
    finally:sys.argv=saved


if __name__=='__main__':main()
