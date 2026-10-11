"""Test the same whole-cloud support rule on two additional TRAIN shapes."""
import argparse
import gc
import sys
import traceback
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc
from weld_pipeline.sjg_global_support_train_pilot_939 import main as pilot


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    write(dest/'registered_protocol.json',dict(stage='TWO_HARD_TRAIN_GLOBAL_SUPPORT_PILOT',sources=['4','10'],
        maximum_planes=128,selection='canonical_inlier_count_then_extent',extent_only_selection_replaced=True,
        change_from_run124='plane hypothesis budget 32 to 128; same points, ranking and tolerances',
        original_eight_G=496,optimizer_updates=0,
        source_sha256=digest(__file__),full_contract_pass=False,start_utc=utc()))
    saved=sys.argv[:];rows=[]
    try:
        for key in ('4','10'):
            run=args.run+'/source_'+key
            sys.argv=[saved[0],'--run',run,'--source-key',key,'--maximum-planes','128']
            pilot();result=read(OUT/run/'receipt.json');assert result['status']=='GLOBAL_OBSERVED_SUPPORT_TRAIN_PILOT_COMPLETE'
            rows.append(dict(source_key=key,G=result['G'],supported=result['geometrically_supported'],receipt_sha256=digest(OUT/run/'receipt.json')))
            gc.collect()
        write(dest/'receipt.json',dict(status='HARD_TRAIN_GLOBAL_SUPPORT_PILOT_COMPLETE',sources=rows,
            optimizer_updates=0,not_weld_accuracy=True,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_sources=rows,error=traceback.format_exc(),end_utc=utc()));raise
    finally:sys.argv=saved

if __name__=='__main__':main()
