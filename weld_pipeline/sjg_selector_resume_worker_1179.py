"""Single restore control and continuation of the original 112-update selector."""
import argparse,time,traceback
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_selector_resume_state_1178 import control,reuse_columns
from weld_pipeline.sjg_selector_trial_worker_1174 import bindings
from weld_pipeline.sjg_relation_column_cache_1173 import RelationColumnCache,CachedParentRelations
from weld_pipeline.sjg_recall_set_worker_1162 import train_selector
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--source',default='');ap.add_argument('--limit-seconds',type=float,required=True)
    args=ap.parse_args();dest=OUT/args.run;source=OUT/'run362';begin=time.perf_counter()
    try:
        check_resources()
        if args.stage=='resume_control':control(source,dest)
        elif args.stage=='resume_train':
            assert read(dest/'resume_control.json')['status']=='CHECKPOINT_RESTORE_AND_NEXT_ADAM_BITWISE_PASSED'
            cache=RelationColumnCache(dest/'training_relation_cache',bindings(dest,KEYS))
            reused=reuse_columns(source/'training_relation_cache',cache)
            assert reused['complete_columns_reused']==375
            write(dest/'relation_cache_reuse.json',reused);print('PURE_INPUT_CACHE_REUSED',reused,flush=True)
            try:
                train_selector(dest,relation_factory=lambda parents,metadata,key:CachedParentRelations(parents,metadata,key,cache),
                    resume_from=source,time_limit_seconds=max(.001,args.limit_seconds-(time.perf_counter()-begin)))
            finally:write(dest/'training_relation_cache_receipt.json',cache.receipt())
        else:raise ValueError(args.stage)
        write(dest/f'stage_{args.stage}_{args.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-begin,
            worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{args.stage}_{args.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-begin));raise


if __name__=='__main__':main()
