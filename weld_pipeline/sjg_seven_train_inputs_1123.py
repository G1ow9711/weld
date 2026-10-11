"""Restore only the original seven TRAIN feature inputs; no model fitting."""
import gc,time,traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_joint_object_features_1106 import prepare


def main():
    dest=OUT/'run308';dest.mkdir(exist_ok=False);start=time.perf_counter();rows=[]
    try:
        coverage=read(OUT/'run307/receipt.json')['full_parent_coverage_counts']
        assert coverage['C0']==26 and 26-coverage['C1']>13
        review=read(OUT/'pro_review_run306/receipt.json');assert review['assistant_sentinel_verified'] and review['generation_stopped']
        write(dest/'registered_protocol.json',dict(stage='RESTORE_ORIGINAL_SEVEN_TRAIN_INPUTS_ONLY',
            TRAIN=['21','4','10','25','36','42','BM79A_native_local'],TRAIN_G=363,DEV='64',full_pilot_G=500,
            new_fit_calls=0,feature_dimension=128,new_features=False,DEV_labels_read=0,
            source_sha256=digest(__file__),review_receipt_sha256=digest(OUT/'pro_review_run306/receipt.json'),utc=utc()))
        for key in ('4','25','36','42'):
            check_resources();prepare(key,'run308',geometry_run='run285');gc.collect()
        for key in ('21','4','10','25','36','42','BM79A_native_local'):
            run='run287' if key in ('21','BM79A_native_local','10') else 'run308'
            folder=OUT/run/f'source_{key}';feature=read(folder/'receipt.json')
            assert digest(folder/'object_features.npy')==feature['features_sha256']
            alias=next(c for c in read(OUT/'run288/all_alias_maps_frozen_before_labels.json')['cases'] if c['source']==key)
            rows.append(dict(source=key,feature_run=run,feature_receipt_sha256=digest(folder/'receipt.json'),
                feature_sha256=feature['features_sha256'],alias_map_sha256=alias['map_sha256']))
        write(dest/'receipt.json',dict(status='ORIGINAL_SEVEN_TRAIN_FEATURES_FROZEN',cases=rows,
            new_fit_calls=0,new_feature_channels=0,DEV_labels_read=0,TRAIN_G=363,full_pilot_G=500,
            extraction_correct_rate=None,extraction_precision_mm=None,seconds=time.perf_counter()-start,utc=utc()))
        print('ORIGINAL_SEVEN_TRAIN_INPUTS_READY_NO_TRAINING',time.perf_counter()-start,flush=True)
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start,new_fit_calls=0,utc=utc()));raise


if __name__=='__main__':main()
