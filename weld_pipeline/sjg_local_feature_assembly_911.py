"""Reconstruct original-point feature arrays only after streamed GPU inference."""
import argparse
import gc
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);data=OUT/'run105';records=[]
    assert read(data/'receipt.json')['status']=='LOCAL_STREAM_PTV3_FEATURES_COMPLETE'
    manifest=read(data/'all_predictions_frozen.json')
    write(dest/'registered_protocol.json',dict(stage='ASSEMBLE_LOCAL_RAW_POINT_FEATURES',source_sha256=digest(__file__),
        source_run='run105',manifest_sha256=digest(data/'all_predictions_frozen.json'),
        all_original_point_indices_preserved=True,label_reads=0,optimizer_updates=0,start_utc=utc()))
    try:
        for case in manifest['cases']:
            check_resources();key=case['source_key'];count=case['original_points']
            arrays=dict(embedding=np.empty((count,16),np.float32),normal=np.empty((count,3),np.float32),
                        boundary=np.empty(count,np.float32),core_coverage=np.zeros(count,np.uint8))
            for entry in case['blocks']:
                file=data/entry['file'];assert digest(file)==entry['sha256']
                with np.load(file,allow_pickle=False) as z:
                    ids=z['original_point_ids'];assert len(ids)==entry['core_points']
                    assert ids.dtype==np.int64 and len(np.unique(ids))==len(ids)
                    assert ids.min()>=0 and ids.max()<count and not np.any(arrays['core_coverage'][ids])
                    for name in ('embedding','normal','boundary'):
                        assert np.isfinite(z[name]).all();arrays[name][ids]=z[name]
                    arrays['core_coverage'][ids]+=1
            assert np.all(arrays['core_coverage']==1)
            filename=dest/f'source_{key}_features.npz';np.savez(filename,**arrays)
            row=dict(source_key=key,role=case['role'],points=count,coordinate_frame='workpiece_local',unit='mm',
                input_file=case['input_file'],input_sha256=case['input_sha256'],
                feature_file=filename.name,feature_sha256=digest(filename),all_original_points_covered_once=True)
            records.append(row);write(dest/f'source_{key}_freeze.json',row)
            print('LOCAL_FEATURES_ASSEMBLED',key,count,flush=True);del arrays;gc.collect()
        write(dest/'all_predictions_frozen.json',dict(sources=records,label_reads=0,original_G=496))
        write(dest/'receipt.json',dict(status='LOCAL_FEATURES_ASSEMBLED_AND_VERIFIED',points=sum(r['points'] for r in records),
            original_eight_G=496,optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
