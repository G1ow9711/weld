"""Verify and assemble all original-point surface/material predictions."""
import argparse
import gc
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--feature-run',default='run165');args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    source=OUT/args.feature_run;assert read(source/'receipt.json')['status']=='NATIVE_MATERIAL_SURFACE_FEATURES_FROZEN'
    manifest=read(source/'all_predictions_frozen.json');rows=[]
    write(dest/'registered_protocol.json',dict(stage='NATIVE_MATERIAL_ORIGINAL_POINT_ASSEMBLY',source=args.feature_run,
        source_sha256=digest(__file__),STEP_or_label_reads=0,material_embeddings_not_global_material_IDs=True,start_utc=utc()))
    for case in manifest['cases']:
        check_resources();n=case['original_points'];seen=np.zeros(n,np.uint16)
        output=dict(embedding=np.empty((n,16),np.float32),normal=np.empty((n,3),np.float32),
            boundary=np.empty(n,np.float32),material_embedding=np.empty((n,16),np.float32))
        for entry in case['blocks']:
            path=source/entry['file'];assert digest(path)==entry['sha256']
            with np.load(path) as z:
                ids=z['original_point_ids'];assert len(ids)==entry['core_points'] and np.all((ids>=0)&(ids<n))
                np.add.at(seen,ids,1)
                for field,value in output.items():
                    values=z[field];assert np.isfinite(values).all();value[ids]=values
        assert np.all(seen==1)
        output['core_coverage']=seen.astype(np.uint8);key=case['source_key'];file=dest/f'source_{key}_features.npz';np.savez(file,**output)
        row=dict(source_key=key,role=case['role'],original_points=n,input_file=case['input_file'],input_sha256=case['input_sha256'],
            feature_file=file.name,feature_sha256=digest(file),all_original_points_covered_once=True,
            material_embedding_context='per native PTv3 block; global owner partition not established')
        write(dest/f'source_{key}_freeze.json',row);rows.append(row);print('NATIVE_MATERIAL_ASSEMBLED',key,n,flush=True)
        del output,seen;gc.collect()
    write(dest/'all_predictions_frozen.json',dict(sources=rows,reference_reads=0))
    write(dest/'receipt.json',dict(status='NATIVE_MATERIAL_POINT_FEATURES_READY',sources=rows,original_eight_G=496,
        weld_accuracy_NOT_MEASURED=True,full_contract_pass=False,end_utc=utc()))

if __name__=='__main__':main()
