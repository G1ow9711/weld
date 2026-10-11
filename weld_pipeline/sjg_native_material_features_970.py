"""Free XYZ-only inference for the native finite-face/material backbone."""
import argparse
import builtins
import gc
import io
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,resources,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_material_model_859 import SurfaceMaterialModel
from weld_pipeline.sjg_ptv3_801 import voxelize


def guard():
    if resources()['available_physical_bytes']<1610612736:
        reclaim_own_idle_pages();deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            try:return check_resources()
            except RuntimeError:time.sleep(1)
    return check_resources()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--training-run',default='run162');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);training=OUT/args.training_run
    receipt=read(training/'receipt.json');assert receipt['status']=='NATIVE_PTV3_STRUCTURAL_TRAINING_COMPLETE'
    checkpoint=training/'last.pt';assert digest(checkpoint)==receipt['checkpoint_sha256']
    manifest=read(OUT/'run101/block_manifest.json');assert len(manifest['cases'])==9
    protocol=dict(stage='PURE_XYZ_NATIVE_MATERIAL_SURFACE_FEATURES',training_run=args.training_run,
        checkpoint_sha256=digest(checkpoint),block_manifest_sha256=digest(OUT/'run101/block_manifest.json'),
        source_sha256=digest(__file__),all_nine_before_labels=True,original_eight_G=496,supplemental_BM_G=4,
        material_embeddings_are_predictions_not_reference_IDs=True,no_global_material_partition_claimed=True,
        overlap_context_embeddings_saved=True,overlap_context_dtype='float32',
        python_teacher_or_STEP_reads=0,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol);torch.set_num_threads(2);rows=[]
    try:
        model=SurfaceMaterialModel()
        with checkpoint.open('rb') as stream:state=torch.load(stream,map_location='cpu')
        model.load_state_dict(state['model']);del state;gc.collect();model=model.cuda().eval()
        old_open,old_io=builtins.open,io.open
        def protected(factory):
            def call(path,*a,**kw):
                if isinstance(path,(str,bytes,Path)):
                    value=str(path).lower()
                    if value.endswith(('.stp','.step','.xlsx','.ply')) or any(s in value for s in ('target','teacher','geometry.json','supervision')):
                        raise RuntimeError('Forbidden reference read during free native inference: '+str(path))
                return factory(path,*a,**kw)
            return call
        try:
            builtins.open=protected(old_open);io.open=protected(old_io)
            with torch.inference_mode():
                for case in manifest['cases']:
                    key=case['source_key'];folder=dest/('source_'+key);folder.mkdir();seen=np.zeros(case['original_points'],np.uint8);outputs=[];start=time.perf_counter()
                    for bi,entry in enumerate(case['blocks']):
                        guard();path=OUT/'run101'/entry['file'];assert digest(path)==entry['sha256']
                        with np.load(path) as z:xyz=z['xyz_mm'];ids=z['original_point_ids'];count=int(z['core_count'])
                        assert count==entry['core_count'];block=voxelize(xyz);embedding,normal,boundary,material=model(block)
                        result=dict(original_point_ids=ids[:count],embedding=embedding[:count].cpu().numpy(),normal=normal[:count].cpu().numpy(),
                            boundary=boundary[:count].sigmoid().cpu().numpy(),material_embedding=material[:count].cpu().numpy(),
                            context_original_point_ids=ids[count:],context_embedding=embedding[count:].cpu().numpy(),
                            context_material_embedding=material[count:].cpu().numpy())
                        assert all(np.isfinite(v).all() for v in result.values()) and np.all((ids[:count]>=0)&(ids[:count]<len(seen)))
                        np.add.at(seen,ids[:count],1);file=folder/f'features_{bi:04d}.npz';np.savez(file,**result)
                        outputs.append(dict(file=str(file.relative_to(dest)),sha256=digest(file),core_points=count,input_block_sha256=entry['sha256']))
                        if bi%20==0 or bi+1==len(case['blocks']):
                            write(dest/'progress.json',dict(completed_cases=len(rows),source=key,block=bi+1,blocks=len(case['blocks']),time_utc=utc()))
                            print('FREE_NATIVE_MATERIAL',key,'BLOCK',bi+1,'/',len(case['blocks']),flush=True)
                        del xyz,ids,block,embedding,normal,boundary,material,result
                    assert np.all(seen==1)
                    row=dict(source_key=key,role=case['role'],original_points=case['original_points'],input_sha256=case['input_sha256'],
                        input_file=case['input_file'],blocks=outputs,original_core_coverage_once=True,elapsed_seconds=time.perf_counter()-start)
                    write(folder/'feature_manifest.json',row);rows.append(row);del seen;gc.collect();torch.cuda.empty_cache()
        finally:builtins.open=old_open;io.open=old_io
        write(dest/'all_predictions_frozen.json',dict(cases=rows,python_label_and_STEP_reads=0,native_file_access_not_instrumented=True))
        write(dest/'receipt.json',dict(status='NATIVE_MATERIAL_SURFACE_FEATURES_FROZEN',cases=len(rows),points=sum(r['original_points'] for r in rows),
            new_weld_accuracy_not_measured=True,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_cases=len(rows),error=traceback.format_exc(),full_contract_pass=False,end_utc=utc()));raise

if __name__=='__main__':main()
