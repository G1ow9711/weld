"""PTv3 numeric inference, one input block and one output block in RAM."""
import argparse
import gc
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources,resources,reclaim_own_idle_pages
from weld_pipeline.sjg_ptv3_runtime_807 import SurfaceFeatureModel,core_partitions
from weld_pipeline.sjg_ptv3_801 import voxelize

def main(reclaim_working_set=False,wait_for_page_reclaim=False):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    data=OUT/'run101';checkpoint=OUT/'run26/last.pt'
    assert read(data/'receipt.json')['status']=='LOCAL_STREAM_BLOCKS_READY'
    manifest=read(data/'block_manifest.json');records=[]
    write(dest/'registered_protocol.json',dict(stage='STREAM_NATIVE_LOCAL_PTV3_BASELINE',start_utc=utc(),
        source_sha256=digest(__file__),block_manifest_sha256=digest(data/'block_manifest.json'),
        checkpoint_sha256=digest(checkpoint),checkpoint_training_sources=[21,4,10,25,36,42],
        checkpoint_trained_on_old_world_XYZ=True,coordinate_frame='workpiece_local',unit='mm',
        new_optimizer_updates=0,no_weld_accuracy_claim=True,full_contract_pass=False,
        raw_labels_or_STEP_not_used=True,original_eight_G=496,new_training_candidate_G=4,
        whole_case_XYZ_tree_or_output_not_held_in_GPU_worker=True,resource_guard_unchanged=True,
        reclaim_only_own_idle_working_set=reclaim_working_set,
        wait_for_async_page_reclaim_seconds=45 if wait_for_page_reclaim else 0,
        runtime_sha256=digest(ROOT/'weld_pipeline/sjg_local_runtime_904.py')))
    try:
        # The streaming split must preserve the original inference partition.
        from weld_pipeline.sjg_local_blocks_905 import core_partitions as streamed_parts
        sample=np.random.default_rng(906).normal(size=(9001,3))
        assert all(np.array_equal(a,b) for a,b in zip(list(core_partitions(sample)),list(streamed_parts(sample))))
        model=SurfaceFeatureModel()
        with checkpoint.open('rb') as stream:state=torch.load(stream,map_location='cpu')
        model.load_state_dict(state['model'],strict=True);del state,sample;gc.collect()
        model=model.cuda().eval();print('LOCAL_STREAM_MODEL_READY',resources(),flush=True)
        with torch.inference_mode():
            for case in manifest['cases']:
                key=case['source_key'];folder=dest/f'source_{key}';folder.mkdir();outputs=[]
                start=time.perf_counter();torch.cuda.reset_peak_memory_stats();covered=0
                for bi,entry in enumerate(case['blocks']):
                    if reclaim_working_set and resources()['available_physical_bytes']<1610612736:
                        reclamation=reclaim_own_idle_pages()
                        with (dest/'own_working_set_reclaims.jsonl').open('a',encoding='utf-8') as log:
                            import json
                            log.write(json.dumps(dict(source_key=key,block=bi,utc=utc(),**reclamation))+'\n')
                        print('WORKER_IDLE_PAGES_RELEASED','FREE_MIB',round(reclamation['after']['available_physical_bytes']/2**20),flush=True)
                        if wait_for_page_reclaim:
                            # Windows page writeback after EmptyWorkingSet is
                            # asynchronous. Keep the same worker and unchanged
                            # guard while awaiting the actual available budget.
                            deadline=time.monotonic()+45;attempt=0
                            while time.monotonic()<deadline:
                                try:
                                    check_resources();break
                                except RuntimeError:
                                    if attempt%10==0:print('WAITING_OWN_PAGE_RECLAIM',attempt,resources(),flush=True)
                                    time.sleep(1);attempt+=1
                    check_resources();path=data/entry['file'];assert digest(path)==entry['sha256']
                    with np.load(path,allow_pickle=False) as z:
                        xyz=z['xyz_mm'];ids=z['original_point_ids'];count=int(z['core_count'])
                    block=voxelize(xyz);embedding,normal,boundary=model(block)
                    result=dict(original_point_ids=ids[:count],embedding=embedding[:count].cpu().numpy(),
                                normal=normal[:count].cpu().numpy(),boundary=boundary[:count].sigmoid().cpu().numpy())
                    assert count==entry['core_count'] and all(np.isfinite(v).all() for v in result.values())
                    if bi==0 and len(records)==0:
                        repeated=model(block)
                        assert torch.allclose(embedding,repeated[0],atol=1e-6,rtol=1e-5)
                        assert torch.allclose(normal,repeated[1],atol=1e-6,rtol=1e-5)
                        assert torch.allclose(boundary,repeated[2],atol=1e-6,rtol=1e-5)
                        del repeated
                    file=folder/f'features_{bi:04d}.npz';np.savez(file,**result);covered+=count
                    outputs.append(dict(file=str(file.relative_to(dest)),sha256=digest(file),core_points=count,
                                        input_block_sha256=entry['sha256']))
                    del xyz,ids,block,embedding,normal,boundary,result
                    if bi%20==0 or bi+1==len(case['blocks']):
                        snapshot=dict(source_key=key,block=bi+1,total_blocks=len(case['blocks']),
                            original_core_points=covered,resources=resources(),elapsed_seconds=time.perf_counter()-start)
                        write(dest/'progress.json',dict(completed_cases=len(records),total_cases=len(manifest['cases']),**snapshot))
                        print('LOCAL_STREAM_PTV3',key,'BLOCK',bi+1,'/',len(case['blocks']),'CORE_POINTS',covered,
                              'RAM_FREE_MIB',round(snapshot['resources']['available_physical_bytes']/2**20),flush=True)
                assert covered==case['original_points']
                row=dict(source_key=key,role=case['role'],original_points=case['original_points'],
                    input_file=case['input_file'],input_sha256=case['input_sha256'],
                    coordinate_frame='workpiece_local',unit='mm',blocks=outputs,
                    gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                    elapsed_seconds=time.perf_counter()-start)
                records.append(row);write(folder/'feature_manifest.json',row)
                print('LOCAL_STREAM_CASE_FROZEN',key,covered,flush=True);gc.collect();torch.cuda.empty_cache()
        write(dest/'all_predictions_frozen.json',dict(cases=records,label_reads=0,optimizer_updates=0))
        write(dest/'receipt.json',dict(status='LOCAL_STREAM_PTV3_FEATURES_COMPLETE',points=sum(c['original_points'] for c in records),
            cases=len(records),original_G=496,optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_cases=len(records),resources=resources(),error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
