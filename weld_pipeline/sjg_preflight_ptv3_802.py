"""Real-cloud PTv3 runtime controls. No task-training or accuracy claim."""
import argparse
import gc
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_ptv3_801 import CONFIG,build_backbone,xyz_block,cuda_input


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(802)
    protocol=dict(stage='PTV3_REAL_XYZ_RUNTIME_PREFLIGHT',start_utc=utc(),config=CONFIG,
        sources=[21,4],raw_crop_points=[4096,8192],task_training_updates=0,
        technical_optimizer_updates_planned=2,STEP_or_weld_labels_read=False,
        imported_dependencies=dict(torch=torch.__version__),contract_accuracy_not_tested=True,
        preflight_source_sha256=digest(Path(__file__)),
        adapter_source_sha256=digest(ROOT/'weld_pipeline/sjg_ptv3_801.py'),
        backbone_source_sha256=digest(ROOT/'training/third_party/ptv3_801/model.py'))
    write(dest/'registered_protocol.json',protocol)
    results=[];updates=0
    try:
        inventory=read(OUT/'inventory.json');check_resources()
        model=build_backbone().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        print('PTV3_REAL_CLOUD_PREFLIGHT parameters',sum(p.numel() for p in model.parameters()),'not a training epoch',flush=True)
        for source,count in zip(protocol['sources'],protocol['raw_crop_points']):
            check_resources();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            raw=ROOT/inventory[source]['raw']['original_pcd_path']
            xyz=read_xyz_pcd(raw)*1000.;ids,block=xyz_block(xyz,count)
            np.savez_compressed(dest/f'source_{source:02d}_mapping.npz',point_ids=ids,
                representative=block['representative'],inverse=block['inverse'])
            model.train();optimizer.zero_grad(set_to_none=True)
            output=model(cuda_input(block)).feat
            assert output.shape==(len(block['coord']),32) and torch.isfinite(output).all()
            # Synthetic numerical objective, explicitly not a task-trained model.
            loss=output.square().mean();loss.backward()
            gradient_names=[n for n,p in model.named_parameters() if p.grad is not None and p.grad.abs().sum()>0]
            assert any('attn.qkv' in n for n in gradient_names)
            assert any('embedding' in n for n in gradient_names)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            name,param=next((n,p) for n,p in model.named_parameters() if 'attn.qkv' in n)
            before=param.detach().clone();optimizer.step();updates+=1
            delta=float((param-before).abs().max());assert delta>0
            model.eval()
            with torch.no_grad():
                baseline=model(cuda_input(block)).feat
                expanded=baseline[torch.from_numpy(block['inverse']).cuda()]
                assert len(expanded)==len(ids)
                # Serialization must recover the input point order.
                perm=np.random.default_rng(802).permutation(len(block['coord'])).astype(np.int64)
                shuffled={**block,**{k:block[k][perm] for k in ('coord','grid_coord','feat')}}
                permuted=model(cuda_input(shuffled)).feat
                permutation_error=float((permuted-baseline[torch.from_numpy(perm).cuda()]).abs().max())
                assert permutation_error<1e-3,permutation_error
                saved=dest/'technical_checkpoint.pt'
                with saved.open('wb') as handle:torch.save({'model':model.state_dict(),'config':CONFIG},handle)
                with saved.open('rb') as handle:state=torch.load(handle,map_location='cuda')
                model.load_state_dict(state['model'])
                reloaded=model(cuda_input(block)).feat
                reload_error=float((baseline-reloaded).abs().max());assert reload_error==0
            torch.cuda.synchronize()
            result=dict(source=source,raw_sha256=digest(raw),original_points=len(xyz),crop_points=len(ids),
                input_voxels=len(block['coord']),output_channels=32,gradient_parameter_tensors=len(gradient_names),
                qkv_parameter_update_max=delta,loss=float(loss),permutation_max_error=permutation_error,
                reload_max_error=reload_error,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                elapsed_seconds=time.perf_counter()-start,memory=check_resources())
            results.append(result);write(dest/'progress.json',results)
            print('PTV3_RUNTIME_CASE_PASS',result,flush=True)
            del output,baseline,reloaded,permuted,expanded,state,xyz;gc.collect();torch.cuda.empty_cache()
        write(dest/'receipt.json',dict(status='PTV3_REAL_CLOUD_RUNTIME_PASS',end_utc=utc(),results=results,
            technical_optimizer_updates=updates,task_training_updates=0,contract_pass=False,
            checkpoint_for_runtime_test_only=True,model_sha256=digest(dest/'technical_checkpoint.pt')))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),results=results,
            technical_optimizer_updates=updates,task_training_updates=0,error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
