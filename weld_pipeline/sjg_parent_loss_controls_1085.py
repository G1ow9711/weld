"""Numerical/backprop controls for the actual parent objective; zero updates."""
import argparse
import gc
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_parent_set_loss_1084 import ParentTargets,ParentSetObjective,normalization_controls
from weld_pipeline.sjg_parent_set_network_1075 import ParentSetModel
from weld_pipeline.sjg_parent_input_1078 import ParentInput
from weld_pipeline.sjg_parent_anchor_bank_1080 import CarrierBank
from weld_pipeline.sjg_parent_set_runtime_1081 import child_geometry
from weld_pipeline.sjg_mls_network_control_1079 import parameter_hash
from weld_pipeline.sjg_cuda_startup_pages_1040 import startup_pages


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run265')
    ap.add_argument('--geometry-only',action='store_true');ap.add_argument('--geometry-run')
    args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);start=time.perf_counter()
    try:
        torch.set_num_threads(2);torch.manual_seed(1031);reclaim_own_idle_pages();check_resources()
        controls=normalization_controls();geometry=[]
        assert read(OUT/'run264/receipt.json')['status']=='PARENT_EXPRESSION_CONTROL_PASSED'
        if args.geometry_run:
            prior=read(OUT/args.geometry_run/'receipt.json')
            assert prior['status']=='PARENT_LOSS_GEOMETRY_CONTROLS_PASSED'
            assert prior['loss_source_sha256']==digest(__import__('pathlib').Path(__file__).with_name('sjg_parent_set_loss_1084.py'))
            geometry=prior['geometry']
        for key in (() if args.geometry_run else ('21','BM79A_native_local','10')):
            bank=CarrierBank(key);compiled=read(OUT/f'run264/source_{key}_targets.json')
            targets=ParentTargets(compiled,bank,'cpu');maximum=0.;alternatives=0
            for rows in targets.children:
                for row in rows:
                    for ai,a in enumerate(row['alternatives']):
                        selected={k:(v[[ai]] if torch.is_tensor(v) and k!='target' else v) for k,v in row.items()}
                        actual=targets.control_points(selected,selected['fractions']).squeeze(0).squeeze(0).numpy()
                        expected,_=child_geometry(bank.get(a['profile'])['curve'],a['lo'],a['hi'],a['forward'],a['closed'],a['wrap'])
                        maximum=max(maximum,float(np.max(np.abs(actual-expected))));alternatives+=1
            assert maximum<.001,(key,maximum)
            geometry.append(dict(source=key,alternatives=alternatives,control_points_max_error_mm=maximum))
            print('LOSS_GEOMETRY_CONTROL',key,alternatives,maximum,flush=True)
            del bank,compiled,targets;gc.collect();reclaim_own_idle_pages()
        if args.geometry_only:
            write(dest/'receipt.json',dict(status='PARENT_LOSS_GEOMETRY_CONTROLS_PASSED',geometry=geometry,
                loss_source_sha256=digest(__import__('pathlib').Path(__file__).with_name('sjg_parent_set_loss_1084.py')),
                normalization_controls=controls,optimizer_updates=0,utc=utc()))
            return
        warmup=startup_pages();device='cuda';model=ParentSetModel().to(device)
        before=parameter_hash(model)
        with np.load(OUT/'run260/input_normalization.npz') as z:norm={k:z[k] for k in z.files}
        data=ParentInput('21',norm);c,b=data.arrays(slice(None))
        bank=CarrierBank('21');targets=ParentTargets(read(OUT/'run264/source_21_targets.json'),bank,device)
        result=model(torch.as_tensor(c,device=device),torch.as_tensor(b,device=device))
        loss,diagnostics=ParentSetObjective(model,targets)(result)
        assert torch.isfinite(loss)
        loss.backward();torch.cuda.synchronize()
        grads={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
        assert all(np.isfinite(x) for x in grads.values())
        assert grads['bin_encoder.weight']>0 and grads['exists.weight']>0
        assert parameter_hash(model)==before
        receipt=dict(status='PARENT_SET_LOSS_CONTROLS_PASSED',normalization_controls=controls,geometry=geometry,
            actual_input_source='21',complete_profiles=len(data),loss=float(loss),diagnostics=diagnostics,
            gradients=grads,parameter_hash_unchanged=before,optimizer_updates=0,
            extraction_correct_rate=None,extraction_precision_mm=None,
            seconds=time.perf_counter()-start,utc=utc(),warmup=warmup)
        write(dest/'receipt.json',receipt);print('PARENT_LOSS_CONTROLS_PASSED',float(loss),diagnostics,flush=True)
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),optimizer_updates=0,utc=utc()));raise


if __name__=='__main__':main()
