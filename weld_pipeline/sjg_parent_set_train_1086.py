"""Registered 21/BM/10 parent-set pilot: 30 updates, 180 seconds, final only."""
import argparse
import gc
import random
import shutil
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,resources,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_parent_set_network_1075 import ParentSetModel
from weld_pipeline.sjg_parent_input_1078 import ParentInput
from weld_pipeline.sjg_parent_anchor_bank_1080 import CarrierBank
from weld_pipeline.sjg_parent_set_loss_1084 import ParentTargets,ParentSetObjective
from weld_pipeline.sjg_parent_resource_start_1090 import start_cuda
from weld_pipeline.sjg_local_event_train_930 import save_checkpoint
from weld_pipeline.sjg_mls_network_control_1079 import parameter_hash
from weld_pipeline.sjg_parent_stream_encoder_1087 import stream_encode,controls as stream_controls

KEYS=('21','BM79A_native_local','10')


def load_complete_input(key, normalization, device='cuda'):
    data=ParentInput(key,normalization)
    binding=data.binding()
    gc.collect();reclaim_own_idle_pages();check_resources()
    return data,binding


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run266');ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--resume')
    args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    updates=0;history=[];started=None;model=None;optimizer=None;prior_seconds=0.;resume_checkpoint=None
    protocol=dict(stage='COMPLETE_PARENT_SET_WITH_MLS',sources=list(KEYS),TRAIN_G=213,full_pilot_G=500,
        epochs=10,updates=30,budget_seconds=180,seed=1031,AdamW_lr=.0003,weight_decay=.0001,clip_norm=5.,
        parent_queries=256,children_per_parent=5,width=64,bin_channels=242,
        existence_threshold=.5,profile_pruning=False,GT_count_inference=False,
        STEP_runtime=False,A_used_as_input=False,upstream_frozen=True,final_checkpoint_only=True,
        test_order=['21','BM79A_native_local','10','64'],stop_after_first_failed_gate=True,
        controls_are_not_scientific_training=True,preflight_only=args.preflight_only,full_contract_pass=False,utc=utc())
    protocol.update(object_encoding='all objects, exact gradient block recomputation, chunk 1024',
        child_cost='profile NLL + masked interval NLL + masked fraction L1 + ordered authoritative control-point error / 3mm + type/direction/closure/wrap/known-connection NLL',
        parent_cost='existence + child-count + parent-closure + mean child/complete-witness transition costs',
        normalization='positive parent mean + unmatched query NO_OBJECT mean; matching subtracts displaced NO_OBJECT cost')
    write(dest/'registered_protocol.json',protocol);(dest/'source_snapshot').mkdir()
    for n in (904,907,1075,1077,1078,1080,1081,1082,1084,1086,1087,1090):
        for p in (ROOT/'weld_pipeline').glob(f'*_{n}.py'):shutil.copy2(p,dest/'source_snapshot'/p.name)
    try:
        if args.resume:
            previous=OUT/args.resume;failure=read(previous/'failure.json')
            previous_protocol=read(previous/'registered_protocol.json')
            for field in ('sources','TRAIN_G','full_pilot_G','epochs','updates','budget_seconds','seed','AdamW_lr','weight_decay','clip_norm','parent_queries','children_per_parent','width','bin_channels'):
                assert previous_protocol[field]==protocol[field]
            if not failure['error'].splitlines()[-1].startswith('RuntimeError: Resource guard:'):
                raise RuntimeError('Automatic resume allowed only for the verified resource interruption')
            with (previous/'interrupted.pt').open('rb') as f:resume_checkpoint=torch.load(f,map_location='cpu')
            updates=int(resume_checkpoint['updates']);history=read(previous/'history.json')
            assert updates==failure['actual_updates']==len(history) and 0<updates<30
            prior_seconds=float(failure['training_seconds'])
            write(dest/'resume_binding.json',dict(previous_run=args.resume,checkpoint_sha256=digest(previous/'interrupted.pt'),
                preserved_updates=updates,prior_training_seconds=prior_seconds,remaining_updates=30-updates,
                remaining_training_budget_seconds=180-prior_seconds,
                objective_and_optimizer_unchanged=True,change='own read-only mapped-page reclamation only',
                old_checkpoint_rng_unavailable=True,network_has_no_stochastic_training_layers=True))
        assert read(OUT/'run264/receipt.json')['status']=='PARENT_EXPRESSION_CONTROL_PASSED'
        assert read(OUT/'run263/receipt.json')['status']=='COMPLETE_PARENT_SET_DECODER_CONTROLS_PASSED'
        assert read(OUT/'run268/receipt.json')['status']=='PARENT_SET_LOSS_CONTROLS_PASSED'
        torch.set_num_threads(2);torch.manual_seed(1031);np.random.seed(1031);random.seed(1031)
        write(dest/'stream_controls.json',stream_controls())
        torch.manual_seed(1031)
        reclaim_own_idle_pages();write(dest/'startup_pages.json',start_cuda())
        freeze=read(OUT/'run260/input_freeze.json')
        assert digest(OUT/'run260/input_normalization.npz')==freeze['normalization_sha256']
        shutil.copy2(OUT/'run260/input_normalization.npz',dest/'normalization.npz')
        with np.load(dest/'normalization.npz') as z:normalization={k:z[k] for k in z.files}
        inputs={};targets={};bindings=[];preparation=time.perf_counter()
        for key in KEYS:
            print('PREPARING_COMPLETE_INPUT',key,'epoch not started',flush=True)
            data,binding=load_complete_input(key,normalization)
            inputs[key]=data;bindings.append(binding)
            compiled=read(OUT/f'run264/source_{key}_targets.json')
            row=next(r for r in read(OUT/'run264/receipt.json')['cases'] if r['source']==key)
            assert digest(OUT/f'run264/source_{key}_targets.json')==row['compiled_sha256']
            bank=CarrierBank(key);targets[key]=ParentTargets(compiled,bank,'cuda')
            del bank;gc.collect();reclaim_own_idle_pages();check_resources()
            print('COMPLETE_INPUT_READY',key,'profiles',len(data),'G',compiled['G'],flush=True)
        write(dest/'input_bindings.json',dict(cases=bindings,preparation_seconds=time.perf_counter()-preparation,
            normalization_sha256=digest(dest/'normalization.npz'),freeze_sha256=digest(OUT/'run260/input_freeze.json')))
        model=ParentSetModel().cuda()
        if resume_checkpoint is not None:model.load_state_dict(resume_checkpoint['state_dict'])
        initial_hash=parameter_hash(model)
        objective={k:ParentSetObjective(model,targets[k]) for k in KEYS}
        controls=[]
        # Warm actual kernels and verify all three complete-input gradients, with
        # no optimizer and no state changes. These timings are disclosed.
        for key in KEYS:
            tick=time.perf_counter();print('ZERO_UPDATE_PREFLIGHT',key,flush=True)
            result=model.forward_encoded(stream_encode(model,inputs[key]));loss,parts=objective[key](result);loss.backward();torch.cuda.synchronize()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            controls.append(dict(source=key,seconds=time.perf_counter()-tick,loss=float(loss),parts=parts))
            model.zero_grad(set_to_none=True);del result,loss;gc.collect();reclaim_own_idle_pages();check_resources()
            print('ZERO_UPDATE_PREFLIGHT_PASSED',key,round(controls[-1]['seconds'],3),flush=True)
        assert parameter_hash(model)==initial_hash
        write(dest/'gradient_controls.json',dict(cases=controls,optimizer_updates=0,unchanged_parameters=initial_hash))
        if args.preflight_only:
            write(dest/'receipt.json',dict(status='PARENT_COMPLETE_PREFLIGHT_PASSED',updates=0,utc=utc()));return
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
        if resume_checkpoint is not None:optimizer.load_state_dict(resume_checkpoint['optimizer_state'])
        resume_updates=updates
        started=time.perf_counter()
        for epoch in range(1,11):
            for key in KEYS:
                if (epoch-1)*len(KEYS)+KEYS.index(key)<resume_updates:continue
                if prior_seconds+time.perf_counter()-started>=180:raise RuntimeError('Registered 180-second training budget exhausted')
                if resources()['available_physical_bytes']<1879048192:reclaim_own_idle_pages()
                check_resources();optimizer.zero_grad(set_to_none=True)
                result=model.forward_encoded(stream_encode(model,inputs[key]));loss,parts=objective[key](result)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                torch.cuda.synchronize()
                if prior_seconds+time.perf_counter()-started>=180:raise RuntimeError('180-second budget exhausted before next optimizer update')
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite training gradient')
                optimizer.step();torch.cuda.synchronize();updates+=1
                row=dict(epoch=epoch,source=key,update=updates,loss=float(loss.detach()),parts=parts,
                    gradient_norm=float(norm),training_seconds=prior_seconds+time.perf_counter()-started)
                history.append(row);write(dest/'progress.json',row)
                print('EPOCH',epoch,'/10 UPDATE',updates,'/30 SOURCE',key,'LOSS',round(row['loss'],5),
                    'SECONDS',round(row['training_seconds'],2),flush=True)
                del result,loss;optimizer.zero_grad(set_to_none=True)
        seconds=prior_seconds+time.perf_counter()-started
        if seconds>180:raise RuntimeError('Training finished outside registered budget')
        save_checkpoint(dict(state_dict=model.state_dict(),optimizer_state=optimizer.state_dict(),updates=updates,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),protocol=protocol,normalization_sha256=digest(dest/'normalization.npz'),
            input_bindings_sha256=digest(dest/'input_bindings.json')),dest/'final.pt')
        write(dest/'history.json',history)
        write(dest/'receipt.json',dict(status='PARENT_SET_TRAIN_COMPLETE',updates=updates,epochs=10,
            training_seconds=seconds,checkpoint_sha256=digest(dest/'final.pt'),initial_parameter_sha256=initial_hash,
            final_parameter_sha256=parameter_hash(model),TRAIN_G=213,full_pilot_G=500,full_contract_pass=False,utc=utc()))
        print('PARENT_SET_TRAIN_COMPLETE',updates,'SECONDS',round(seconds,3),flush=True)
    except BaseException:
        error=traceback.format_exc();write(dest/'history.json',history)
        if model is not None and updates:
            if optimizer is not None:
                save_checkpoint(dict(state_dict=model.state_dict(),optimizer_state=optimizer.state_dict(),updates=updates,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(),protocol=protocol,status='INCOMPLETE_DO_NOT_TREAT_AS_REGISTERED_FINAL'),dest/'interrupted.pt')
        write(dest/'failure.json',dict(error=error,actual_updates=updates,
            training_seconds=prior_seconds if started is None else prior_seconds+time.perf_counter()-started,utc=utc()));raise


if __name__=='__main__':main()
