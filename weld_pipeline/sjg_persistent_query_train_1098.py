"""Third and final free-query trial: persistent identities and query interaction."""
import argparse
import gc
import hashlib
import random
import shutil
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,resources,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_parent_set_loss_1084 import ParentTargets,ParentSetObjective
from weld_pipeline.sjg_parent_anchor_bank_1080 import CarrierBank
from weld_pipeline.sjg_parent_stream_encoder_1087 import stream_encode
from weld_pipeline.sjg_parent_resource_start_1090 import start_cuda
from weld_pipeline.sjg_parent_set_runtime_1081 import network_choices
from weld_pipeline.sjg_local_event_train_930 import save_checkpoint
from weld_pipeline.sjg_mls_network_control_1079 import parameter_hash
from weld_pipeline.sjg_roboweldar_pair_support_1093 import PairParentInput
from weld_pipeline.sjg_pair_set_revision_1094 import loss_controls
from weld_pipeline.sjg_persistent_query_network_1097 import warm_start
from weld_pipeline.sjg_parent_query_diagnosis_1092 import cosine_summary

KEYS=('21','BM79A_native_local','10')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run280');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    model=optimizer=None;updates=0;history=[];started=None
    protocol=dict(stage='PERSISTENT_QUERY_IDENTITY_AND_SELF_ATTENTION',sources=list(KEYS),
        TRAIN_G=213,full_pilot_G=500,epochs=10,updates=30,budget_seconds=180,
        seed=1031,AdamW_lr=.0003,weight_decay=.0001,clip_norm=5.,parent_queries=256,
        children_per_parent=5,width=64,bin_channels=247,existence_threshold=.5,
        warm_start='run271/final.pt',optimizer_restored=True,new_input_columns_zero_initialized=True,
        loss='(sum matched complete parent costs + sum unmatched softplus)/max(P,1)',
        matching='(positive cost - displaced background)/max(P,1)',
        pair_support_features=5,all_candidates_retained=True,GT_count_inference=False,STEP_runtime=False,
        upstream_frozen=True,final_checkpoint_only=True,
        test_order=['21','BM79A_native_local','10','64'],stop_after_first_failed_gate=True,
        decoder='persistent_query_1097',persistent_query_position='layer_norm_without_affine(parent_query)',
        query_self_attention_layers=2,heads=4,self_attention_output_projection_initial_zero=True,
        self_attention_new_Adam_state=True,dropout=0.,free_query_trial_number=3,
        failed_third_trial_retires_free_query_decoder=True,
        previous_pair_features_retained=True,same_run271_restart_and_objective_as_run277=True,
        full_contract_pass=False,utc=utc())
    write(dest/'registered_protocol.json',protocol)
    (dest/'source_snapshot').mkdir()
    for n in (1075,1078,1080,1081,1084,1087,1090,1093,1094,1097,1098):
        for p in (ROOT/'weld_pipeline').glob(f'*_{n}.py'):shutil.copy2(p,dest/'source_snapshot'/p.name)
    try:
        pro=read(OUT/'pro_review_run278/receipt.json')
        assert pro['generation_stopped'] and pro['assistant_sentinel_verified']
        # Collector hashes the DOM text before Windows newline translation.
        answer=(OUT/'pro_review_run278/answer.md').read_text(encoding='utf-8')
        assert hashlib.sha256(answer.encode('utf-8')).hexdigest()==pro['answer_sha256']
        write(dest/'consult_binding.json',dict(dom_text_sha256=pro['answer_sha256'],
              saved_file_sha256=digest(OUT/'pro_review_run278/answer.md')))
        write(dest/'loss_controls.json',loss_controls())
        assert read(OUT/'run274/controls.json')['status']=='PAIR_SUPPORT_CONTROLS_PASSED'
        torch.set_num_threads(2);torch.manual_seed(1031);np.random.seed(1031);random.seed(1031)
        reclaim_own_idle_pages();write(dest/'startup_pages.json',start_cuda())
        source=OUT/'run271';previous=read(source/'receipt.json')
        assert previous['status']=='PARENT_SET_TRAIN_COMPLETE' and digest(source/'final.pt')==previous['checkpoint_sha256']
        with (source/'final.pt').open('rb') as f:checkpoint=torch.load(f,map_location='cpu')
        model,optimizer=warm_start(checkpoint,'cuda')
        torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['python_rng'])
        del checkpoint;gc.collect();reclaim_own_idle_pages();check_resources()
        initial_hash=parameter_hash(model)
        assert digest(source/'normalization.npz')==read(OUT/'run260/input_freeze.json')['normalization_sha256']
        shutil.copy2(source/'normalization.npz',dest/'normalization.npz')
        with np.load(dest/'normalization.npz') as z:normalization={k:z[k] for k in z.files}
        inputs={};objectives={};bindings=[];controls=[];prep=time.perf_counter()
        for key in KEYS:
            print('PREPARING_PAIR_INPUT',key,'epoch not started',flush=True)
            data=PairParentInput(key,normalization);inputs[key]=data;bindings.append(data.binding())
            compiled=read(OUT/f'run264/source_{key}_targets.json')
            row=next(r for r in read(OUT/'run264/receipt.json')['cases'] if r['source']==key)
            assert digest(OUT/f'run264/source_{key}_targets.json')==row['compiled_sha256']
            bank=CarrierBank(key);targets=ParentTargets(compiled,bank,'cuda')
            objective=ParentSetObjective(model,targets,common_denominator=True);objectives[key]=objective
            reclaim_own_idle_pages();check_resources()
            tick=time.perf_counter();control=dict(source=key)
            if key=='21':
                model.mechanism_enabled=False
                with torch.no_grad():
                    bypass=model.forward_encoded(stream_encode(model,data))
                    choices=network_choices(model,bypass,bank)
                old=read(OUT/'run272/choices.json')
                assert [c['query'] for c in choices]==[c['query'] for c in old]
                diff=0.
                for a,b in zip(choices,old):
                    assert a['child_count']==b['child_count'] and a['closed']==b['closed']
                    for x,y in zip(a['children'],b['children']):
                        for field in ('profile','intervals','forward','closed','wrap','weld_type','side','connection'):
                            assert x[field]==y[field],field
                        diff=max(diff,float(np.max(np.abs(np.array(x['fractions'])-y['fractions']))))
                assert diff<2e-5
                control.update(bypass_old_seven_queries_unchanged=True,maximum_fraction_difference=diff,
                               bypass_latent_cosine=cosine_summary(bypass['parent_latent']))
                del bypass;model.mechanism_enabled=True;model.capture_mechanism=True
            result=model.forward_encoded(stream_encode(model,data))
            loss,parts=objective(result)
            logits_gradient=torch.autograd.grad(loss,result['existence'],retain_graph=True)[0]
            match=objective.match(result);matched=set(int(v) for v in match['rows'])
            assert all(float(logits_gradient[i])<0 for i in matched)
            assert all(float(logits_gradient[i])>0 for i in range(256) if i not in matched)
            control.update(parts=parts,positive_negative_logit_gradient_signs_pass=True)
            loss.backward();torch.cuda.synchronize()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            pair_grad=model.bin_encoder.weight.grad.reshape(model.width,16,247)[:,:,242:]
            control.update(new_pair_channel_gradient_norm=float(pair_grad.norm()),seconds=time.perf_counter()-tick)
            assert control['new_pair_channel_gradient_norm']>0
            if key=='21':
                position_grad=float(model.captured_position.grad.norm())
                out_grads=[float(layer.out_proj.weight.grad.norm()) for layer in model.query_attention]
                assert position_grad>0 and all(g>0 for g in out_grads)
                control.update(persistent_position_gradient_norm=position_grad,self_attention_output_gradients=out_grads,
                    layer_latent_cosines=[cosine_summary(x) for x in model.captured_layers],
                    complete_attention_distributions=model.captured_attention,
                    old_seven_enabled_gradients=[dict(query=c['query'],matched=c['query'] in matched,
                        gradient=float(logits_gradient[c['query']])) for c in choices],
                    enabled_initial_output_not_required_to_equal_old=True)
                model.capture_mechanism=False
                del model.captured_position,model.captured_layers,model.captured_attention
            controls.append(control)
            optimizer.zero_grad(set_to_none=True);del result,loss,logits_gradient,bank;gc.collect();reclaim_own_idle_pages();check_resources()
            print('PAIR_PREFLIGHT_PASSED',key,'SECONDS',round(control['seconds'],3),flush=True)
        assert parameter_hash(model)==initial_hash
        write(dest/'gradient_controls.json',dict(cases=controls,optimizer_updates=0))
        write(dest/'input_bindings.json',dict(cases=bindings,preparation_seconds=time.perf_counter()-prep))
        save_checkpoint(dict(state_dict=model.state_dict(),optimizer_state=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),stage_updates=0,previous_total_updates=30),dest/'initial.pt')
        started=time.perf_counter()
        for epoch in range(1,11):
            for key in KEYS:
                if time.perf_counter()-started>=180:raise RuntimeError('180-second training budget exhausted')
                if resources()['available_physical_bytes']<1879048192:reclaim_own_idle_pages()
                check_resources();optimizer.zero_grad(set_to_none=True)
                result=model.forward_encoded(stream_encode(model,inputs[key]));loss,parts=objectives[key](result)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);torch.cuda.synchronize()
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                if time.perf_counter()-started>=180:raise RuntimeError('180-second budget exhausted before update')
                optimizer.step();torch.cuda.synchronize();updates+=1
                item=dict(epoch=epoch,source=key,update=updates,loss=float(loss.detach()),parts=parts,
                          gradient_norm=float(norm),training_seconds=time.perf_counter()-started)
                history.append(item);write(dest/'progress.json',item)
                print('EPOCH',epoch,'/10 UPDATE',updates,'/30 SOURCE',key,'LOSS',round(item['loss'],5),
                      'SECONDS',round(item['training_seconds'],2),flush=True)
                del result,loss;optimizer.zero_grad(set_to_none=True)
        seconds=time.perf_counter()-started
        if seconds>180:raise RuntimeError('Training outside registered budget')
        save_checkpoint(dict(state_dict=model.state_dict(),optimizer_state=optimizer.state_dict(),updates=updates,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),protocol=protocol,normalization_sha256=digest(dest/'normalization.npz'),
            input_bindings_sha256=digest(dest/'input_bindings.json')),dest/'final.pt')
        write(dest/'history.json',history)
        write(dest/'receipt.json',dict(status='PARENT_SET_TRAIN_COMPLETE',updates=updates,epochs=10,training_seconds=seconds,
            checkpoint_sha256=digest(dest/'final.pt'),initial_parameter_sha256=initial_hash,
            final_parameter_sha256=parameter_hash(model),warm_start_checkpoint_sha256=previous['checkpoint_sha256'],
            previous_stage_updates=30,cumulative_updates=60,TRAIN_G=213,full_pilot_G=500,full_contract_pass=False,utc=utc()))
        print('PARENT_SET_TRAIN_COMPLETE',updates,'SECONDS',round(seconds,3),flush=True)
    except BaseException:
        error=traceback.format_exc();write(dest/'history.json',history)
        if model is not None and optimizer is not None and updates:
            save_checkpoint(dict(state_dict=model.state_dict(),optimizer_state=optimizer.state_dict(),updates=updates,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),protocol=protocol,status='INCOMPLETE'),dest/'interrupted.pt')
        write(dest/'failure.json',dict(error=error,actual_updates=updates,
            training_seconds=0 if started is None else time.perf_counter()-started,utc=utc()));raise


if __name__=='__main__':main()
