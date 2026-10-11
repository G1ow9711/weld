"""TRAIN-only paired continuation: isolate endpoint positive weighting."""
import argparse
import gc
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest, utc, check_resources
from weld_pipeline.sjg_local_event_train_930 import as_tensors, save_checkpoint, material_loss
from weld_pipeline.sjg_material_curve_model_953 import MaterialCurveEventModel


def endpoint_loss(logits, offset, labels, positive_weight):
    mask=labels['target_mask'].to(logits.dtype)
    weights=logits.new_tensor([1.,positive_weight,positive_weight,1.,1.])
    event=(F.binary_cross_entropy_with_logits(logits,labels['target'],reduction='none',pos_weight=weights)*mask).sum()/mask.sum().clamp_min(1)
    known=labels['offset_mask'].to(logits.dtype)
    regression=(F.smooth_l1_loss(offset,labels['offset'],reduction='none')*known).sum()/known.sum().clamp_min(1)
    return event+regression


def controls():
    from weld_pipeline.sjg_local_event_train_930 import loss_for
    torch.manual_seed(958)
    labels=dict(target=torch.rand(5,5),target_mask=torch.ones(5,5,dtype=torch.bool),offset=torch.rand(5,2),offset_mask=torch.ones(5,2,dtype=torch.bool))
    a=torch.randn(5,5);b=torch.randn(5,2)
    assert torch.allclose(endpoint_loss(a,b,labels,8),loss_for(a,b,labels)[0])
    ids=torch.as_tensor(np.array([0,2],dtype=np.int32),dtype=torch.long)
    assert a[ids].shape==(2,5)
    # Weighted soft BCE optimum differs from the label; this is not a new threshold chosen on outputs.
    y=torch.tensor(.1);p=8*y/(1+7*y);z=torch.logit(p).requires_grad_()
    F.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(8.)).backward()
    assert abs(float(z.grad))<1e-6 and p>.4
    return dict(control_loss_exactly_matches_930=True,soft_BCE_weight_changes_probability_scale=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);epochs=32;batch=256;seed=958
    protocol=dict(stage='PAIRED_ENDPOINT_WEIGHT_TRAIN_CONTINUATION',source_sha256=digest(__file__),
        initial='run140/final.pt',initial_sha256=digest(OUT/'run140/final.pt'),data_run='run133',relation_run='run139',
        arms={'weighted_control':8.,'unit_endpoint_weight':1.},epochs_per_arm=epochs,seed=seed,
        batch_queries=batch,AdamW_lr=.001,weight_decay=.0001,new_optimizer_same_for_both=True,
        same_weights_data_order_architecture_decoder=True,only_arm_difference='endpoint_positive_weight',
        final_epoch_only=True,development_reads=0,original_eight_G=496,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol);write(dest/'controls.json',controls())
    write(dest/'prior_failed_round_local_review.json',dict(failed_round='run144',diagnosis='run147',
        source21_geometry=dict(TP=0,G=1,FP=242),true_endpoint_oracle=dict(TP=1,G=1,FP=0),
        true_occupancy_oracle=dict(TP=0,G=1,FP=125),material_class_balance_audit='run139/class_balance_audit.json',
        reason_for_control='Separate effect of longer training from removal of endpoint probability weighting.',
        material_semantics_and_parent_E16_still_unresolved=True,external_consultation_sent=False))
    manifest=read(OUT/'run133/training_manifest.json');graphs={r['source_key']:r for r in read(OUT/'run133/all_graphs_frozen_before_labels.json')['cases']}
    auxiliary=read(OUT/'run139/manifest.json');assert auxiliary['data_manifest_sha256']==digest(OUT/'run133/training_manifest.json')
    relations={r['source_key']:r for r in auxiliary['rows']};datasets=[]
    for row in manifest['cases']:
        g=graphs[row['source_key']];rel=relations[row['source_key']]
        for file,sha in [(g['file'],g['sha256']),(row['input_file'],row['input_sha256']),(row['label_file'],row['label_sha256'])]:assert digest(OUT/'run133'/file)==sha
        assert rel['query_input_sha256']==row['input_sha256'] and digest(OUT/'run139'/rel['file'])==rel['sha256']
        labels=as_tensors(OUT/'run133'/row['label_file']);labels.update(as_tensors(OUT/'run139'/rel['file']))
        datasets.append(dict(key=row['source_key'],graph=as_tensors(OUT/'run133'/g['file']),inputs=as_tensors(OUT/'run133'/row['input_file']),labels=labels))
    state=torch.load(OUT/'run140/final.pt',map_location='cpu');assert state['data_manifest_sha256']==digest(OUT/'run133/training_manifest.json')
    completed=[]
    try:
        for arm,weight in protocol['arms'].items():
            folder=dest/arm;folder.mkdir();torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=MaterialCurveEventModel();model.load_state_dict(state['state_dict'])
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            expected=epochs*sum(int(np.ceil(len(d['inputs']['nodes'])/batch)) for d in datasets);updates=0;history=[];start=time.perf_counter()
            for epoch in range(1,epochs+1):
                model.train();losses=[]
                for di in rng.permutation(len(datasets)):
                    d=datasets[int(di)];order=rng.permutation(len(d['inputs']['nodes']))
                    for begin in range(0,len(order),batch):
                        ids=torch.as_tensor(order[begin:begin+batch],dtype=torch.long);inputs=dict(d['graph'],**{k:v[ids] for k,v in d['inputs'].items() if k!='q_mm'})
                        labels={k:v[ids] for k,v in d['labels'].items()};optimizer.zero_grad(set_to_none=True)
                        logits,offset,relation=model(inputs);loss=endpoint_loss(logits,offset,labels,weight)+.5*material_loss(relation,labels)
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss.detach()))
                row=dict(arm=arm,epoch=epoch,epochs=epochs,optimizer_updates=updates,expected_updates=expected,mean_loss=float(np.mean(losses)),elapsed_seconds=time.perf_counter()-start)
                history.append(row);write(dest/'progress.json',row);print('EPOCH',arm,epoch,'/',epochs,'UPDATES',updates,'/',expected,'LOSS',round(row['mean_loss'],6),flush=True)
                del inputs,labels,logits,offset,relation,loss;gc.collect();check_resources()
            assert updates==expected
            change=sum(float((v-state['state_dict'][k]).double().square().sum()) for k,v in model.state_dict().items())**.5
            assert change>0
            save_checkpoint(dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),updates=updates,
                epochs=epochs,base_updates=448,model_family='material_curve_event_953',endpoint_positive_weight=weight,
                data_manifest_sha256=digest(OUT/'run133/training_manifest.json')),folder/'final.pt')
            receipt=dict(status='LOCAL_TYPED_EVENT_TRAINING_COMPLETE',arm=arm,epochs=epochs,optimizer_updates=updates,base_updates=448,
                parameter_change_l2=change,final_sha256=digest(folder/'final.pt'),history=history,full_contract_pass=False,end_utc=utc())
            write(folder/'receipt.json',receipt);completed.append(dict(arm=arm,checkpoint_sha256=receipt['final_sha256'],updates=updates))
            del model,optimizer;gc.collect()
        write(dest/'receipt.json',dict(status='ENDPOINT_PAIRED_TRAINING_COMPLETE',completed=completed,free_prediction_pending=True,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed=completed,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
