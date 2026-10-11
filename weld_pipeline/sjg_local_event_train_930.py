"""Fixed-budget learning of local finite-face graph and line/arc event heads."""
import argparse
import gc
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_local_curve_model_917 import LocalCurveEventModel


def as_tensors(path):
    with np.load(path,allow_pickle=False) as z:return {k:torch.from_numpy(z[k].copy()) for k in z.files}


def save_checkpoint(value,path):
    # Torch 1.13's C++ zip writer fails on this native Windows Unicode path.
    # A Python-owned file handle preserves the actual project-local path.
    from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
    atomic_binary_write(path, lambda stream: torch.save(value, stream))


def loss_for(logits,offset,labels):
    known=labels['target_mask'].to(logits.dtype)
    weights=logits.new_tensor([1.,8.,8.,1.,1.])
    binary=F.binary_cross_entropy_with_logits(logits,labels['target'],reduction='none',pos_weight=weights)
    events=(binary*known).sum()/known.sum().clamp_min(1.)
    offsets=labels['offset_mask'].to(logits.dtype)
    regression=(F.smooth_l1_loss(offset,labels['offset'],reduction='none')*offsets).sum()/offsets.sum().clamp_min(1.)
    return events+regression,events,regression


def material_loss(logits,labels):
    known=labels['material_relation_known']
    if not known.any():return logits.sum()*0
    target=labels['different_material']
    if logits.ndim==2:target=torch.stack([labels['same_material_possible'],target],dim=-1)
    return F.binary_cross_entropy_with_logits(logits[known],target[known])


def controls():
    # Exercise both known and unknown direction/offset masks. Unknown labels
    # contribute neither gradient nor a fabricated zero-valued target.
    a=torch.zeros((3,5),requires_grad=True);b=torch.zeros((3,2),requires_grad=True)
    labels=dict(target=torch.zeros(3,5),target_mask=torch.ones(3,5,dtype=torch.bool),
                offset=torch.ones(3,2),offset_mask=torch.zeros(3,2,dtype=torch.bool))
    labels['target_mask'][:,4]=False
    value,_,regression=loss_for(a,b,labels);value.backward()
    assert torch.isfinite(value) and regression==0 and not torch.any(a.grad[:,4]) and not torch.any(b.grad)
    relation=torch.zeros(3,requires_grad=True)
    auxiliary=dict(material_relation_known=torch.tensor([True,False,True]),different_material=torch.tensor([0.,1.,1.]))
    material_loss(relation,auxiliary).backward()
    assert relation.grad[0]>0 and relation.grad[1]==0 and relation.grad[2]<0
    return dict(unknown_direction_excluded=True,unknown_offsets_excluded=True,zero_known_offset_batch_finite=True,
        ambiguous_material_no_gradient=True,same_and_different_material_gradients_opposite=True)


def main(default_data_run='run116',default_model_family='local_curve_event_917'):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--data-run',default=default_data_run);ap.add_argument('--relation-run')
    ap.add_argument('--model-family',choices=['local_curve_event_917','surface_material_joint_985'],default=default_model_family)
    args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    data=OUT/args.data_run;epochs=8;batch=256;seed=930;torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    protocol=dict(stage='LOCAL_LINE_ARC_GRAPH_EVENT_TRAINING',source_sha256=digest(__file__),model_sha256=digest(ROOT/'weld_pipeline/sjg_local_curve_model_917.py'),
        data_run=args.data_run,epochs=epochs,batch_queries=batch,seed=seed,optimizer='AdamW',learning_rate=.001,weight_decay=.0001,
        event_positive_weights=[1,8,8,1,1],offset_loss='smooth_L1_on_known_offsets',device='cpu',
        feature_backbone='PTv3_run26_frozen_local_XYZ_run106',trainable='finite_face_GNN_and_typed_curve_event_heads',
        original_eight_G=496,training_G=363,development_not_used_in_training=['64','24'],
        final_epoch_only_no_development_selection=True,free_input_inference_required_after_training=True,
        parent_graph_and_E16_not_validated_by_event_loss=True,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol);updates=0;history=[]
    try:
        assert read(data/'receipt.json')['status']=='LOCAL_TYPED_GRAPH_TRAINING_DATA_READY'
        manifest=read(data/'training_manifest.json');graphs={r['source_key']:r for r in read(data/'all_graphs_frozen_before_labels.json')['cases']}
        assert sum(r['original_G'] for r in manifest['cases'])==363
        covered=sum(r['covered_G'] for r in manifest['cases'])
        typed={}
        for kind in ('line','arc'):
            total=sum(r['native_kind_counts'].get(kind,0) for r in manifest['cases'])
            represented=0
            for row in manifest['cases']:
                if 'covered_kind_counts' in row:represented+=row['covered_kind_counts'].get(kind,0)
                else:
                    supervision=read(data/('source_'+row['source_key']+'_supervision.json'))
                    represented+=sum(c['kind']==kind for c in supervision['children'])
            typed[kind]=dict(G=total,covered=represented,eligible=not total or represented/total>.95)
        eligibility=dict(geometrically_supported=covered,original_G=363,fraction=covered/363,
            prerequisite_fraction_strictly_greater_than=.95,by_curve_kind=typed,
            eligible=covered/363>.95 and all(v['eligible'] for v in typed.values()),
            reason='A fixed-carrier decoder cannot recover missing whole children by updating event weights.')
        write(dest/'representation_eligibility.json',eligibility)
        if not eligibility['eligible']:
            raise RuntimeError('Insufficient whole-child geometric support for this fixed-carrier model: '+str(covered)+'/363')
        relation_rows={};model_family=args.model_family
        data_protocol=read(data/'registered_protocol.json')
        protocol['feature_backbone']='PTv3_frozen_features_'+data_protocol['features']
        if model_family=='surface_material_joint_985':
            assert not args.relation_run,'Explicit material graph and legacy pair auxiliary cannot be silently mixed'
            assert data_protocol.get('material_instances_run')
            protocol.update(model_sha256=digest(ROOT/'weld_pipeline/sjg_surface_material_joint_model_985.py'),
                trainable='finite_face_material_membership_GNN_and_typed_curve_events',
                material_instances_run=data_protocol['material_instances_run'])
        if args.relation_run:
            reference=OUT/args.relation_run;aux=read(reference/'manifest.json')
            assert aux['data_manifest_sha256']==digest(data/'training_manifest.json')
            relation_rows={r['source_key']:r for r in aux['rows']}
            assert sum(r['same'] for r in aux['rows'])>0 and sum(r['different'] for r in aux['rows'])>0
            model_family='material_curve_event_953'
            protocol.update(relation_run=args.relation_run,material_auxiliary_loss_weight=.5,
                model_sha256=digest(ROOT/'weld_pipeline/sjg_material_curve_model_953.py'),
                relation_manifest_sha256=digest(reference/'manifest.json'),
                trainable='finite_face_GNN_local_material_relation_and_typed_curve_events')
        protocol['model_family']=model_family
        datasets=[]
        for row in manifest['cases']:
            key=row['source_key'];g=graphs[key]
            for file,sha in ((g['file'],g['sha256']),(row['input_file'],row['input_sha256']),(row['label_file'],row['label_sha256'])):
                assert digest(data/file)==sha
            labels=as_tensors(data/row['label_file'])
            if relation_rows:
                binding=relation_rows[key];assert binding['query_input_sha256']==row['input_sha256']
                assert digest(reference/binding['file'])==binding['sha256'];labels.update(as_tensors(reference/binding['file']))
            datasets.append(dict(key=key,graph=as_tensors(data/g['file']),inputs=as_tensors(data/row['input_file']),labels=labels))
        write(dest/'controls.json',controls());check_resources()
        if model_family=='surface_material_joint_985':
            from weld_pipeline.sjg_surface_material_joint_model_985 import SurfaceMaterialJointModel,controls as graph_controls
            for sample in datasets:
                assert {'material_x','membership_face','membership_material','membership_weight'}<=set(sample['graph'])
            write(dest/'material_graph_controls.json',graph_controls())
            model=SurfaceMaterialJointModel()
        elif relation_rows:
            from weld_pipeline.sjg_material_curve_model_953 import MaterialCurveEventModel
            model=MaterialCurveEventModel()
        else:model=LocalCurveEventModel()
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        save_checkpoint(dict(state_dict=initial,seed=seed,updates=0,model_family=model_family),dest/'initial.pt')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        expected=epochs*sum(int(np.ceil(len(d['inputs']['nodes'])/batch)) for d in datasets)
        protocol.update(expected_updates=expected,train_sources=[d['key'] for d in datasets],parameter_count=sum(p.numel() for p in model.parameters()))
        write(dest/'registered_protocol.json',protocol);start=time.perf_counter()
        for epoch in range(1,epochs+1):
            model.train();losses=[]
            for di in rng.permutation(len(datasets)):
                sample=datasets[int(di)];order=rng.permutation(len(sample['inputs']['nodes']));source_losses=[]
                relation_losses=[];relation_correct=relation_known=0
                for begin in range(0,len(order),batch):
                    ids=torch.as_tensor(order[begin:begin+batch],dtype=torch.long)
                    inputs=dict(sample['graph'],**{k:v[ids] for k,v in sample['inputs'].items() if k!='q_mm'})
                    labels={k:v[ids] for k,v in sample['labels'].items()}
                    optimizer.zero_grad(set_to_none=True);outputs=model(inputs);logits,offset=outputs[:2];loss,event_loss,offset_loss=loss_for(logits,offset,labels)
                    if relation_rows:
                        auxiliary=material_loss(outputs[2],labels);loss=loss+.5*auxiliary
                        relation_losses.append(float(auxiliary.detach()));mask=labels['material_relation_known']
                        relation_target=torch.stack([labels['same_material_possible'],labels['different_material']],dim=-1)
                        relation_known+=2*int(mask.sum())
                        relation_correct+=int(((outputs[2][mask]>=0)==(relation_target[mask]>=.5)).sum())
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    optimizer.step();updates+=1;source_losses.append(float(loss.detach()));losses.append(float(loss.detach()))
                row=dict(epoch=epoch,source=sample['key'],updates=updates,mean_loss=float(np.mean(source_losses)),elapsed_seconds=time.perf_counter()-start,
                    mean_material_loss=float(np.mean(relation_losses)) if relation_losses else None,
                    online_training_material_relation_accuracy=relation_correct/relation_known if relation_known else None,
                    material_known_bit_count=relation_known,training_statistic_not_validation=True)
                write(dest/'progress.json',dict(**row,total_epochs=epochs,expected_updates=expected))
                print('EPOCH',epoch,'/',epochs,'SOURCE',sample['key'],'UPDATES',updates,'/',expected,'LOSS',round(row['mean_loss'],6),flush=True)
                del inputs,labels,logits,offset,outputs,loss,event_loss,offset_loss;gc.collect();check_resources()
            history.append(dict(epoch=epoch,updates=updates,mean_loss=float(np.mean(losses))));write(dest/'history.json',history)
            # Recovery state is separate from final.pt: an interrupted epoch
            # cannot be mistaken for the fixed-budget completed model.
            checkpoint=dest/'epoch_checkpoint.pending.pt'
            save_checkpoint(dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),updates=updates,model_family=model_family,
                            completed_epochs=epoch,numpy_rng_state=rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),
                            data_manifest_sha256=digest(data/'training_manifest.json')),checkpoint)
            checkpoint.replace(dest/'epoch_checkpoint.pt')
        assert updates==expected
        delta=sum(float((value-initial[key]).double().square().sum()) for key,value in model.state_dict().items())**.5
        assert delta>0 and all(torch.isfinite(p).all() for p in model.parameters())
        save_checkpoint(dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),updates=updates,epochs=epochs,model_family=model_family,
                        data_manifest_sha256=digest(data/'training_manifest.json')),dest/'final.pt')
        write(dest/'receipt.json',dict(status='LOCAL_TYPED_EVENT_TRAINING_COMPLETE',epochs=epochs,optimizer_updates=updates,
            parameter_change_l2=delta,initial_sha256=digest(dest/'initial.pt'),final_sha256=digest(dest/'final.pt'),
            training_seconds=time.perf_counter()-start,history=history,new_extraction_accuracy_not_yet_measured=True,
            full_contract_pass=False,end_utc=utc()))
        print('TRAINING_COMPLETE','EPOCHS',epochs,'UPDATES',updates,'FREE_PREDICTION_PENDING',flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',optimizer_updates=updates,history=history,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
