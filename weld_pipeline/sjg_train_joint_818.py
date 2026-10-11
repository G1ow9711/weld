"""Frozen-PTv3 joint and event head training on six registered small cases."""
import argparse
import gc
import json
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_joint_model_817 import JointEventModel

DATA=OUT/'run35'
EPOCHS=20
BATCH=512
SEED=818


def cuda(a):return torch.as_tensor(a,device='cuda')


def load_case(source,training):
    with np.load(DATA/f'source_{source:02d}_inputs.npz') as z:
        graph={k:cuda(z[k]) for k in z.files}
    y=np.load(DATA/f'source_{source:02d}_labels.npy')
    query=None
    if training:
        with np.load(DATA/f'source_{source:02d}_queries.npz') as z:
            query={k:z[k] for k in z.files if k!='q_mm'}
    return graph,y,query


def encode(model,graph):
    return model.encode_faces(graph['face_x'],graph['neighbors'],graph['relation'])


def score(model,dest,label):
    model.eval();records=[]
    with torch.inference_mode():
        for source in TRAIN+VALID+ARC:
            check_resources();graph,y,query=load_case(source,source in TRAIN);h=encode(model,graph)
            probabilities=[]
            for start in range(0,len(y),4096):
                z=model.encode_joints(h,graph['pairs'][start:start+4096],graph['geometry'][start:start+4096])
                probabilities.append(model.relevance(z).squeeze(-1).sigmoid().cpu().numpy())
            probability=np.concatenate(probabilities);np.save(dest/f'{label}_source_{source:02d}_probability.npy',probability)
            pred=probability>=.5;positive=y>0
            truth=read(DATA/f'source_{source:02d}_supervision.json')
            retained=sum(any(pred[i] for i in c['profiles']) for c in truth['children'])
            row=dict(source=source,original_G=truth['original_G'],geometry_representable=truth['covered_children'],
                retained_child_geometry=int(retained),selected_profiles=int(pred.sum()),
                candidate_tp=int((pred&positive).sum()),candidate_fp=int((pred&~positive).sum()),
                candidate_fn=int((~pred&positive).sum()),candidate_tn=int((~pred&~positive).sum()),
                candidate_metrics_not_contract_accuracy=True)
            if query is not None:
                values=[];event_tp=event_fp=event_fn=0;errors=[]
                for start in range(0,len(query['nodes']),BATCH):
                    a={k:cuda(v[start:start+BATCH]) for k,v in query.items()}
                    ids=a['nodes'];z=model.encode_joints(h,graph['pairs'][ids],graph['geometry'][ids])
                    logits,offset=model.query_events(z,a['patch'],a['query'])
                    values.append(float(F.binary_cross_entropy_with_logits(logits,a['targets']).cpu()))
                    p=logits[:,0].sigmoid()>=.5;t=a['targets'][:,0]>.5
                    event_tp+=int((p&t).sum());event_fp+=int((p&~t).sum());event_fn+=int((~p&t).sum())
                    mask=a['offset_mask']>0
                    errors.extend(((offset-a['offsets']).abs()[mask]*2).cpu().tolist())
                row['fixed_training_queries']=dict(bce=float(np.mean(values)),foreground_tp=event_tp,
                    foreground_fp=event_fp,foreground_fn=event_fn,
                    endpoint_offset_p95_mm=float(np.quantile(errors,.95)) if errors else None,
                    teacher_sampled_not_free_inference=True)
            records.append(row);print('JOINT_EVENT_EVAL',label,row,flush=True)
            del graph,y,query,h;gc.collect();torch.cuda.empty_cache()
    write(dest/f'{label}_diagnostics.json',records);return records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED);updates=0;complete=0
    protocol=dict(stage='PTV3_FACE_GRAPH_JOINT_AND_EVENT_HEAD_TRAINING',train=TRAIN,development=VALID,arc=ARC,
        epochs=EPOCHS,batch_queries=BATCH,queries_per_source=8192,steps_per_epoch=96,
        epoch_definition='Every one of the 49152 frozen TRAIN queries once; joint classification balanced subsampling',
        learning_rate=.0003,PTv3_frozen=True,PTv3_training_checkpoint='run26/last.pt',
        joint_positive_definition='Any 3mm-compatible GT child; duplicate material hypotheses unresolved',
        candidate_classification_threshold=.5,event_threshold=.5,validation_used_for_training_or_selection=False,
        original_G=496,unimplemented=['curves','material identity','parent path assembly','complete E16'],
        no_full_dataset_expansion=True,no_external_consultation=True,full_contract_pass=False,
        data_receipt_sha256=digest(DATA/'receipt.json'),source_sha256=digest(__file__),
        model_source_sha256=digest(ROOT/'weld_pipeline/sjg_joint_model_817.py'),start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    try:
        assert read(DATA/'receipt.json')['status']=='PTV3_JOINT_DATA_COMPLETE'
        manifest=read(DATA/'all_inputs_frozen.json')
        for record in manifest['sources']:assert digest(DATA/record['file'])==record['sha256']
        check_resources();model=JointEventModel().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
        protocol['trainable_parameters']=sum(p.numel() for p in model.parameters());write(dest/'registered_protocol.json',protocol)
        before=score(model,dest,'before');initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.cuda.reset_peak_memory_stats()
        with (dest/'epochs.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,EPOCHS+1):
                model.train();rng=np.random.default_rng(SEED+epoch);losses=[];start=time.perf_counter();batches=0
                for source in rng.permutation(TRAIN):
                    graph,y,query=load_case(int(source),True);positive=np.flatnonzero(y>0);negative=np.flatnonzero(y==0)
                    order=rng.permutation(len(query['nodes'])).astype(np.int64)
                    for begin in range(0,len(order),BATCH):
                        if batches%12==0:check_resources()
                        ids=order[begin:begin+BATCH];a={k:cuda(v[ids]) for k,v in query.items()}
                        classified=np.r_[rng.choice(positive,256,replace=len(positive)<256),
                            rng.choice(negative,256,replace=len(negative)<256)].astype(np.int64)
                        selected=cuda(classified);optimizer.zero_grad(set_to_none=True);h=encode(model,graph)
                        c=model.encode_joints(h,graph['pairs'][selected],graph['geometry'][selected])
                        joint_loss=F.binary_cross_entropy_with_logits(model.relevance(c).squeeze(-1),cuda(y[classified]))
                        qids=a['nodes'];z=model.encode_joints(h,graph['pairs'][qids],graph['geometry'][qids])
                        logits,offset=model.query_events(z,a['patch'],a['query'])
                        event_loss=F.binary_cross_entropy_with_logits(logits,a['targets'],pos_weight=cuda(np.asarray([3.,8.,8.],np.float32)))
                        regression=F.smooth_l1_loss(offset,a['offsets'],reduction='none')
                        offset_loss=(regression*a['offset_mask']).sum()/a['offset_mask'].sum().clamp_min(1)
                        loss=joint_loss+event_loss+.2*offset_loss
                        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                        updates+=1;batches+=1;losses.append([float(joint_loss.detach()),float(event_loss.detach()),float(offset_loss.detach())])
                    del graph,y,query;gc.collect()
                assert batches==96;complete=epoch
                row=dict(epoch=epoch,epochs=EPOCHS,actual_updates=updates,batches=batches,
                    queries_seen=49152,mean_losses=np.mean(losses,0).tolist(),elapsed_seconds=time.perf_counter()-start)
                log.write(json.dumps(row)+'\n');log.flush()
                print('EPOCH',epoch,'/',EPOCHS,'UPDATES',updates,'/1920','JOINT/EVENT/OFFSET',row['mean_losses'],flush=True)
                with (dest/'last.pt').open('wb') as handle:
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,protocol=protocol),handle)
        after=score(model,dest,'after')
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),initial[k])]
        assert changed and any(k.startswith('layers.') for k in changed) and any(k.startswith('event.') for k in changed)
        write(dest/'receipt.json',dict(status='JOINT_EVENT_HEAD_REGISTERED_TRAINING_COMPLETE',actual_optimizer_updates=updates,
            complete_query_epochs=complete,changed_parameter_tensors=len(changed),checkpoint_sha256=digest(dest/'last.pt'),
            peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,end_utc=utc(),after=after,
            free_weld_output_not_yet_evaluated=True,full_contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',actual_optimizer_updates=updates,
            complete_query_epochs=complete,end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
