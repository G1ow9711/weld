"""Train interval validity and parent links, then freeze eight free outputs."""
import argparse
import gc
import json
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_parent_model_831 import IntervalParentModel
from weld_pipeline.sjg_parent_decode_832 import assemble,suppress_parent_hypotheses,decoded
from weld_pipeline.contract_geometry_d3_409 import score_source

DATA=OUT/'run46';BASE=OUT/'run45';EPOCHS=24;STEPS_PER_CASE=8;SEED=833


def load_input(source):
    with np.load(DATA/f'source_{source:02d}_inputs.npz') as z:return {k:torch.from_numpy(z[k]) for k in z.files}


def balanced_indices(y,rng,n=256):
    positive=np.flatnonzero(y>0);negative=np.flatnonzero(y<=0)
    if not len(positive) or not len(negative):return rng.choice(len(y),n,replace=len(y)<n).astype(np.int64)
    return np.r_[rng.choice(positive,n//2,replace=len(positive)<n//2),rng.choice(negative,n-n//2,replace=len(negative)<n-n//2)].astype(np.int64)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(SEED)
    updates=0;cases={};completed=0
    protocol=dict(stage='PURE_POINT_INTERVAL_VALIDITY_AND_PARENT_TRAIN',train=TRAIN,development=VALID,arc=ARC,
        original_G=496,epochs=EPOCHS,steps_per_nonempty_case=STEPS_PER_CASE,
        epoch_definition='Eight balanced interval/link minibatches per nonempty TRAIN case',
        learning_rate=.001,seed=SEED,device='cpu',node_threshold=.5,link_threshold=.5,shared_threshold=.5,
        geometric_parent_NMS_tolerance_mm=1.,material_identity_unknown=True,
        dedup_is_geometric_hypothesis_suppression_not_material_truth=True,
        no_GT_parent_input=True,STEP_reads=0,all_output_freezes_before_test_GT=True,
        source_sha256=digest(__file__),model_sha256=digest(ROOT/'weld_pipeline/sjg_parent_model_831.py'),
        decoder_sha256=digest(ROOT/'weld_pipeline/sjg_parent_decode_832.py'),
        full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    try:
        assert read(DATA/'receipt.json')['status']=='PURE_PARENT_INPUTS_AND_TRAIN_TARGETS_COMPLETE'
        for record in read(DATA/'all_inputs_frozen.json')['sources']:assert digest(DATA/record['file'])==record['sha256']
        for source in TRAIN:
            graph=load_input(source)
            with np.load(DATA/f'source_{source:02d}_targets.npz') as z:y={k:z[k] for k in z.files}
            if len(graph['nodes']):cases[source]=(graph,y)
        model=IntervalParentModel();node=torch.cat([g['nodes'] for g,y in cases.values()])
        edge=torch.cat([g['relation'] for g,y in cases.values() if len(g['relation'])])
        model.node_mean.copy_(node.mean(0));model.node_scale.copy_(node.std(0,unbiased=False).clamp_min(.01))
        model.edge_mean.copy_(edge.mean(0));model.edge_scale.copy_(edge.std(0,unbiased=False).clamp_min(.01))
        del node,edge;initial={k:v.clone() for k,v in model.state_dict().items()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        protocol['updates_per_epoch']=STEPS_PER_CASE*len(cases);protocol['nonempty_training_cases']=list(cases)
        write(dest/'registered_protocol.json',protocol)
        with (dest/'epochs.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,EPOCHS+1):
                check_resources();model.train();rng=np.random.default_rng(SEED+epoch);losses=[]
                for source in rng.permutation(list(cases)):
                    graph,y=cases[int(source)]
                    for _ in range(STEPS_PER_CASE):
                        ni=torch.from_numpy(balanced_indices(y['node'],rng))
                        h=model.encode(graph['nodes']);node_loss=F.binary_cross_entropy_with_logits(model.node_logits(h[ni]),torch.from_numpy(y['node'])[ni])
                        link_loss=shared_loss=h.sum()*0
                        if len(graph['pairs']):
                            ei=torch.from_numpy(balanced_indices(y['edge'][:,0],rng))
                            logits=model.edge_logits(h,graph['pairs'][ei],graph['relation'][ei]);truth=torch.from_numpy(y['edge'])[ei]
                            link_loss=F.binary_cross_entropy_with_logits(logits[:,0],truth[:,0])
                            shared_loss=(F.binary_cross_entropy_with_logits(logits[:,1],truth[:,1],reduction='none')*truth[:,0]).sum()/truth[:,0].sum().clamp_min(1)
                        loss=node_loss+link_loss+shared_loss;assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                        updates+=1;losses.append([float(node_loss.detach()),float(link_loss.detach()),float(shared_loss.detach())])
                completed=epoch;row=dict(epoch=epoch,epochs=EPOCHS,actual_updates=updates,mean_losses=np.mean(losses,0).tolist())
                log.write(json.dumps(row)+'\n');log.flush();print('PARENT_EPOCH',row,flush=True)
        with (dest/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=completed,updates=updates,protocol=protocol),h)
        changed=sum(not torch.equal(v,initial[k]) for k,v in model.state_dict().items());assert changed>0
        del cases;gc.collect();model.eval();frozen=[]
        with torch.inference_mode():
            for source in TRAIN+VALID+ARC:
                check_resources();graph=load_input(source);h=model.encode(graph['nodes']);node_p=model.node_logits(h).sigmoid().numpy()
                edge_parts=[]
                for begin in range(0,len(graph['pairs']),4096):edge_parts.append(model.edge_logits(h,graph['pairs'][begin:begin+4096],graph['relation'][begin:begin+4096]).sigmoid().numpy())
                edge_p=np.concatenate(edge_parts) if edge_parts else np.empty((0,2))
                raw=read(BASE/f'source_{source:02d}_predictions.json');assert not raw['invalid_intervals']
                parents,detail=assemble(raw['predictions'],graph['pairs'].numpy(),node_p,edge_p)
                kept,suppressed=suppress_parent_hypotheses(parents,tolerance=1.)
                raw_path=dest/f'source_{source:02d}_parents_before_nms.json';path=dest/f'source_{source:02d}_parents.json'
                write(raw_path,decoded(parents));write(path,decoded(kept))
                write(dest/f'source_{source:02d}_selection.json',dict(node_probability=node_p.tolist(),
                    **detail,suppressed_parent_hypotheses=suppressed))
                record=dict(source=source,file=path.name,sha256=digest(path),before_nms_file=raw_path.name,
                    before_nms_sha256=digest(raw_path),original_intervals=len(raw['predictions']),
                    retained_intervals=sum(p['child_count'] for p in parents),final_children=sum(p['child_count'] for p in kept),
                    parents=len(kept),suppressed_parents=len(suppressed))
                frozen.append(record);print('PURE_PARENT_OUTPUT_FROZEN',record,flush=True)
        write(dest/'all_predictions_frozen.json',dict(sources=frozen,all_before_test_GT=True))
        inventory=read(OUT/'inventory.json');rows=[]
        for record in frozen:
            source=record['source'];target_path=ROOT/inventory[source]['target']['target_path']
            assert digest(target_path)==inventory[source]['target']['target_sha256'];target=read(target_path)
            score=score_source(read(dest/record['file']),target);before=score_source(read(dest/record['before_nms_file']),target)
            write(dest/f'source_{source:02d}_strict_geometry.json',score);write(dest/f'source_{source:02d}_strict_before_nms.json',before)
            row=dict(source=source,strict=score['categories']['overall'],before_nms=before['categories']['overall']);rows.append(row)
            print('PURE_PARENT_STRICT_RESULT',row,flush=True)
        write(dest/'receipt.json',dict(status='PURE_PARENT_TRAIN_AND_STRICT_GEOMETRY_COMPLETE',actual_optimizer_updates=updates,
            completed_epochs=completed,changed_parameter_tensors=changed,checkpoint_sha256=digest(dest/'last.pt'),
            rows=rows,end_utc=utc(),full_contract_pass=False,curves_missing=True,material_E16_unresolved=True))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',actual_optimizer_updates=updates,completed_epochs=completed,
            end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
