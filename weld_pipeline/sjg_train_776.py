"""Registered experiment A. Conditional structure scores are not PCD acceptance."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.signal import find_peaks
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_model_776 import SurfaceJointNet
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import TRAIN,VALID,ARC,OUT,ROOT
from weld_pipeline.contract_child_evaluation import score_children
from weld_pipeline.world_set_evaluation_285 import reference_segment

SEED=20261007
BATCH=512
EPOCHS=20
UPDATE_LIMIT=2000

def utc(): return datetime.now(timezone.utc).isoformat()
def check_resources():
    from weld_pipeline.reports.full25_source_geometry_263_20261003.run_remaining_source_batch_263 import memory
    import shutil
    m=memory()
    if m["available_physical_bytes"]<1342177280 or m["remaining_commit_bytes"]<2684354560:
        raise RuntimeError("Resource guard: insufficient physical RAM or remaining commit")
    if m["private_commit_bytes"]>12884901888: raise RuntimeError("Resource guard: worker commit exceeds 12 GiB")
    if shutil.disk_usage(ROOT).free<5368709120: raise RuntimeError("Resource guard: less than 5 GiB disk free")
    return m

def load_source(si):
    folder=OUT/("source_%02d"%si)
    freeze=read(folder/"input_freeze.json")
    assert digest(folder/"inputs.npz")==freeze["input_sha256"]
    assert digest(folder/"geometry.json")==freeze["geometry_sha256"]
    binding=next(s for s in read(OUT/"pilot_selection.json")["sources"] if s["source"]==si)
    assert digest(folder/"supervision_v2.npz")==binding["supervision_sha256"]
    with np.load(folder/"inputs.npz",allow_pickle=False) as z: a={k:z[k] for k in z.files}
    with np.load(folder/"supervision_v2.npz",allow_pickle=False) as z: a.update({k:z[k] for k in z.files})
    a["contacts"]=read(folder/"geometry.json")["contacts"]
    a["query_node"]=np.repeat(np.arange(len(a["nodes"]),dtype=np.int64),np.diff(a["ptr"]))
    a["si"]=si
    for k in ("nodes","edge_index","edge_type"):
        a[k+"_gpu"]=torch.from_numpy(a[k]).cuda()
    return a

def choose_queries(a):
    rng=np.random.default_rng(SEED+a["si"])
    hot=np.flatnonzero(a["targets"][:,1:].max(1)>.1)
    foreground=np.flatnonzero(a["targets"][:,0]>.5)
    negative=np.flatnonzero(a["targets"].max(1)<.01)
    chosen=set(map(int,hot))
    for pool,budget in ((foreground,2048),(negative,4096)):
        if len(pool): chosen.update(map(int,rng.choice(pool,min(len(pool),budget),replace=False)))
    remaining=np.setdiff1d(np.arange(len(a["query"])),np.fromiter(chosen,dtype=np.int64))
    fill=max(0,8192-len(chosen))
    if fill and len(remaining): chosen.update(map(int,rng.choice(remaining,min(fill,len(remaining)),replace=False)))
    # Never remove endpoint neighbourhoods to hit an arbitrary sample cap.
    return np.asarray(sorted(chosen),np.int64)

def tensors(a,ids):
    return [torch.from_numpy(np.ascontiguousarray(a[k][ids])).cuda()
            for k in ("patches","query","query_node","targets","offsets","offset_mask")]

def decode_free(a,probs,offsets):
    output=[]
    for ni,c in enumerate(a["contacts"]):
        start,stop=a["ptr"][ni:ni+2];p=probs[start:stop];o=offsets[start:stop]
        step=c["query_step_mm"]
        starts=find_peaks(np.r_[-1,p[:,1],-1],height=.5,distance=2)[0]-1
        ends=find_peaks(np.r_[-1,p[:,2],-1],height=.5,distance=2)[0]-1
        segments=[];used=set()
        for s in starts:
            legal=[int(e) for e in ends if e>=s and int(e) not in used]
            if not legal: continue
            e=legal[0]
            lo=float(np.clip(s*step+2*np.clip(o[s,0],-1.5,1.5),0,c["length"]))
            hi=float(np.clip(e*step+2*np.clip(o[e,1],-1.5,1.5),0,c["length"]))
            if hi-lo<1 or float(p[s:e+1,0].mean())<.5: continue
            used.add(e)
            d=np.asarray(c["direction"]);origin=np.asarray(c["p0"])
            segments.append(dict(kind="line",points=[(origin+lo*d).tolist(),(origin+hi*d).tolist()]))
        if segments:
            output.append(dict(id=str(ni),type="vertical" if abs(c["direction"][2])>.707 else "horizontal",
                               segments=segments,weld_side=None))
    return output

def evaluate(model,a):
    model.eval()
    probs=[];offs=[]
    with torch.no_grad():
        h=model.encode_graph(a["nodes_gpu"],a["edge_index_gpu"],a["edge_type_gpu"])
        for begin in range(0,len(a["query"]),2048):
            ids=np.arange(begin,min(begin+2048,len(a["query"])))
            pp,qq,ii,_,_,_=tensors(a,ids)
            l,o=model.decode(h,pp,qq,ii)
            probs.append(l.sigmoid().cpu().numpy());offs.append(o.cpu().numpy())
    return decode_free(a,np.concatenate(probs),np.concatenate(offs))

def score_frozen(predictions,si,inventory):
    target_path=ROOT/inventory[si]["target"]["target_path"]
    assert digest(target_path)==inventory[si]["target"]["target_sha256"]
    target=read(target_path)
    truths=[dict(id=p["original_parent_id"],type=p["children"][0]["weld_type"],
                 segments=[reference_segment(c) for c in p["children"]],weld_side=None)
            for p in target["parents"]]
    score=score_children(predictions,truths)
    score.update(source=si,scope="CONDITIONAL_STEP_GRAPH_CHILD_GEOMETRY_DIAGNOSTIC",
        full_G=target["original_children"],formal_independent=False,
        full_parent_path_and_E16_pass=False,deployable_pure_XYZ=False,
        full_contract_pass=False)
    return score

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",default="run01");args=ap.parse_args()
    out=OUT/args.run;out.mkdir(exist_ok=False)
    start=time.monotonic();all_updates=0
    try:
        torch.set_num_threads(2)
        assert torch.cuda.is_available()
        print("SJG-776 EXPERIMENT A | CONDITIONAL STRUCTURE | NOT DELIVERY ACCURACY",flush=True)
        resources=check_resources()
        inventory=read(OUT/"inventory.json")
        data={i:load_source(i) for i in TRAIN+VALID}
        selected={i:choose_queries(data[i]) for i in TRAIN}
        for i,ids in selected.items(): np.save(out/("train_queries_%02d.npy"%i),ids)
        per_epoch=sum(int(np.ceil(len(ids)/BATCH)) for ids in selected.values())
        protocol=dict(utc=utc(),train_sources=TRAIN,holdout_sources=VALID,arc_unavailable_structure=ARC,
            complete_pilot_G=sum(inventory[i]["G"] for i in TRAIN+VALID+ARC),
            train_G=sum(inventory[i]["G"] for i in TRAIN),
            seed=SEED,batch=BATCH,epochs=EPOCHS,maximum_updates_per_arm=UPDATE_LIMIT,
            batches_per_complete_epoch=per_epoch,epoch_means_full_registered_query_list=True,
            selection={str(i):len(ids) for i,ids in selected.items()},
            threshold=.5,optimizer="AdamW",learning_rate=.001,weight_decay=.0001,
            decoder_minimum_predicted_length_mm=1.,event_peak_distance_samples=2,
            resources=resources,torch=torch.__version__,cuda_device=torch.cuda.get_device_name(),
            graph_uses_STEP=True,PCD_acceptance=False,no_expansion_without_full_small_sample_gates=True,
            source_hashes={Path(p).name:digest(p) for p in
                          [Path(__file__),ROOT/"weld_pipeline/sjg_model_776.py",ROOT/"weld_pipeline/sjg_prepare_776.py"]},
            full_contract_pass=False)
        write(out/"registered_protocol.json",protocol)
        y=np.concatenate([data[i]["targets"][ids] for i,ids in selected.items()])
        weights=np.clip((len(y)-y.sum(0))/np.maximum(y.sum(0),1),1,30)
        pos_weight=torch.as_tensor(weights,dtype=torch.float32,device="cuda")
        summaries={}
        with (out/"epochs.jsonl").open("w",encoding="utf-8") as log:
            for arm in ("local_classifier","relation_graph"):
                torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED)
                model=SurfaceJointNet(data[TRAIN[0]]["nodes"].shape[1],message_passing=arm=="relation_graph").cuda()
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                updates=0;rng=np.random.default_rng(SEED)
                for epoch in range(1,EPOCHS+1):
                    model.train();loss_sum=0.;steps=0;epoch_started=time.monotonic()
                    for si in rng.permutation(TRAIN):
                        a=data[int(si)];indices=rng.permutation(selected[int(si)])
                        for begin in range(0,len(indices),BATCH):
                            if updates>=UPDATE_LIMIT: break
                            ids=indices[begin:begin+BATCH]
                            pp,qq,ii,yy,oo,mask=tensors(a,ids)
                            opt.zero_grad(set_to_none=True)
                            logits,pred_offset=model(a["nodes_gpu"],a["edge_index_gpu"],a["edge_type_gpu"],pp,qq,ii)
                            ce=F.binary_cross_entropy_with_logits(logits,yy,pos_weight=pos_weight)
                            offset=(F.smooth_l1_loss(pred_offset,oo,reduction="none")*mask).sum()/mask.sum().clamp_min(1)
                            loss=ce+.25*offset
                            if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
                            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                            updates+=1;all_updates+=1;steps+=1;loss_sum+=float(loss.detach())
                            if updates%25==0:
                                check_resources()
                                status=dict(utc=utc(),arm=arm,epoch=epoch,epochs=EPOCHS,
                                    batch=steps,batches_per_epoch=per_epoch,optimizer_updates=updates,
                                    total_optimizer_updates=all_updates,loss=float(loss.detach()))
                                write(out/"progress.json",status)
                                print("TRAIN",arm,"Epoch",str(epoch)+"/"+str(EPOCHS),"Batch",
                                      str(steps)+"/"+str(per_epoch),"Updates",updates,"Loss",round(status["loss"],6),flush=True)
                        if updates>=UPDATE_LIMIT: break
                    entry=dict(utc=utc(),arm=arm,epoch=epoch,complete_epoch=steps==per_epoch,
                        batches=steps,updates=updates,mean_loss=loss_sum/max(steps,1),
                        seconds=time.monotonic()-epoch_started,
                        gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
                    log.write(json.dumps(entry)+"\n");log.flush()
                    print("EPOCH_END",json.dumps(entry),flush=True)
                    checkpoint=out/(arm+"_last.pt")
                    temporary=checkpoint.with_suffix(".pt.tmp")
                    with temporary.open("wb") as stream:
                        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                            updates=updates,node_dim=data[TRAIN[0]]["nodes"].shape[1],
                            message_passing=arm=="relation_graph",protocol=protocol),stream)
                    temporary.replace(checkpoint)
                    if updates>=UPDATE_LIMIT: break
                model.eval();armdir=out/arm;armdir.mkdir()
                frozen=[]
                for si in TRAIN+VALID:
                    check_resources()
                    pred=evaluate(model,data[si])
                    path=armdir/("source_%02d_predictions.json"%si);write(path,pred)
                    frozen.append(dict(source=si,path=path.name,sha256=digest(path)))
                    print("FROZEN_FREE_DECODE",arm,si,"predicted",sum(len(p["segments"]) for p in pred),flush=True)
                write(armdir/"prediction_freeze.json",dict(outputs=frozen,all_predictions_frozen_before_scoring=True))
                scores=[]
                for item in frozen:
                    p=armdir/item["path"];assert digest(p)==item["sha256"]
                    score=score_frozen(read(p),item["source"],inventory)
                    write(armdir/("source_%02d_score.json"%item["source"]),score);scores.append(score)
                    print("SCORE",arm,item["source"],json.dumps({k:score[k] for k in
                        ("truth_count","prediction_count","valid_children","false_predictions","valid_extraction_rate")}),flush=True)
                summaries[arm]=dict(updates=updates,scores=scores,checkpoint_sha256=digest(out/(arm+"_last.pt")))
                del model,opt
                torch.cuda.empty_cache()
        write(out/"receipt.json",dict(utc=utc(),status="CONDITIONAL_A_TRAINING_AND_GEOMETRY_DIAGNOSTIC_COMPLETE",
            total_optimizer_updates=all_updates,arms=summaries,elapsed_seconds=time.monotonic()-start,
            pure_XYZ_accuracy=None,full_contract_pass=False,large_scale_training_started=False,
            unimplemented_or_unverified=["pure_XYZ_surface_material_graph","arc_structure","full_parent_paths","E16_truth_and_prediction"],
            scope="DEVELOPMENT_CONDITIONAL_STRUCTURE_NOT_FINAL_MODEL"))
        print("PILOT_COMPLETE | Full contract NOT passed | No large-scale training",flush=True)
    except BaseException as e:
        write(out/"failure.json",dict(utc=utc(),error=str(e),traceback=traceback.format_exc(),
            actual_optimizer_updates=all_updates,full_contract_pass=False))
        raise
if __name__=="__main__":main()
