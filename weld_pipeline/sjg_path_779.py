"""Learned parent continuation and shared nodes on frozen small-pilot geometry.

Training uses original parent identities only as labels. Free decoding uses
predicted children, input contact relations and fixed learned parameters.
"""
import argparse
from copy import deepcopy
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_model_777 import SurfaceJointNet777
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_contract_score_778 import adapt,total
from weld_pipeline.contract_geometry_d3_409 import score_source

BASE=OUT/"run03"
SEED=2026100779

class ParentLink(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.register_buffer("mean",torch.zeros(dim))
        self.register_buffer("scale",torch.ones(dim))
        self.net=nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))
    def forward(self,x):return self.net((x-self.mean)/self.scale)

def input_context(si,base_model):
    folder=OUT/f"source_{si:02d}";freeze=read(folder/"input_freeze.json")
    assert digest(folder/"inputs.npz")==freeze["input_sha256"]
    assert digest(folder/"geometry.json")==freeze["geometry_sha256"]
    with np.load(folder/"inputs.npz",allow_pickle=False) as a:
        nodes=torch.from_numpy(a["nodes"]).cuda()
        edges=torch.from_numpy(a["edge_index"]).cuda()
        types=torch.from_numpy(a["edge_type"]).cuda()
    with torch.no_grad(): h=base_model.encode_graph(nodes,edges,types).cpu().numpy()
    return read(folder/"geometry.json")["contacts"],h

def canonical_child(contact,lo,hi,parent=None,child=None):
    d=np.asarray(contact["direction"]);p=np.asarray(contact["p0"])
    return dict(start=p+lo*d,end=p+hi*d,lo=float(lo),hi=float(hi),parent=parent,child=child)

def candidate_features(children,contacts,h):
    pairs=[];features=[]
    for i,a in enumerate(children):
        ca=contacts[a["contact"]];da=np.asarray(ca["direction"])
        for j,b in enumerate(children):
            if i==j:continue
            cb=contacts[b["contact"]];db=np.asarray(cb["direction"])
            if float(da@db)<.999:continue
            delta=b["start"]-a["end"];along=float(delta@da)
            perp=float(np.linalg.norm(delta-along*da))
            ahead=float(((b["start"]+b["end"])-(a["start"]+a["end"]))@da)
            shared_materials=len(set(ca["materials"])&set(cb["materials"]))
            if ahead<=1e-6 or along< -3. or perp>3. or not shared_materials:continue
            geometry=np.r_[along/100,perp/3,np.log1p(a["hi"]-a["lo"])/5,
                np.log1p(b["hi"]-b["lo"])/5,a["lo"]/100,a["hi"]/100,
                b["lo"]/100,b["hi"]/100,ca["length"]/100,cb["length"]/100,
                shared_materials/2,float(a["contact"]==b["contact"]),da,db]
            features.append(np.r_[geometry,h[a["contact"]],h[b["contact"]]])
            pairs.append((i,j))
    return np.asarray(pairs,np.int64).reshape(-1,2),np.asarray(features,np.float32).reshape(-1,18+2*h.shape[1])

def train_examples(si,contacts,h,inventory):
    coverage=read(OUT/f"source_{si:02d}/coverage_v2.json")
    target=read(ROOT/inventory[si]["target"]["target_path"])
    children=[]
    for a in coverage["assignments"]:
        c=canonical_child(contacts[a["contact"]],a["lo"],a["hi"],a["parent"],a["child"])
        c["contact"]=a["contact"];children.append(c)
    pairs,x=candidate_features(children,contacts,h);y=[]
    for i,j in pairs:
        a,b=children[i],children[j]
        link=a["parent"]==b["parent"] and abs(a["child"]-b["child"])==1
        shared=bool(link and target["parents"][a["parent"]]["gaps"][min(a["child"],b["child"])]=="SHARED")
        y.append([float(link),float(shared)])
    return x,np.asarray(y,np.float32).reshape(-1,2)

def free_children(predictions,contacts):
    out=[]
    for parent in predictions:
        ci=int(parent["id"]);c=contacts[ci]
        for segment in parent["segments"]:
            q=(np.asarray(segment["points"])[[0,-1]]-c["p0"])@c["direction"]
            a=canonical_child(c,min(q),max(q));a["contact"]=ci;a["weld_type"]=parent["type"]
            out.append(a)
    return out

def assemble(predictions,contacts,h,model):
    children=free_children(predictions,contacts);pairs,x=candidate_features(children,contacts,h)
    with torch.no_grad():
        prob=model(torch.from_numpy(x).cuda()).sigmoid().cpu().numpy() if len(x) else np.empty((0,2))
    following={};previous={};shared_edges={};accepted=[]
    for k in np.argsort(-prob[:,0]):
        if prob[k,0]<.5:break
        i,j=map(int,pairs[k])
        if i in following or j in previous:continue
        cursor=j;cycle=False
        while cursor in following:
            cursor=following[cursor]
            if cursor==i:cycle=True;break
        if cycle:continue
        following[i]=j;previous[j]=i;shared_edges[(i,j)]=bool(prob[k,1]>=.5)
        accepted.append(dict(first=i,second=j,link_probability=float(prob[k,0]),shared_probability=float(prob[k,1])))
    final=[];covered=set()
    for start in range(len(children)):
        if start in previous:continue
        chain=[start]
        while chain[-1] in following:chain.append(following[chain[-1]])
        points=[np.asarray([children[i]["start"],children[i]["end"]]).copy() for i in chain]
        for k,(i,j) in enumerate(zip(chain[:-1],chain[1:])):
            if shared_edges[(i,j)]:
                # The learned state authorizes one shared coordinate. It is
                # derived from both predictions; no GT position is accessed.
                shared=(points[k][1]+points[k+1][0])*.5
                points[k][1]=shared;points[k+1][0]=shared
        final.append(dict(id=str(len(final)),type=children[start]["weld_type"],weld_side=None,
                          segments=[dict(kind="line",points=p.tolist()) for p in points]))
        covered.update(chain)
    assert len(covered)==len(children),"Parent assembly dropped a predicted child"
    return final,accepted

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",default="run04");args=ap.parse_args()
    out=OUT/args.run;out.mkdir(exist_ok=False);updates=0;started=time.monotonic()
    try:
        check_resources();torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED)
        inventory=read(OUT/"inventory.json")
        base_path=BASE/"relation_graph_last.pt";base_sha=digest(base_path)
        assert base_sha==read(BASE/"receipt.json")["arms"]["relation_graph"]["checkpoint_sha256"]
        with base_path.open("rb") as f: checkpoint=torch.load(f,map_location="cuda")
        base=SurfaceJointNet777(checkpoint["node_dim"]).cuda();base.load_state_dict(checkpoint["model"]);base.eval()
        for p in base.parameters():p.requires_grad_(False)
        context={si:input_context(si,base) for si in TRAIN+VALID}
        pieces=[train_examples(si,*context[si],inventory) for si in TRAIN]
        x=np.concatenate([p[0] for p in pieces]);y=np.concatenate([p[1] for p in pieces])
        assert y[:,0].sum()>0 and (y[:,0]==0).sum()>0
        model=ParentLink(x.shape[1]).cuda()
        model.mean.copy_(torch.from_numpy(x.mean(0)).cuda())
        model.scale.copy_(torch.from_numpy(x.std(0).clip(.01)).cuda())
        xx=torch.from_numpy(x).cuda();yy=torch.from_numpy(y).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
        batch=128;epochs=100;max_updates=2000
        pw=float((len(y)-y[:,0].sum())/y[:,0].sum())
        protocol=dict(utc=utc(),stage="LEARNED_PARENT_CONTINUATION_AND_SHARED_NODE",
            train=TRAIN,holdout=VALID,epochs=epochs,batch=batch,maximum_updates=max_updates,
            batches_per_epoch=int(np.ceil(len(x)/batch)),training_pairs=len(x),
            positive_links=int(y[:,0].sum()),positive_shared=int(y[:,1].sum()),
            seed=SEED,optimizer="AdamW",learning_rate=.003,
            frozen_geometry_checkpoint_sha256=base_sha,
            continuation_threshold=.5,shared_threshold=.5,
            no_manual_gap_tolerance_snap=True,no_GT_reads_during_prediction=True,
            STEP_conditioned_not_PCD_acceptance=True,large_scale_allowed=False,
            local_review_of_failed_run="run03",external_messages=0,
            source_sha256=digest(__file__),full_contract_pass=False)
        write(out/"registered_protocol.json",protocol)
        rng=np.random.default_rng(SEED)
        with (out/"epochs.jsonl").open("w",encoding="utf8") as log:
            for epoch in range(1,epochs+1):
                order=rng.permutation(len(x));loss_sum=0.;steps=0
                for begin in range(0,len(x),batch):
                    if updates>=max_updates:break
                    ids=torch.as_tensor(order[begin:begin+batch],dtype=torch.long,device="cuda")
                    logits=model(xx[ids]);target=yy[ids]
                    link=F.binary_cross_entropy_with_logits(logits[:,0],target[:,0],pos_weight=torch.tensor(pw,device="cuda"))
                    shared=(F.binary_cross_entropy_with_logits(logits[:,1],target[:,1],reduction="none")*target[:,0]).sum()/target[:,0].sum().clamp_min(1)
                    loss=link+shared
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                    updates+=1;steps+=1;loss_sum+=float(loss.detach())
                status=dict(utc=utc(),epoch=epoch,epochs=epochs,batches=steps,updates=updates,
                    complete_epoch=steps==protocol["batches_per_epoch"],loss=loss_sum/max(steps,1))
                log.write(__import__("json").dumps(status)+"\n");log.flush();write(out/"progress.json",status)
                print("PARENT_LINK_TRAIN Epoch",str(epoch)+"/"+str(epochs),"Updates",updates,"Loss",round(status["loss"],6),flush=True)
                if epoch%10==0:check_resources()
                if updates>=max_updates:break
        with (out/"parent_link.pt").open("wb") as f:torch.save(dict(model=model.state_dict(),dim=x.shape[1],protocol=protocol),f)
        model.eval();frozen=[]
        for si in TRAIN+VALID:
            base_freeze=read(BASE/"relation_graph/prediction_freeze.json")
            row=next(r for r in base_freeze["outputs"] if r["source"]==si)
            path=BASE/"relation_graph"/row["path"];assert digest(path)==row["sha256"]
            pred,links=assemble(read(path),*context[si],model)
            target=out/f"source_{si:02d}_predictions.json";write(target,pred)
            write(out/f"source_{si:02d}_learned_links.json",links)
            frozen.append(dict(source=si,path=target.name,sha256=digest(target)))
            print("FROZEN_PATH_OUTPUT",si,"children",sum(len(p["segments"]) for p in pred),"parents",len(pred),flush=True)
        write(out/"prediction_freeze.json",dict(outputs=frozen,all_frozen_before_scoring=True))
        scores={}
        for row in frozen:
            si=row["source"];path=out/row["path"];assert digest(path)==row["sha256"]
            target_path=ROOT/inventory[si]["target"]["target_path"]
            assert digest(target_path)==inventory[si]["target"]["target_sha256"]
            scores[si]=score_source(adapt(read(path)),read(target_path))
            write(out/f"source_{si:02d}_contract_score.json",scores[si])
            print("CONTRACT_SCORE",si,scores[si]["categories"]["overall"],flush=True)
        assert digest(base_path)==base_sha
        write(out/"receipt.json",dict(utc=utc(),status="SMALL_PARENT_LINK_TRAIN_AND_CONTRACT_GEOMETRY_COMPLETE",
            actual_optimizer_updates=updates,train=total([scores[i] for i in TRAIN]),
            development_holdout=total([scores[i] for i in VALID]),elapsed_seconds=time.monotonic()-started,
            base_geometry_weights_unchanged=True,checkpoint_sha256=digest(out/"parent_link.pt"),
            pure_XYZ_accuracy=None,E16_status="NOT_IMPLEMENTED_AND_INCOMPLETE_REVIEWED_TRUTH",
            arc_G_unimplemented=inventory[ARC[0]]["G"],full_contract_pass=False,large_scale_allowed=False))
        print("PARENT_LINK_PILOT_COMPLETE | full acceptance not passed",flush=True)
    except BaseException as error:
        write(out/"failure.json",dict(error=str(error),traceback=traceback.format_exc(),updates=updates))
        raise
if __name__=="__main__":main()
