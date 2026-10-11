"""Joint endpoint compatibility, retaining every admitted geometric anchor.

No material-role truth is fabricated. Whole-face observations provide context;
the outputs remain interval proposals until parent assembly is validated.
"""
import argparse
import gc
import json
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, check_resources, reclaim_own_idle_pages
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_endpoint_role_train_1246 import inputs
from weld_pipeline.sjg_whole_joint_targets_1229 import TRAIN

EPOCHS=12
BATCH=64
CHUNK=64


def prefix_at(grid, probability, q):
    """Exact integral of the linearly interpolated frozen occupancy."""
    cumulative=np.r_[0., np.cumsum(np.diff(grid)*(probability[:-1]+probability[1:])/2)]
    i=np.clip(np.searchsorted(grid,q,side='right')-1,0,len(grid)-2)
    h=q-grid[i]
    return cumulative[i]+h*probability[i]+.5*h*h*(probability[i+1]-probability[i])/(grid[i+1]-grid[i])


def load_inputs(key,input_run='run396',context_run='run380',sequence_run='run390',occupancy_run='run392'):
    x,m,r=inputs(key,input_run)
    directory=OUT/f'{context_run}/source_{key}';cr=read(directory/'receipt.json')
    assert cr['GT_reads']==cr['STEP_reads']==0
    assert digest(directory/'joint_features.npy')==cr['feature_sha256']
    context=np.load(directory/'joint_features.npy',mmap_mode='r')
    sr=read(OUT/f'{sequence_run}/source_{key}/receipt.json')
    assert digest(OUT/f'{sequence_run}/source_{key}/sequence_map.npz')==sr['map_sha256']
    with np.load(OUT/f'{sequence_run}/source_{key}/sequence_map.npz') as z:
        gridmap={k:z[k] for k in z.files}
    op=OUT/f'{occupancy_run}/proposal_{key}/occupancy.npz'
    assert digest(op)==r['occupancy_sha256']
    with np.load(op) as z: occup={k:z[k] for k in z.files}
    assert np.array_equal(m['rows'],occup['admitted_rows'])
    lookup={int(p):i for i,p in enumerate(gridmap['profile_ids'])}
    prefixes=[]
    for i,row in enumerate(m['rows']):
        a,b=m['offsets'][i:i+2];j=lookup[int(row)//2]
        u,v=gridmap['offsets'][j:j+2];lo,hi=occup['offsets'][i:i+2]
        prefixes.append(prefix_at(gridmap['q_mm'][u:v],occup['probabilities'][lo:hi],m['q_mm'][a:b]))
    return dict(x=x,m=m,context=context,prefix=np.concatenate(prefixes),
        receipt=r,context_sha256=cr['feature_sha256'])


class PairModel(nn.Module):
    def __init__(self,classes):
        super().__init__();self.classes=classes;c=len(classes)
        self.endpoint=nn.Sequential(nn.Linear(180,64),nn.ReLU(),nn.Linear(64,32),nn.Tanh())
        self.context=nn.Sequential(nn.Linear(160,48),nn.ReLU(),nn.Linear(48,32),nn.Tanh())
        self.pair=nn.Parameter(torch.randn(c,32,32)*.02)
        self.unary=nn.Linear(32,c)
        self.span=nn.Sequential(nn.Linear(4,12),nn.ReLU(),nn.Linear(12,c))

    def encode(self,x,context):
        return self.endpoint(x)+self.context(context)[None,:]

    def scores(self,a,b,q,prefix,length,closed,lo,hi):
        left=a[lo:hi]
        score=torch.einsum('ik,ckl,jl->ijc',left,self.pair,b)/np.sqrt(32)
        score=score+self.unary(left)[:,None,:]+self.unary(b)[None,:,:]
        span=(q[None,:]-q[lo:hi,None]).clamp_min(0)
        mean=(prefix[None,:]-prefix[lo:hi,None])/span.clamp_min(1e-6)
        geometry=torch.stack((torch.log1p(span)/10,span/length,mean,
                              torch.full_like(span,float(closed))),dim=-1)
        return score+self.span(geometry)


def tensors(source,record,norm,device):
    a=record['q_offset'];n=record['n'];f=record['feature_offset']
    raw=np.array(source['x'][f:f+2*n],copy=True)
    raw=(raw-norm['mean'])/norm['scale']
    context=(np.array(source['context'][record['row']],copy=True)-norm['context_mean'])/norm['context_scale']
    q=source['m']['q_mm'][a:a+n];prefix=source['prefix'][a:a+n]
    return [torch.as_tensor(z,dtype=torch.float32,device=device) for z in (raw,context,q,prefix)]


def select_intervals(margin,category):
    """Exact maximum-weight nonoverlapping intervals; adjacent segments allowed."""
    n=len(margin);best=np.zeros(n);previous=np.full(n,-1,int);taken=np.zeros(n,bool)
    for j in range(1,n):
        values=best[:j]+margin[:j,j];i=int(np.argmax(values))
        best[j]=best[j-1];previous[j]=j-1
        if values[i]>best[j]:best[j]=values[i];previous[j]=i;taken[j]=True
    chosen=[];j=n-1
    while j>0:
        i=int(previous[j])
        if taken[j]:chosen.append((i,j,int(category[i,j]),float(margin[i,j])))
        j=i
    return chosen[::-1]


def controls(model,dest):
    import itertools
    m=np.full((5,5),-10.);m[0,2]=3;m[2,4]=4;m[0,4]=6
    selected=select_intervals(m,np.ones((5,5),int))
    assert [(a,b) for a,b,_,_ in selected]==[(0,2),(2,4)]
    # Exhaustive small interval subsets independently verify DP optimality.
    rng=np.random.default_rng(813)
    for _ in range(12):
        m=rng.normal(size=(5,5));edges=list(zip(*np.triu_indices(5,1)));opt=0.
        for mask in range(1<<len(edges)):
            es=[edges[i] for i in range(len(edges)) if mask>>i&1]
            if any(max(a,c)<min(b,d) for (a,b),(c,d) in itertools.combinations(es,2)):continue
            opt=max(opt,sum(m[a,b] for a,b in es))
        assert abs(sum(v for a,b,c,v in select_intervals(m,np.ones((5,5),int)))-opt)<1e-10
    q=np.array([0.,.5,2.,5.]);assert np.allclose(prefix_at(np.array([0.,5.]),np.array([0.,1.]),q),q*q/10)
    with torch.no_grad():
        x=torch.randn(14,180);c=torch.randn(160);h=model.encode(x,c);q=torch.arange(7,dtype=torch.float32)
        p=q*.5
        all_scores=model.scores(h[:7],h[7:],q,p,6,False,0,7)
        chunks=torch.cat([model.scores(h[:7],h[7:],q,p,6,False,i,min(i+2,7)) for i in range(0,7,2)])
        delta=float((all_scores-chunks).abs().max());assert delta<1e-5
    y=torch.randint(len(model.classes),(7,7));w=torch.triu(torch.rand(7,7),diagonal=1)
    model.zero_grad(set_to_none=True);h=model.encode(x,c)
    full=model.scores(h[:7],h[7:],q,p,6,False,0,7)
    (F.cross_entropy(full.permute(0,2,1),y,reduction='none')*w).sum().backward()
    gradients=[v.grad.clone() for v in model.parameters()]
    model.zero_grad(set_to_none=True);h=model.encode(x,c)
    for lo in range(0,7,2):
        hi=min(lo+2,7);scores=model.scores(h[:7],h[7:],q,p,6,False,lo,hi)
        (F.cross_entropy(scores.permute(0,2,1),y[lo:hi],reduction='none')*w[lo:hi]).sum().backward(retain_graph=hi<7)
    grad_delta=max(float((v.grad-g).abs().max()) for v,g in zip(model.parameters(),gradients))
    assert grad_delta<1e-5
    model.zero_grad(set_to_none=True)
    write(dest/'controls.json',dict(DP_matches_exhaustive=True,adjacent_segments_preserved=True,
        occupancy_integral_exact=True,chunk_logits_max_abs=delta,chunk_gradient_max_abs=grad_delta,GT_reads=0))


def train(dest,resume=False):
    if (OUT/'run404/confirmed_supervision_conflicts.json').exists():
        raise RuntimeError('run398 inherits an incomplete native-D3 compatibility mapping; rebuild supervision before further fitting')
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(1031)
    torch.backends.cudnn.enabled=False;torch.backends.cuda.matmul.allow_tf32=False
    supervision=read(OUT/'run398/receipt.json');assert supervision['G']==213
    assert supervision['classes']==list(range(len(supervision['classes'])))
    assert read(OUT/'run398/supervision_integrity_audit.json')['status']=='PASS'
    sources={};records=[];contexts=[];bindings=[]
    for c in supervision['cases']:
        key=c['source'];f=OUT/f'run398/source_{key}_pair_targets.json'
        assert digest(f)==c['targets_sha256'];v=read(f);s=load_inputs(key);sources[key]=s
        assert s['receipt']['feature_sha256']==v['feature_sha256'] and s['receipt']['map_sha256']==v['map_sha256']
        for row in v['records']:
            row['negative_weight']=v['negative_weight'];records.append(row)
            contexts.append(np.array(s['context'][row['row']],copy=True))
        bindings.append(dict(source=key,supervision_sha256=digest(f),context_sha256=s['context_sha256']))
    assert digest(OUT/'run398/normalization.npz')==supervision['normalization_sha256']
    with np.load(OUT/'run398/normalization.npz') as z:norm={k:z[k] for k in z.files}
    cx=np.asarray(contexts,dtype=np.float64);cs=cx.std(0);cs[cs<1e-6]=1
    norm.update(context_mean=cx.mean(0).astype(np.float32),context_scale=cs.astype(np.float32))
    model=PairModel(supervision['classes']);controls(model,dest)
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(1031)
    trace=[];first=1;elapsed=0.;updates=0
    if resume:
        state=torch.load(dest/'latest_training_state.pt',map_location=device)
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['rng'];trace=state['trace'];first=state['epoch']+1
        elapsed=trace[-1]['seconds'];updates=state['updates']
        assert all(np.array_equal(norm[k],state['norm'][k]) for k in norm)
        write(dest/'resume_receipt.json',dict(checkpoint_sha256=digest(dest/'latest_training_state.pt'),next_epoch=first))
    reclaim_own_idle_pages();check_resources()
    print('PAIR_TRAIN',len(records),'CARRIERS',supervision['all_pairs'],'PAIRS_PER_EPOCH',device,flush=True)
    for epoch in range(first,EPOCHS+1):
        ep=time.perf_counter();total_loss=0.;pair_count=0;order=rng.permutation(len(records))
        for base in range(0,len(order),BATCH):
            ids=order[base:base+BATCH];optimizer.zero_grad(set_to_none=True)
            for ix in ids:
                r=records[ix];n=r['n']
                if n<2:continue
                x,c,q,p=tensors(sources[r['source']],r,norm,device);h=model.encode(x,c)
                indices=np.asarray(r['positive_indices'],dtype=np.int64)
                labels=np.asarray(r['positive_labels'],dtype=np.int64)
                weights=np.asarray(r['positive_weights'],dtype=np.float32)
                for lo in range(0,n-1,CHUNK):
                    hi=min(lo+CHUNK,n-1);score=model.scores(h[:n],h[n:],q,p,r['curve_length_mm'],r['is_closed'],lo,hi)
                    y=torch.zeros((hi-lo,n),dtype=torch.long,device=device)
                    w=torch.full((hi-lo,n),r['negative_weight'],device=device)
                    left=np.searchsorted(indices,lo*n);right=np.searchsorted(indices,hi*n)
                    if right>left:
                        pos=torch.as_tensor(indices[left:right]-lo*n,device=device)
                        y.view(-1)[pos]=torch.as_tensor(labels[left:right],device=device)
                        w.view(-1)[pos]=torch.as_tensor(weights[left:right],device=device)
                    valid=torch.arange(n,device=device)[None,:]>torch.arange(lo,hi,device=device)[:,None]
                    value=(F.cross_entropy(score.permute(0,2,1),y,reduction='none')*w*valid).sum()
                    assert torch.isfinite(value)
                    (value*len(records)/len(ids)).backward(retain_graph=hi<n-1)
                    total_loss+=float(value.detach());pair_count+=int(sum(n-i-1 for i in range(lo,hi)))
                del x,c,q,p,h,score,y,w,value,valid
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            assert torch.isfinite(gradient);optimizer.step();updates+=1
        assert pair_count==supervision['all_pairs']
        trace.append(dict(epoch=epoch,loss=total_loss,all_pairs_seen=pair_count,
            epoch_seconds=time.perf_counter()-ep,seconds=elapsed+time.perf_counter()-start,updates=updates))
        atomic_binary_write(dest/'latest_training_state.pt',lambda f:torch.save(dict(state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,trace=trace,epoch=epoch,norm=norm,updates=updates),f))
        write(dest/'training_progress.json',dict(trace=trace,total_epochs=EPOCHS))
        print('EPOCH',epoch,'/',EPOCHS,'LOSS',round(total_loss,6),'PAIRS',pair_count,'SECONDS',round(trace[-1]['epoch_seconds'],2),flush=True)
        gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        write(dest/'latest_memory_trim.json',reclaim_own_idle_pages());check_resources()
    model.cpu().eval();controls(model,dest)
    payload=dict(state_dict=model.state_dict(),norm=norm,classes=supervision['classes'],epochs=EPOCHS)
    atomic_binary_write(dest/'pair_model.pt',lambda f:torch.save(payload,f))
    reloaded=PairModel(payload['classes']);reloaded.load_state_dict(torch.load(dest/'pair_model.pt')['state_dict'])
    assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items())
    write(dest/'training_receipt.json',dict(status='JOINT_INTERVAL_MODEL_TRAINED',TRAIN=list(TRAIN),G=213,
        epochs=EPOCHS,trace=trace,input_bindings=bindings,model_sha256=digest(dest/'pair_model.pt'),
        reload_exact=True,DEV_reads=0,STEP_reads=0,endpoint_gate_used=False,
        explicit_material_role_labels_available=False,full_contract_pass=False,seconds=time.perf_counter()-start+elapsed))
    print('PAIR_TRAIN_COMPLETE',flush=True)


def infer(dest,key,input_run=None,context_run='run380',sequence_run='run390',occupancy_run='run392'):
    from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
    from weld_pipeline.sjg_finite_face_domains_1235 import decode_shared,ContinuousGraphBank
    from weld_pipeline.sjg_local_interval_pilot_1238 import attributes
    start=time.perf_counter();torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    receipt=read(dest/'training_receipt.json');assert digest(dest/'pair_model.pt')==receipt['model_sha256']
    payload=torch.load(dest/'pair_model.pt',map_location='cpu');device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32=False
    model=PairModel(payload['classes']).to(device).eval();model.load_state_dict(payload['state_dict'])
    source=load_inputs(key,input_run or payload.get('input_run','run396'),context_run,sequence_run,occupancy_run);case=load_case(key);g=case['graph'];case['x']._mmap.close()
    choices=[];interval_records=[];total_pairs=0
    for i,row in enumerate(source['m']['rows']):
        row=int(row);a,b=map(int,source['m']['offsets'][i:i+2]);n=b-a;curve=g.curve(row//2)
        if n<2:continue
        r=dict(row=row,q_offset=a,feature_offset=2*a,n=n,curve_length_mm=curve.length,
            is_closed=curve.kind=='arc' and curve.arc.is_closed)
        x,c,q,p=tensors(source,r,payload['norm'],device)
        margins=np.full((n,n),-np.inf,np.float32);categories=np.zeros((n,n),np.uint8)
        with torch.no_grad():
            h=model.encode(x,c)
            for lo in range(0,n-1,CHUNK):
                hi=min(lo+CHUNK,n-1);score=model.scores(h[:n],h[n:],q,p,curve.length,r['is_closed'],lo,hi)
                value,index=(score[:,:,1:]-score[:,:,:1]).max(-1)
                margins[lo:hi]=value.cpu().numpy();categories[lo:hi]=np.asarray(payload['classes'])[index.cpu().numpy()+1]
        rawq=source['m']['q_mm'][a:b]
        for left,right,category,margin in select_intervals(margins,categories):
            attr=attributes(category)
            choices.append(dict(query=len(choices),probability=float(1/(1+np.exp(-min(margin,80)))),child_count=1,
                closed=attr['closed'],children=[dict(profile=row//2,intervals=[0,0],
                    fractions=[float(rawq[left]/curve.length),float(rawq[right]/curve.length)],
                    side='UNKNOWN',connection='UNKNOWN',**attr)]))
            interval_records.append(dict(row=row,lo_mm=float(rawq[left]),hi_mm=float(rawq[right]),category=category,margin=margin))
        total_pairs+=n*(n-1)//2
    result=decode_shared(choices,ContinuousGraphBank(g))
    result.update(scope='JOINT_INTERVAL_PROPOSALS_NOT_FINAL_PARENT_OUTPUT',GT_reads=0,STEP_reads=0)
    folder=dest/f'proposal_{key}';folder.mkdir(exist_ok=False)
    with (folder/'decoded.json').open('w',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    write(folder/'selected_intervals.json',dict(intervals=interval_records))
    write(folder/'prediction_receipt.json',dict(status='PURE_INPUT_PROPOSALS_FROZEN_BEFORE_EVALUATION',source=key,
        decoded_sha256=digest(folder/'decoded.json'),model_sha256=receipt['model_sha256'],
        proposed_intervals=len(choices),all_pairs_scored=total_pairs,GT_reads=0,STEP_reads=0,
        explicit_bottom_splice_exclusion_implemented=False,full_contract_pass=False,seconds=time.perf_counter()-start))
    print('PAIR_PREDICT',key,'INTERVALS',len(choices),'SECONDS',round(time.perf_counter()-start,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',choices=['train','infer'],required=True)
    ap.add_argument('--source');ap.add_argument('--resume',action='store_true');args=ap.parse_args();dest=OUT/args.run
    if args.stage=='train' and not args.resume:
        dest.mkdir(exist_ok=False)
        write(dest/'registered_protocol.json',dict(source_sha256=digest(__file__),TRAIN=list(TRAIN),DEV=['64'],epochs=EPOCHS,
            batch=BATCH,pair_chunk=CHUNK,optimizer='Adam',learning_rate=.001,seed=1031,endpoint_gate=False,
            all_pairs_each_epoch=True,final_epoch_only=True,preserve_best379=True,external_consultation_paused=True,
            rule_labels_not_fabricated=True,output_scope='interval proposals; parent assembly still required'))
    try:
        train(dest,args.resume) if args.stage=='train' else infer(dest,args.source)
    except BaseException:
        write(dest/f'{args.stage}_{args.source or "all"}_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
