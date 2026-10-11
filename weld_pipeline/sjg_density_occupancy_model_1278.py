"""Fit a fresh native-supervised auxiliary occupancy model, never DEV labels.

The model supplies probabilities to interval features. It does not output final
welds or claim parent, endpoint, attribute, or contract acceptance.
"""
import argparse
import gc
import shutil
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources,resources,reclaim_own_idle_pages
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_lossless_training_buffers_1268 import LosslessRows
from weld_pipeline.sjg_dense_joint_sequence_1237 import ordered_features
from weld_pipeline.sjg_density_feature_runtime_1277 import DensityView,controls as density_controls
from weld_pipeline.sjg_structured_sequence_1241 import SymmetricBlock,pointwise

TRAIN=('21','BM79A_native_local','10','25','36')
EPOCHS=20
BATCH=8
INPUT='run421/sequence56'
ARCH='DENSITY_AUGMENTED_NATIVE_OCCUPANCY_TCN_V1'


class OccupancyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Conv1d(56,32,1);self.context=nn.Conv1d(64,32,1)
        self.blocks=nn.ModuleList([SymmetricBlock(32,2**i) for i in range(10)])
        self.occupancy=nn.Conv1d(32,1,1)

    def forward(self,x,lengths,closed):
        mask=(torch.arange(x.shape[2],device=x.device)[None,:]<lengths[:,None])[:,None,:]
        h=F.silu(pointwise(self.input,x))*mask
        mean=h.sum(2,keepdim=True)/lengths[:,None,None]
        maximum=h.masked_fill(~mask,-torch.inf).amax(2,keepdim=True)
        h=(h+F.silu(pointwise(self.context,torch.cat((mean,maximum),1))))*mask
        for block in self.blocks:h=block(h,lengths,closed,mask)
        return pointwise(self.occupancy,h)[:,0]


def objective(logits,b):
    p=logits.sigmoid();known=b['known'];truth=b['y']
    bce=(F.binary_cross_entropy_with_logits(logits,truth,reduction='none')*b['w']).sum()
    dice=1-(2*(p*truth*known).sum(1)+1e-6)/((p*known).sum(1)+(truth*known).sum(1)+1e-6)
    return bce+(dice*b['dw']).sum()


def controls(model,dest):
    model.eval();rng=torch.Generator().manual_seed(719)
    x=torch.randn(2,56,37,generator=rng);lengths=torch.tensor([37,37]);closed=torch.tensor([False,True])
    with torch.no_grad():
        a=model(x,lengths,closed);reverse=model(x.flip(2),lengths,closed).flip(1)
        padded=model(F.pad(x,(0,17)),lengths,closed)[:,:37]
    reversal=float((a-reverse).abs().max());padding=float((a-padded).abs().max())
    assert reversal<2e-5 and padding<2e-5
    logits=torch.tensor([[.2,-.3,.8]],requires_grad=True)
    b=dict(y=torch.tensor([[1.,0.,0.]]),known=torch.tensor([[True,False,True]]),
           w=torch.tensor([[.25,0.,.25]]),dw=torch.tensor([.5]))
    v=objective(logits,b);v.backward();assert logits.grad[0,1].item()==0.
    other=dict(b,y=torch.tensor([[1.,1.,0.]]));assert torch.equal(v,objective(logits,other))
    padded_b=dict(y=F.pad(b['y'],(0,2)),known=F.pad(b['known'],(0,2)),w=F.pad(b['w'],(0,2)),dw=b['dw'])
    assert torch.equal(v,objective(F.pad(logits,(0,2)),padded_b))
    write(dest/'architecture_controls.json',dict(status='PASS',reversal_max_abs=reversal,padding_max_abs=padding,
        uncertain_target_no_loss_or_gradient=True,padding_no_loss=True,boundary_head_not_used=True))


def load_sequence(key):
    folder=OUT/INPUT/f'source_{key}';r=read(folder/'receipt.json')
    assert r['GT_reads']==0 and digest(folder/'features.npy')==r['features_sha256']
    assert digest(folder/'sequence_map.npz')==r['map_sha256']
    x=np.load(folder/'features.npy')
    assert x.ndim==2 and x.shape[1]==56 and np.isfinite(x).all()
    with np.load(folder/'sequence_map.npz') as z:m={k:z[k] for k in z.files}
    return x,m,r


def prepare(dest):
    audit=read(OUT/'run428/independent_audit.json');assert audit['status']=='PASS' and audit['original_G']==269
    manifest=read(OUT/'run428/receipt.json');assert manifest['status']=='NATIVE_OCCUPANCY_TARGETS_AUDITED_READY_NO_FIT'
    records=[];sources={};bindings={};sums=np.zeros(56);squares=np.zeros(56);count=0;mass=[]
    for c in manifest['cases']:
        key=c['source'];assert key in TRAIN;folder=OUT/f'run428/source_{key}'
        assert digest(folder/'occupancy_targets.npz')==c['labels_sha256']
        assert digest(folder/'owner_registry.json')==c['owner_registry_sha256']
        print('LOAD_NATIVE_OCCUPANCY_TRAIN',key,flush=True)
        x,m,sr=load_sequence(key)
        assert sr['features_sha256']==c['sequence_features_sha256'] and sr['map_sha256']==c['sequence_map_sha256']
        with np.load(folder/'occupancy_targets.npz') as z:rows=z['rows'];offsets=z['offsets'];labels=z['labels']
        registry=read(folder/'owner_registry.json');owners={tuple(o['owner']):o for o in registry['owners']}
        assert len(owners)==c['G'] and all(o['certain_positive_sites']>0 for o in owners.values())
        windows={r['row']:r for r in registry['windows']};lookup={int(p):i for i,p in enumerate(m['profile_ids'])}
        positive_mass=negative_mass=0.
        for i,row in enumerate(rows):
            row=int(row);j=lookup[row//2];a,b=map(int,m['offsets'][j:j+2]);grid=m['q_mm'][a:b]
            y=labels[offsets[i]:offsets[i+1]].copy();assert len(y)==b-a
            w=np.zeros(len(y),np.float64)
            for owner,window in zip(windows[row]['owners'],windows[row]['windows']):
                core=(grid>=window[1])&(grid<=window[2]);assert np.all(y[core]==1)
                w[core]+=1/(2*len(TRAIN)*c['G']*owners[tuple(owner)]['certain_positive_sites'])
            w[y==0]=1/(2*len(TRAIN)*c['negative_sites'])
            assert np.all(w[y<0]==0) and np.all(w[y>=0]>0)
            pm=float(w[y==1].sum());positive_mass+=pm;negative_mass+=float(w[y==0].sum())
            raw=ordered_features(x[a:b],row%2).astype(np.float64)
            sums+=raw.sum(0);squares+=(raw*raw).sum(0);count+=len(raw)
            records.append(dict(source=key,row=row,a=a,b=b,y=np.maximum(y,0).astype(np.float32),
                known=y>=0,w=w.astype(np.float32),dice_weight=pm,closed=bool(x[a,27]>.5)))
        assert abs(positive_mass-.1)<1e-8 and abs(negative_mass-.1)<1e-8
        packed=LosslessRows(x)
        for first in range(0,len(x),8192):assert np.array_equal(packed[first:first+8192],x[first:first+8192])
        sources[key]=packed;bindings[key]=dict(features_sha256=sr['features_sha256'],map_sha256=sr['map_sha256'],
            labels_sha256=c['labels_sha256'],owner_registry_sha256=c['owner_registry_sha256'])
        mass.append(dict(source=key,positive_weight=positive_mass,negative_weight=negative_mass,raw_bytes=x.nbytes,
            buffered_bytes=packed.stored_bytes,all_input_buffer_slices_exact=True))
        print('NATIVE_OCCUPANCY_BUFFER_READY',key,'ROWS',len(rows),'BYTES',packed.stored_bytes,flush=True)
        del x,raw;gc.collect();check_resources()
    assert set(sources)==set(TRAIN)
    mean=sums/count;scale=np.sqrt(np.maximum(0,squares/count-mean*mean));scale[scale<1e-6]=1.
    mean=mean.astype(np.float32);scale=scale.astype(np.float32)
    write(dest/'supervision_receipt.json',dict(TRAIN=list(TRAIN),G=269,sequences=len(records),point_rows=count,
        bindings=bindings,weight_and_buffer_audit=mass,normalization_fit_on_current_TRAIN_only=True,
        positive_child_mass_equal_within_each_case=True,uncertain_boundary_sites_excluded_from_loss=True,
        DEV_reads=0,STEP_reads=0,full_contract_pass=False))
    return records,sources,mean,scale,bindings


def batch(records,sources,mean,scale,device,density_views=None):
    n=len(records);size=max(r['b']-r['a'] for r in records)
    arrays=dict(x=np.zeros((n,56,size),np.float32),y=np.zeros((n,size),np.float32),
        w=np.zeros((n,size),np.float32),known=np.zeros((n,size),bool))
    for i,r in enumerate(records):
        length=r['b']-r['a'];raw=ordered_features(sources[r['source']][r['a']:r['b']],r['row']%2)
        if r.get('density_half',False):
            assert density_views is not None
            raw=density_views[r['source']].sequence(raw,r['a'],r['b'])
        arrays['x'][i,:,:length]=((raw-mean)/scale).T
        for field in ('y','w','known'):arrays[field][i,:length]=r[field]
    result={k:torch.from_numpy(v).to(device) for k,v in arrays.items()}
    result.update(lengths=torch.tensor([r['b']-r['a'] for r in records],device=device),
        closed=torch.tensor([r['closed'] for r in records],device=device),
        dw=torch.tensor([r['dice_weight'] for r in records],device=device))
    return result



def augmentation_batch_controls(records,sources,mean,scale,views,dest):
    selected=[next(r for r in records if r['source']==key) for key in TRAIN]
    original=batch(selected,sources,mean,scale,'cpu')
    augmented=batch([dict(r,density_half=True) for r in selected],sources,mean,scale,'cpu',views)
    retained=np.r_[0:28,43:56]
    assert torch.equal(original['x'][:,retained],augmented['x'][:,retained])
    for key in original:
        if key!='x':assert torch.equal(original[key],augmented[key]),key
    deltas=[]
    for i,r in enumerate(selected):
        n=r['b']-r['a'];raw=ordered_features(sources[r['source']][r['a']:r['b']],r['row']%2)
        saved=raw.copy();expected=views[r['source']].sequence(raw,r['a'],r['b'])
        assert np.array_equal(raw,saved)
        assert torch.equal(augmented['x'][i,:,:n],torch.from_numpy(((expected-mean)/scale).T.copy()))
        delta=float((original['x'][i]-augmented['x'][i]).abs().max());assert delta>0
        deltas.append(dict(source=r['source'],normalized_max_difference=delta))
    # Resuming the generator must reproduce the next augmentation choices.
    rng=np.random.default_rng(1031);rng.integers(0,2,37);state=rng.bit_generator.state
    expected=rng.integers(0,2,81);restored=np.random.default_rng();restored.bit_generator.state=state
    assert np.array_equal(expected,restored.integers(0,2,81))
    write(dest/'augmentation_batch_controls.json',dict(status='PASS',sources=deltas,
        unchanged_sequence_columns=41,labels_weights_masks_lengths_closed_unchanged=True,
        original_arrays_not_mutated=True,rng_restore_reproduces_choices=True,DEV_reads=0))


def resource_gate(dest,stage):
    if resources()['available_physical_bytes']<1342177280:
        write(dest/f'own_trim_{stage}.json',reclaim_own_idle_pages())
    return check_resources()


def train(dest,resume=False):
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(1031)
    torch.backends.cudnn.enabled=False;torch.backends.cuda.matmul.allow_tf32=False
    model=OccupancyModel();controls(model,dest)
    records,sources,mean,scale,bindings=prepare(dest)
    density_views={key:DensityView(key) for key in TRAIN}
    for key,view in density_views.items():
        assert view.receipt['map_sha256']==bindings[key]['map_sha256']
        bindings[key]['density_view_sha256']=view.receipt['alternate_sha256']
    write(dest/'density_controls.json',density_controls())
    augmentation_batch_controls(records,sources,mean,scale,density_views,dest)
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(1031)
    trace=[];first_epoch=1;elapsed=0.
    if resume:
        state=torch.load(dest/'latest_training_state.pt',map_location=device)
        assert state['bindings']==bindings and np.array_equal(state['mean'],mean) and np.array_equal(state['scale'],scale)
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer']);rng.bit_generator.state=state['rng']
        trace=state['trace'];first_epoch=state['epoch']+1;elapsed=trace[-1]['seconds']
        write(dest/'resume_receipt.json',dict(next_epoch=first_epoch,optimizer_rng_and_normalization_restored=True,
            checkpoint_sha256=digest(dest/'latest_training_state.pt')))
    resource_gate(dest,'before_epochs')
    print('NATIVE_OCCUPANCY_TRAIN',device,'SEQUENCES',len(records),'EPOCHS',EPOCHS,flush=True)
    for epoch in range(first_epoch,EPOCHS+1):
        model.train();ep=time.perf_counter();ids=rng.permutation(len(records));groups=[]
        for lo in range(0,len(ids),BATCH*16):
            bucket=sorted(ids[lo:lo+BATCH*16],key=lambda i:records[i]['b']-records[i]['a'])
            groups.extend(bucket[j:j+BATCH] for j in range(0,len(bucket),BATCH))
        rng.shuffle(groups);value=0.;seen=0;half_seen=0
        for bi,group in enumerate(groups):
            chosen=[dict(records[i],density_half=bool(rng.integers(0,2))) for i in group]
            half_seen+=sum(r['density_half'] for r in chosen)
            b=batch(chosen,sources,mean,scale,device,density_views)
            optimizer.zero_grad(set_to_none=True)
            v=objective(model(b['x'],b['lengths'],b['closed']),b)*len(records)/len(group)
            assert torch.isfinite(v);v.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
            optimizer.step();value+=float(v.detach())*len(group)/len(records);seen+=len(group)
            if (bi+1)%100==0:print('EPOCH_BATCH',epoch,'/',EPOCHS,'SEQUENCES',seen,'/',len(records),flush=True)
        assert seen==len(records)
        trace.append(dict(epoch=epoch,loss=value,sequences_seen=seen,density_half_sequences=half_seen,original_sequences=seen-half_seen,epoch_seconds=time.perf_counter()-ep,seconds=elapsed+time.perf_counter()-start))
        atomic_binary_write(dest/'latest_training_state.pt',lambda f:torch.save(dict(state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),epoch=epoch,rng=rng.bit_generator.state,trace=trace,mean=mean,scale=scale,bindings=bindings),f))
        write(dest/'training_progress.json',dict(trace=trace,total_epochs=EPOCHS))
        print('EPOCH',epoch,'/',EPOCHS,'LOSS',round(value,6),'SECONDS',round(trace[-1]['epoch_seconds'],2),flush=True)
        del b,v;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        resource_gate(dest,f'epoch_{epoch}')
    model.cpu().eval();controls(model,dest)
    payload=dict(state_dict=model.state_dict(),mean=mean,scale=scale,epochs=EPOCHS,input_run=INPUT,architecture=ARCH)
    atomic_binary_write(dest/'sequence_model.pt',lambda f:torch.save(payload,f))
    reloaded=OccupancyModel().eval();reloaded.load_state_dict(torch.load(dest/'sequence_model.pt',map_location='cpu')['state_dict'])
    b=batch(records[:2],sources,mean,scale,'cpu')
    with torch.no_grad():assert torch.equal(model(b['x'],b['lengths'],b['closed']),reloaded(b['x'],b['lengths'],b['closed']))
    write(dest/'training_receipt.json',dict(status='NATIVE_AUXILIARY_OCCUPANCY_TRAINED',TRAIN=list(TRAIN),G=269,
        epochs=EPOCHS,trace=trace,input_run=INPUT,architecture=ARCH,model_sha256=digest(dest/'sequence_model.pt'),
        bindings=bindings,density_augmentation_run='run436',augmentation_rng_checkpointed=True,reload_exact=True,DEV_reads=0,STEP_reads=0,final_epoch_only=True,
        auxiliary_probabilities_not_final_welds=True,seconds=elapsed+time.perf_counter()-start,full_contract_pass=False))
    print('NATIVE_OCCUPANCY_TRAIN_COMPLETE',flush=True)


def infer(dest,key):
    start=time.perf_counter();torch.set_num_threads(2);r=read(dest/'training_receipt.json')
    assert digest(dest/'sequence_model.pt')==r['model_sha256']
    p=torch.load(dest/'sequence_model.pt',map_location='cpu');assert p['architecture']==ARCH
    model=OccupancyModel().eval();model.load_state_dict(p['state_dict'])
    raw,m,ir=load_sequence(key);lookup={int(pid):i for i,pid in enumerate(m['profile_ids'])}
    probabilities=[];offsets=[0]
    for ri,row in enumerate(m['admitted_rows']):
        row=int(row);j=lookup[row//2];a,b=map(int,m['offsets'][j:j+2]);x=ordered_features(raw[a:b],row%2)
        x=(x-p['mean'])/p['scale']
        with torch.no_grad():
            logits=model(torch.from_numpy(x.T.copy())[None],torch.tensor([len(x)]),torch.tensor([bool(raw[a,27]>.5)]))
            probability=logits[0].sigmoid().numpy()
        assert np.isfinite(probability).all();probabilities.append(probability);offsets.append(offsets[-1]+len(x))
        if (ri+1)%200==0:print('NATIVE_OCCUPANCY_INFER',key,ri+1,'/',len(m['admitted_rows']),flush=True)
    folder=dest/f'proposal_{key}';folder.mkdir(exist_ok=False)
    np.savez_compressed(folder/'occupancy.npz',admitted_rows=m['admitted_rows'],offsets=np.asarray(offsets),probabilities=np.concatenate(probabilities))
    write(folder/'prediction_receipt.json',dict(status='PURE_INPUT_AUXILIARY_OCCUPANCY_FROZEN',source=key,
        model_sha256=r['model_sha256'],occupancy_sha256=digest(folder/'occupancy.npz'),features_sha256=ir['features_sha256'],
        map_sha256=ir['map_sha256'],admitted_carriers=len(m['admitted_rows']),GT_reads=0,STEP_reads=0,
        auxiliary_probabilities_not_final_welds=True,inference_uses_original_features_only=True,seconds=time.perf_counter()-start,full_contract_pass=False))
    print('NATIVE_OCCUPANCY_INFER_COMPLETE',key,round(time.perf_counter()-start,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['train','infer'],required=True);ap.add_argument('--source');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    dest=OUT/'run437'
    if a.stage=='train' and not a.resume:
        dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir();bindings={}
        for name in ('sjg_density_occupancy_model_1278.py','sjg_density_feature_runtime_1277.py','sjg_structured_sequence_1241.py','sjg_lossless_training_buffers_1268.py','sjg_dense_joint_sequence_1237.py'):
            src=ROOT/'weld_pipeline'/name;shutil.copy2(src,snapshot/name);bindings[name]=digest(src)
        write(dest/'registered_protocol.json',dict(TRAIN=list(TRAIN),DEV=['64'],G=269,module_sha256=bindings,
            input_run=INPUT,supervision_run='run428',independent_audit_sha256=digest(OUT/'run428/independent_audit.json'),
            density_augmentation_run='run436',density_manifest_sha256=digest(OUT/'run436/receipt.json'),
            augmentation='per-sequence Bernoulli 0.5; replace only dimensional kNN columns 28:43; original targets and masks',
            normalization='fresh fit on original TRAIN features only; same transform for augmented views',
            epochs=EPOCHS,batch=BATCH,seed=1031,optimizer='fresh Adam',learning_rate=.001,
            initialization='fresh deterministic weights; no run392 weights or normalization reused',
            loss='case and native-child balanced masked BCE plus masked Dice',boundary_uncertain_mask=-1,
            final_epoch_only=True,all_known_sequences_every_epoch=True,keep_best379=True,
            external_consultation_paused=True,auxiliary_probabilities_not_final_welds=True,full_contract_pass=False))
    protocol=read(dest/'registered_protocol.json')
    assert digest(OUT/'run436/receipt.json')==protocol['density_manifest_sha256']
    for name,sha in protocol['module_sha256'].items():assert digest(ROOT/'weld_pipeline'/name)==sha
    try:
        if a.stage=='train':train(dest,a.resume)
        else:
            assert a.source in (*TRAIN,'64');infer(dest,a.source)
    except BaseException:
        write(dest/f'failure_{a.stage}_{a.source or "all"}.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
