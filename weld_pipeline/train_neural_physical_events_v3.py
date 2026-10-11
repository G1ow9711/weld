"""Two fixed shared deep physical-offset candidates; FIT-first development gate."""
import gc
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame, restore_probability_order
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.continuous_event_decoder import decode_event_sets
from weld_pipeline.continuous_center_modes import RESOLUTION_MM
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.forest_event_centers import build_inputs
from weld_pipeline.neural_physical_event_sets import VERSION,FEATURE_COUNT,PhysicalEventNet,infer_event_sets
from weld_pipeline.train_forest_event_centers import BASE,load_cached

OLD_DATA=Path('weld_pipeline/data/forest_event_centers_pilot166')
OLD_RUN=Path('weld_pipeline/runs/forest_event_centers_seed166')
DATA=Path('weld_pipeline/data/neural_physical_event_pilot170_v2')
OUT=Path('weld_pipeline/runs/deep_physical_events_170_v3')


def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')


def state_digest(state):
    h=hashlib.sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous().numpy()
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def prepare(config):
    receipt=json.loads((DATA/'complete.json').read_text('utf8'))
    if (not receipt['training_allowed'] or receipt['oracle_score']['truth_count']!=1221
            or receipt['oracle_score']['valid_children']!=1221 or receipt['oracle_score']['false_predictions']):
        raise ValueError('Completed unchanged direct physical representation required')
    for name in config['train']:
        folder=DATA/name
        if sha256(folder/'manifest.json')!=receipt['manifests'][name]:raise ValueError('Frozen prepared physical manifest changed')
        manifest=json.loads((folder/'manifest.json').read_text('utf8'))
        for file,digest in manifest['files'].items():
            if sha256(folder/file)!=digest:raise ValueError('Frozen prepared physical samples changed')
    dump(OUT/'reused_sampling_receipt.json',dict(data_complete_sha256=sha256(DATA/'complete.json'),
        reuses_completed170v2_sampling_without_modification=True,sources=config['train'],full_contract_pass=False))
    print('DEEP_DIRECT_FIXED_SAMPLING_VERIFIED',len(config['train']),flush=True)




def evaluation(model,mean,std,coordinates,names,folder,runtime=False):
    folder.mkdir();scores={}
    for name in names:
        if runtime:
            manifest=json.loads((BASE/name/'manifest.json').read_text('utf8'))
            piece=load_piece(manifest['source_cache_root'],verify=True)
            sequences,data=build_inputs(read_xyz_pcd(piece['record']['input'])*1000.)
        else:
            data,_,_,piece=load_cached(name);sequences=piece['sequences']
        foregrounds,event_sets=infer_event_sets(model,data,sequences,mean,std,coordinates)
        predictions=decode_event_sets(sequences,foregrounds,event_sets)
        truth=[w for w in piece['truth'] if w['type']=='horizontal' and all(s['kind']=='line' for s in w['segments'])]
        score=score_children(predictions,truth);scores[name]=score
        dump(folder/(name+'.json'),dict(score=score,predictions=predictions,event_sets=event_sets,full_contract_pass=False))
        np.savez_compressed(folder/(name+'_foregrounds.npz'),foregrounds=np.concatenate(foregrounds),lengths=np.asarray([len(a) for a in foregrounds]))
        print('DEEP_DIRECT_SCORE',folder.parent.name,folder.name,name,score['valid_children'],'/',score['truth_count'],'FP',score['false_predictions'],flush=True)
        del data,piece,sequences,foregrounds,event_sets;gc.collect()
    result=aggregate(scores);dump(folder/'summary.json',result);return result



def main():
    if OUT.exists():raise ValueError('Preserve shared physical-event trial')
    old=json.loads((OLD_RUN/'config.json').read_text('utf8'))
    proof=json.loads(Path('weld_pipeline/reports/continuous_event_decoder_169_20261002/complete.json').read_text('utf8'))
    intervention=json.loads(Path('weld_pipeline/reports/forest_center_head_interventions_168_20261002/complete.json').read_text('utf8'))
    if not proof['all_1221_original_children_represented'] or not intervention['learned_baseline_matches_all_nine_frozen_outputs']:
        raise ValueError('Completed continuous representation and causal evidence required')
    sources=[__file__,'weld_pipeline/neural_physical_event_sets.py','weld_pipeline/continuous_center_modes.py','weld_pipeline/neural_physical_event_votes.py',
        'weld_pipeline/continuous_event_decoder.py','weld_pipeline/forest_event_centers.py','weld_pipeline/near_event_votes.py',
        'weld_pipeline/small_finite_surface_events_v3.py','weld_pipeline/contract_child_evaluation.py']
    config=dict(version=VERSION,train=old['train'],held=old['held'],feature_count=FEATURE_COUNT,
        variants={'relative_context':False,'actual_xyz':True},seed=170,width=512,residual_blocks=3,epochs=240,batch_size=4096,
        optimizer='AdamW',learning_rate=3e-4,weight_decay=1e-5,gradient_clip_norm=10.,
        support_loss='source/class-balanced BCE',offset_loss='source-balanced mm SmoothL1 beta1mm',
        radius_mm=16.,out_of_support_predictions_do_not_vote=True,normalizer_fit_only_on_training_samples=True,
        coordinates_used_as_ordinary_shared_neural_inputs_not_lookup_keys=True,
        no_name_ID_fixed_coordinate_table_or_fixed_process_planes=True,
        original_data_and_protected_predictors_unchanged=True,held_is_already_read_development_not_blind=True,
        selected_before_held='eligible FIT max TP, min FP, variant lexical order',
        source_sha256={p:sha256(p) for p in sources},full_contract_pass=False)
    config.update(reuses_completed_sampling_from_failed170_without_refitting=True,
        engineering_repair='Windows permutation indices explicitly torch.long; no successful earlier optimizer steps',
        direct_continuous_centers_no_sampled_peak_bridge=True,numerical_position_resolution_mm=RESOLUTION_MM,
        direct_mode_proof_sha256=sha256('weld_pipeline/reports/direct_center_modes_171_20261002/complete.json'))
    OUT.mkdir(parents=True);dump(OUT/'config.json',config);snapshot=OUT/'source_snapshot';snapshot.mkdir()
    for path in sources:shutil.copy2(path,snapshot/Path(path).name)
    began=time.perf_counter();prepare(config)
    xs=[];ys=[];ks=[];ranges={};count=0
    for name in config['train']:
        folder=DATA/name;manifest=json.loads((folder/'manifest.json').read_text('utf8'))
        for file,digest in manifest['files'].items():
            if sha256(folder/file)!=digest:raise ValueError('Frozen physical sampling changed')
        x=np.load(folder/'features.npy',allow_pickle=False);xs.append(x)
        ys.append(np.load(folder/'targets.npy',allow_pickle=False));ks.append(np.load(folder/'known.npy',allow_pickle=False))
        ranges[name]=count,count+len(x);count+=len(x)
    base_x=np.concatenate(xs);y=np.concatenate(ys);known=np.concatenate(ks);del xs,ys,ks;gc.collect()
    weights=np.zeros_like(y)
    for channel in range(7):
        for a,b in ranges.values():
            valid=known[a:b,channel]>0
            if channel<4:
                for positive in (False,True):
                    ids=np.flatnonzero(valid & ((y[a:b,channel]>.5)==positive))+a
                    if len(ids):weights[ids,channel]=1./(2.*len(ids))
            else:
                ids=np.flatnonzero(valid)+a
                if len(ids):weights[ids,channel]=1./len(ids)
        if weights[:,channel].sum()<=0:raise ValueError('Required known role missing')
        weights[:,channel]*=len(weights)/weights[:,channel].sum()
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dump(OUT/'runtime.json',dict(device=str(device),torch_version=torch.__version__,training_rows=count,
         gpu_name=torch.cuda.get_device_name(0) if device.type=='cuda' else None))
    target=torch.from_numpy(y).to(device);weight=torch.from_numpy(weights).to(device)
    results={};initial_digest=None
    for variant,coordinates in config['variants'].items():
        random.seed(170);np.random.seed(170);torch.manual_seed(170)
        if device.type=='cuda':torch.cuda.manual_seed_all(170)
        rng=np.random.default_rng(170);model=PhysicalEventNet().to(device)
        digest=state_digest(model.state_dict())
        if initial_digest is not None and digest!=initial_digest:raise ValueError('Common initialization differs')
        initial_digest=digest
        x=base_x.copy()
        if not coordinates:x[:,-3:]=0.
        mean=x.mean(axis=0,dtype=np.float64).astype(np.float32)
        std=np.maximum(x.std(axis=0,dtype=np.float64),1e-3).astype(np.float32)
        if not coordinates:mean[-3:]=0.;std[-3:]=1.
        values=torch.from_numpy((x-mean)/std).to(device);del x
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
        folder=OUT/variant;folder.mkdir();history=[];start=time.perf_counter()
        for epoch in range(1,241):
            model.train();permutation=rng.permutation(count);loss_total=0.;steps=0
            for a in range(0,count,4096):
                ids=torch.from_numpy(permutation[a:a+4096]).to(device=device,dtype=torch.long)
                raw=model(values[ids]);yy=target[ids];ww=weight[ids]
                binary=F.binary_cross_entropy_with_logits(raw[:,:4],yy[:,:4].clamp(0.,1.),reduction='none')
                regression=F.smooth_l1_loss(raw[:,4:7]*16.,yy[:,4:7]*16.,reduction='none',beta=1.)
                loss=(binary*ww[:,:4]).mean()+(regression*ww[:,4:7]).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite physical training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
                loss_total+=float(loss.detach());steps+=1
            history.append(dict(epoch=epoch,mean_loss=loss_total/steps,steps=steps))
            if epoch%20==0 or epoch==1:
                print('DEEP_PHYSICAL_TRAINING',variant,'EPOCH',epoch,'LOSS',loss_total/steps,flush=True)
            if epoch%40==0:
                torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                    mean=mean,std=std,config=config,variant=variant,actual_xyz=coordinates,rng_state=rng.bit_generator.state),folder/'latest.pt')
                dump(folder/'history.json',history)
        checkpoint=folder/'final.pt'
        torch.save(dict(state_dict=model.state_dict(),mean=mean,std=std,config=config,
            variant=variant,actual_xyz=coordinates,epoch=240),checkpoint)
        del values;torch.cuda.empty_cache() if device.type=='cuda' else None
        fit=evaluation(model,mean,std,coordinates,config['train'],folder/'FIT')
        eligible=fit['truth_count']==1221 and fit['valid_children']/1221>.95 and fit['false_predictions']/1221<=.05
        receipt=dict(checkpoint_sha256=sha256(checkpoint),common_initialization_sha256=digest,
            actual_epochs=240,actual_optimizer_steps=240*int(np.ceil(count/4096)),fit=fit,
            eligible_on_fit=eligible,elapsed_seconds=time.perf_counter()-start,full_contract_pass=False)
        dump(folder/'complete.json',receipt);results[variant]=receipt
        del model,optimizer;gc.collect()
    eligible=[v for v,r in results.items() if r['eligible_on_fit']]
    selected=sorted(eligible,key=lambda v:(-results[v]['fit']['valid_children'],results[v]['fit']['false_predictions'],v))[0] if eligible else None
    dump(OUT/'selection_before_held.json',dict(selected=selected,criteria=config['selected_before_held'],variant_receipts={v:sha256(OUT/v/'complete.json') for v in results}))
    held=None
    if selected:
        saved=torch.load(OUT/selected/'final.pt',map_location=device);model=PhysicalEventNet().to(device);model.load_state_dict(saved['state_dict'])
        held=evaluation(model,saved['mean'],saved['std'],saved['actual_xyz'],config['held'],OUT/selected/'HELD',runtime=True)
    dump(OUT/'complete.json',dict(config_sha256=sha256(OUT/'config.json'),actual_trained_shared_deep_models=2,
        common_initialization_sha256=initial_digest,variants=results,selected_before_held=selected,held=held,
        held_read_for_this_trial=held is not None,observed_read_for_tuning=False,default_predictor_replaced=False,
        input_data_complete_sha256=sha256(DATA/'complete.json'),elapsed_seconds=time.perf_counter()-began,full_contract_pass=False))
    print('DEEP_PHYSICAL_COMPLETE',{v:(r['fit']['valid_children'],r['fit']['false_predictions']) for v,r in results.items()},'SELECTED',selected,flush=True)


if __name__=='__main__':main()
