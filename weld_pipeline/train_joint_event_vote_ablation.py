"""Paired 2x2 event localization ablation against the frozen seed151 baseline."""
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from training.labels import sha256
from weld_pipeline import train_joint_event_vote_capacity as pilot
from weld_pipeline.joint_event_votes import JointEventVotes
from weld_pipeline.typed_contact_model import Residual
from weld_pipeline.train_joint_relation_crossfit import DATA,load_data,seed_all,atomic_save

OUT=Path('weld_pipeline/runs/joint_event_vote_ablation_152')
TARGETS=Path('weld_pipeline/data/joint_event_votes_pilot151')
BASELINE=Path('weld_pipeline/runs/joint_event_vote_capacity_seed151')
VARIANTS=(('mm_loss',False,64.),('wide_context',True,1.),('wide_context_mm_loss',True,64.))


def prefix_hash(model):
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        if key.startswith('point_layers.3.'):continue
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def loss_fn(logits,target,known,weight):
    original=pilot.loss_fn(logits,target,known)
    offset=torch.nn.functional.smooth_l1_loss(logits[:,4:7],target[:,4:7],beta=.05,reduction='none')
    offset=(offset*known[:,4:7]).sum((0,2))/known[:,4:7].sum((0,2)).clamp_min(1.)
    return original+(weight-1.)*offset.mean()


def load_targets(data,names,baseline):
    manifest=json.loads((TARGETS/'manifest.json').read_text('utf8'))
    if manifest!=baseline['target_manifests']:raise ValueError('Frozen target manifest differs')
    for name in names:
        root=TARGETS;record=manifest[name]
        if sha256(DATA/name/'manifest.json')!=record['original_source_manifest_sha256']:
            raise ValueError('Original point feature source changed')
        for suffix,key in [('targets','targets_sha256'),('known','known_sha256')]:
            if sha256(root/(name+'_'+suffix+'.npy'))!=record[key]:raise ValueError('Frozen vote target/mask changed')
        piece=data[name]
        piece['vote_targets']=np.load(root/(name+'_targets.npy'),mmap_mode='r')
        piece['vote_known']=np.load(root/(name+'_known.npy'),mmap_mode='r')
        cuts=[]
        for entry in piece['horizontal']:
            ids=np.flatnonzero(piece['targets'][entry['start']:entry['stop'],3]>.25)
            if len(ids):cuts.append(dict(entry,event_indices=ids.tolist()))
        piece['cut_entries']=cuts


def model_for(wide):
    model=JointEventVotes()
    if wide:model.point_layers=nn.Sequential(*model.point_layers,Residual(model.width,32))
    return model


def main():
    if OUT.exists():raise ValueError('Preserve paired localization ablation')
    baseline=json.loads((BASELINE/'complete.json').read_text('utf8'))
    base_config=json.loads((BASELINE/'config.json').read_text('utf8'))
    if baseline['fit_gate_passed'] or sha256(BASELINE/'final.pt')!=baseline['checkpoint_sha256']:
        raise ValueError('Expected unchanged failed baseline')
    for source,digest in base_config['sources'].items():
        if sha256(source)!=digest:raise ValueError('Baseline training or inference source changed')
    names=base_config['train'];data_config=json.loads((DATA/'config.json').read_text('utf8'))
    precision=Path('weld_pipeline/reports/joint_event_vote_precision_151_20261001/complete.json')
    audit=json.loads(precision.read_text('utf8'))
    if audit['checkpoint_sha256']!=baseline['checkpoint_sha256']:raise ValueError('Precision evidence checkpoint differs')
    OUT.mkdir(parents=True)
    sources=[__file__,'weld_pipeline/train_joint_event_vote_capacity.py','weld_pipeline/joint_event_votes.py',
             'weld_pipeline/typed_contact_model.py','weld_pipeline/joint_relation_heatmaps.py']
    config=dict(base_config,trial='152_paired_loss_units_and_receptive_context',variants=VARIANTS,
        baseline_complete_sha256=sha256(BASELINE/'complete.json'),precision_audit_sha256=sha256(precision),
        target_manifest_sha256=sha256(TARGETS/'manifest.json'),sources={p:sha256(p) for p in sources},
        initialization_seed=151,sampling_seed=151,prefix_initialization_equal=True,
        baseline_local_radius_mm=42.,wide_local_radius_mm=106.,
        selection='Only strict FIT-pass variants eligible. Highest FIT TP, then lowest FP, then lexical variant name. Freeze winner before held read.',
        full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    snap=OUT/'source_snapshot';snap.mkdir()
    for source in sources:shutil.copy2(source,snap/Path(source).name)
    seed_all(151);data=load_data(dict(data_config,pieces=names));load_targets(data,names,baseline)
    seed_all(151);expected_prefix=prefix_hash(JointEventVotes())
    results={};started=time.perf_counter()
    for label,wide,weight in VARIANTS:
        run=OUT/label;run.mkdir();pilot.OUT=run
        seed_all(151);model=model_for(wide).cuda()
        actual_prefix=prefix_hash(model)
        if actual_prefix!=expected_prefix:raise ValueError('Paired prefix initialization differs')
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
        rng=np.random.default_rng(151);source_steps={n:0 for n in names};began=time.perf_counter()
        # Consume the same actual-gradient preflight batch as seed151, without an update.
        x,node,y,known=pilot.sample(data[names[0]],rng)
        loss=loss_fn(model(x,node,*data[names[0]]['torch_graph'])[:,:,64:1088],y,known,weight)
        loss.backward();optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):raise ValueError('Paired gradient preflight nonfinite')
        print('EVENT_VOTE_ABLATION_START',label,'OFFSET_WEIGHT',weight,'WIDE',wide,flush=True)
        for epoch in range(1,121):
            model.train();losses=[]
            for _ in range(64):
                name=names[int(rng.integers(len(names)))];piece=data[name]
                x,node,y,known=pilot.sample(piece,rng)
                loss=loss_fn(model(x,node,*piece['torch_graph'])[:,:,64:1088],y,known,weight)
                if not torch.isfinite(loss):raise ValueError('Nonfinite paired localization loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step();losses.append(float(loss.detach()));source_steps[name]+=1
            with (run/'training.jsonl').open('a',encoding='utf8') as stream:
                stream.write(json.dumps(dict(epoch=epoch,loss=float(np.mean(losses)),elapsed_seconds=time.perf_counter()-began))+'\n')
            if epoch%16==0 or epoch==120:
                atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                    sampling_rng=rng.bit_generator.state,source_steps=source_steps,config_sha256=sha256(OUT/'config.json')),
                    run/'latest.pt')
                print('EVENT_VOTE_ABLATION_EPOCH',label,epoch,'LOSS',round(float(np.mean(losses)),6),flush=True)
        if source_steps!=baseline['source_steps']:raise ValueError('Paired training source sampling differs')
        atomic_save(dict(model=model.state_dict(),epoch=120,variant=label,wide=wide,offset_weight=weight,
            prefix_initialization_sha256=actual_prefix,config_sha256=sha256(OUT/'config.json')),run/'final.pt')
        frozen=sha256(run/'final.pt');fit=pilot.evaluate(model,data,names,'FIT')
        result=dict(variant=label,wide=wide,offset_weight=weight,fit=fit,fit_gate_passed=fit['geometry_core_gates_pass'],
            checkpoint_sha256=frozen,prefix_initialization_sha256=actual_prefix,same_source_step_counts=True,
            source_steps=source_steps,elapsed_seconds=time.perf_counter()-began,full_contract_pass=False)
        (run/'complete.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8');results[label]=result
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
    eligible=[r for r in results.values() if r['fit_gate_passed']]
    eligible.sort(key=lambda r:(-r['fit']['valid_children'],r['fit']['false_predictions'],r['variant']))
    selected=eligible[0]['variant'] if eligible else None
    decision=dict(eligible_variants=[r['variant'] for r in eligible],selected=selected,
        selection_only_uses_FIT=True,held_sources_not_yet_read=True,observed_check_read=False,
        baseline=baseline['fit'],variant_fit={n:r['fit'] for n,r in results.items()},full_contract_pass=False)
    (OUT/'fit_selection_decision.json').write_text(json.dumps(decision,indent=2),encoding='utf8')
    held=None
    if selected:
        run=OUT/selected;pilot.OUT=run
        held_data=load_data(dict(data_config,pieces=base_config['held']))
        state=torch.load(run/'final.pt',map_location='cpu');model=model_for(state['wide']).cuda();model.load_state_dict(state['model'])
        held=pilot.evaluate(model,held_data,base_config['held'],'POST_FIT_HELD')
        del model;torch.cuda.empty_cache()
    unchanged=sha256(BASELINE/'final.pt')==baseline['checkpoint_sha256'] and all(
        sha256(OUT/n/'final.pt')==r['checkpoint_sha256'] for n,r in results.items())
    if not unchanged:raise ValueError('Frozen baseline or new weights changed during evaluation')
    complete=dict(config_sha256=sha256(OUT/'config.json'),baseline=baseline['fit'],variants=results,
        selected=selected,held=held,held_status='FIXED_POST_FIT_SELECTED_VARIANT' if selected else 'NOT_RUN_ALL_FIT_GATES_FAILED',
        models_completed=3,elapsed_seconds=time.perf_counter()-started,all_weights_unchanged=True,
        observed_check_read=False,three_seed_source_adoption_not_validated=True,full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(complete,indent=2,ensure_ascii=False),encoding='utf8')
    print('EVENT_VOTE_ABLATION_COMPLETE',{n:(r['fit']['valid_children'],r['fit']['false_predictions']) for n,r in results.items()},
          'SELECTED',selected,flush=True)


if __name__=='__main__':main()
