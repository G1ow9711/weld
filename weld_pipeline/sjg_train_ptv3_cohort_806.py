"""PTv3 surface learning on the frozen six-case TRAIN cohort; streamed blocks."""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_train_ptv3_804 import DATA,SurfaceModel,prepare_blocks,losses,evaluate
from weld_pipeline.sjg_ptv3_801 import CONFIG


def current_input_norm(model):
    """Batch size one: point-block statistics, no persistent cross-case state."""
    for module in model.modules():
        if isinstance(module,torch.nn.BatchNorm1d):
            module.running_mean=None;module.running_var=None
            module.num_batches_tracked=None;module.track_running_stats=False
    return model


def cache_blocks(dest):
    records={r['source']:r for r in read(DATA/'all_inputs_frozen.json')['sources']}
    labels={r['source']:r for r in read(DATA/'label_summary.json')}
    cases={};training=[]
    for source in TRAIN+VALID:
        check_resources();record=records[source];teacher=labels[source]
        assert digest(DATA/record['xyz_file'])==record['xyz_sha256']
        assert digest(DATA/teacher['teacher_file'])==teacher['teacher_sha256']
        xyz=np.load(DATA/record['xyz_file'])
        with np.load(DATA/teacher['teacher_file']) as z:y={k:z[k] for k in ('instance','normal','boundary_target')}
        folder=dest/f'blocks_{source:02d}';folder.mkdir()
        blocks=prepare_blocks(xyz,y,folder);files=[]
        for bi,(block,label) in enumerate(blocks):
            file=folder/f'block_{bi:04d}.npz'
            np.savez_compressed(file,**{f'x_{k}':v for k,v in block.items()},**{f'y_{k}':v for k,v in label.items()})
            files.append(dict(path=str(file.relative_to(dest)),sha256=digest(file)))
            if source in TRAIN:training.append((source,str(file.relative_to(dest))))
        cases[str(source)]=dict(source=source,points=len(xyz),blocks=files,
            role='TRAIN' if source in TRAIN else 'VALID_DEVELOPMENT',
            input_sha256=record['xyz_sha256'],teacher_sha256=teacher['teacher_sha256'],
            mapping_sha256=digest(folder/'input_blocks.npz'))
        print('PTV3_BLOCK_CACHE',source,'points',len(xyz),'blocks',len(blocks),flush=True)
        del xyz,y,blocks;gc.collect()
    write(dest/'block_manifest.json',dict(cases=cases,train_schedule=training,arc_not_used_in_surface_training=ARC,
        all_original_points_have_exactly_one_core_block=True))
    return cases,training


def load_block(file):
    with np.load(file,allow_pickle=False) as z:
        x={k[2:]:z[k] for k in z.files if k.startswith('x_')}
        y={k[2:]:z[k] for k in z.files if k.startswith('y_')}
    return x,y


def score(model,cases,dest,weight):
    result={}
    for key,case in cases.items():
        blocks=(load_block(dest/file['path']) for file in case['blocks'])
        value=evaluate(model,blocks,weight)
        value['scope']='SOURCE_'+key+'_'+case['role']+'_SURFACE_METRICS_ONLY'
        result[key]=value
        print('SURFACE_EVAL',key,case['role'],'loss',round(value['loss']['total'],6),
            'unoriented_normal_p95',round(value['unoriented_normal_p95_deg'],4),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(806)
    protocol=dict(stage='PTV3_FIXED_SIX_CASE_SURFACE_TRAIN',train=TRAIN,valid=VALID,arc=ARC,
        epochs=20,update_limit=2000,seed=806,learning_rate=3e-4,config=CONFIG,
        normalization='current point-block statistics during both train and inference, batch_size=1',
        normalization_audit_sha256=digest(OUT/'run25/receipt.json'),
        initialization_checkpoint=str(OUT/'run23/last.pt'),initialization_sha256=digest(OUT/'run23/last.pt'),
        initialization_training_sources=[21],optimizer='new AdamW state',
        original_cohort_G=496,training_G=359,validation_G=26,arc_diagnostic_G=111,
        XYZ_only_model_input=True,STEP_only_offline_reference=True,
        full_weld_model=False,contract_pass=False,
        unimplemented=['curved primitive supervision','material identity','weld semantics/events/parent paths','E16 acceptance'],
        start_utc=utc(),source_sha256=digest(Path(__file__)),
        imported_source_sha256={n:digest(ROOT/'weld_pipeline'/n) for n in ('sjg_ptv3_801.py','sjg_train_ptv3_804.py')})
    write(dest/'registered_protocol.json',protocol);updates=completed=0
    try:
        assert read(OUT/'run25/receipt.json')['repeat_close']
        cases,schedule=cache_blocks(dest)
        protocol.update(batches_per_full_epoch=len(schedule),original_train_points=sum(cases[str(s)]['points'] for s in TRAIN),
            block_manifest_sha256=digest(dest/'block_manifest.json'))
        write(dest/'registered_protocol.json',protocol)
        check_resources();model=SurfaceModel().cuda()
        with (OUT/'run23/last.pt').open('rb') as h:state=torch.load(h,map_location='cuda')
        model.load_state_dict(state['model']);del state;current_input_norm(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        teachers=[r for r in read(DATA/'label_summary.json') if r['source'] in TRAIN]
        weight=torch.tensor(min(20.,sum(r['boundary_negative'] for r in teachers)/max(1,sum(r['boundary_positive'] for r in teachers))),device='cuda')
        write(dest/'before.json',score(model,cases,dest,weight))
        print('PTV3_COHORT_TRAIN_START six cases batches_per_epoch',len(schedule),'maximum_updates',2000,flush=True)
        with (dest/'epochs.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,21):
                start=time.perf_counter();model.train();epoch_losses=[]
                order=np.random.default_rng(806+epoch).permutation(len(schedule)).astype(np.int64)
                for batch,index in enumerate(order,1):
                    check_resources();source,file=schedule[int(index)];block,y=load_block(dest/file)
                    optimizer.zero_grad(set_to_none=True);loss,values=losses(model(block),y,weight)
                    assert torch.isfinite(loss);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
                    optimizer.step();updates+=1;epoch_losses.append(values['total'])
                    if batch==1 or batch%10==0 or batch==len(schedule) or updates==2000:
                        print('EPOCH',epoch,'/20 BATCH',batch,'/',len(schedule),'SOURCE',source,'UPDATE',updates,'/2000 LOSS',round(values['total'],6),flush=True)
                    if updates>=2000:break
                full=batch==len(schedule)
                if full:completed+=1
                metrics=score(model,cases,dest,weight)
                row=dict(epoch=epoch,complete_epoch=full,completed_epochs=completed,batches_this_epoch=batch,
                    actual_updates=updates,train_loss=float(np.mean(epoch_losses)),surface_metrics=metrics,seconds=time.perf_counter()-start)
                log.write(json.dumps(row)+'\n');log.flush();write(dest/'latest.json',row)
                with (dest/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    protocol=protocol,epoch=epoch,complete_epoch=full,actual_updates=updates),h)
                print('EPOCH_RECORDED',epoch,'complete',full,'updates',updates,flush=True)
                if updates>=2000:break
        write(dest/'receipt.json',dict(status='PTV3_SIX_CASE_REGISTERED_BUDGET_COMPLETE',end_utc=utc(),actual_updates=updates,
            completed_epochs=completed,last_epoch_complete=full,last_epoch_batches=batch,batches_per_full_epoch=len(schedule),
            after=metrics,checkpoint_sha256=digest(dest/'last.pt'),gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            full_weld_model=False,contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),actual_updates=updates,completed_epochs=completed,error=traceback.format_exc()));raise


if __name__=='__main__':main()
