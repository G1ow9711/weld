"""Native XYZ PTv3 multitask learning with dense face and material losses."""
import argparse
import gc
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources,resources,reclaim_own_idle_pages
from weld_pipeline.sjg_ptv3_801 import voxelize
from weld_pipeline.sjg_material_model_859 import SurfaceMaterialModel
from weld_pipeline.sjg_dense_material_loss_863 import dense_material_loss
from weld_pipeline.sjg_local_event_train_930 import save_checkpoint


def load_record(record):
    with np.load(OUT/record['input_file']) as z:xyz=z['xyz_mm']
    block=voxelize(xyz)
    with np.load(OUT/record['label_file']) as z:y={k:z[k] for k in z.files}
    assert len(y['core'])==len(xyz) and int(y['core'].sum())==record['core_count']
    return block,y


def objective(output,y):
    surface,normal,boundary,material=output;device=surface.device
    core=torch.as_tensor(y['core'],device=device,dtype=torch.bool)
    face=torch.as_tensor(y['instance'],device=device,dtype=torch.long)
    owner=torch.as_tensor(y['reference_volume'],device=device,dtype=torch.long)
    face_loss,face_stats=dense_material_loss(surface,face,core)
    material_loss,material_stats=dense_material_loss(material,owner,core)
    mask=(face>=0)&core;zero=normal.sum()*0
    truth=torch.as_tensor(y['normal'],device=device,dtype=normal.dtype)
    normal_loss=(1-(normal[mask]*truth[mask]).sum(-1).abs()).mean() if mask.any() else zero
    by=torch.as_tensor(y['boundary_target'],device=device,dtype=boundary.dtype);bm=(by>=0)&core
    boundary_loss=F.binary_cross_entropy_with_logits(boundary[bm],by[bm]) if bm.any() else zero
    total=face_loss+normal_loss+.5*boundary_loss+material_loss
    stats=dict(face=float(face_loss.detach()),normal=float(normal_loss.detach()),boundary=float(boundary_loss.detach()),
        material=float(material_loss.detach()),core_points=int(core.sum()),face_points=face_stats['sampled_points'],
        material_points=material_stats['sampled_points'])
    return total,stats


def controls():
    # The same objective works on CPU; every known core point must receive a
    # material gradient, while unknown/context points must receive none.
    material=torch.tensor([[0.,0.],[1.,0.],[3.,0.],[4.,0.],[8.,0.]],requires_grad=True)
    labels=torch.tensor([0,0,1,1,-1]);core=torch.tensor([True,True,True,True,True])
    loss,stats=dense_material_loss(material,labels,core);loss.backward()
    assert stats['sampled_points']==4 and torch.all(material.grad[:4].abs().sum(1)>0) and not torch.any(material.grad[4])
    return dict(all_known_core_points_supervised=True,unknown_material_no_gradient=True,no_fixed_64_point_cap=True)


def guard(dest):
    if resources()['available_physical_bytes']<1610612736:
        report=reclaim_own_idle_pages();write(dest/'latest_own_page_reclaim.json',dict(**report,time_utc=utc()))
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            try:return check_resources()
            except RuntimeError:time.sleep(1)
    return check_resources()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--data-run',default='run161');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);data=OUT/args.data_run;torch.set_num_threads(2);torch.manual_seed(966)
    assert read(data/'receipt.json')['status']=='NATIVE_STRUCTURAL_TRAIN_BLOCKS_READY'
    manifest=read(data/'manifest.json');records=manifest['records'];data_receipt=read(data/'receipt.json');epochs=2;updates=0;completed=0;history=[]
    expected=epochs*len(records)
    protocol=dict(stage='NATIVE_PTV3_FINITE_FACE_BOUNDARY_MATERIAL_TRAINING',data_run=args.data_run,
        data_manifest_sha256=digest(data/'manifest.json'),initial_surface='run26/last.pt',initial_sha256=digest(OUT/'run26/last.pt'),
        epochs=epochs,expected_updates=expected,seed=966,learning_rate=.0001,weight_decay=.01,optimizer='AdamW',
        face_and_material_all_known_core_points=True,boundary_pos_weight=1.,normal_unoriented=True,
        original_points_per_epoch=data_receipt['core_points'],known_face_core_per_epoch=data_receipt['known_face_core'],
        known_material_core_per_epoch=data_receipt['known_material_core'],
        input='native XYZ only; no teacher IDs in model',teacher_is_offline_conditional_reference=True,
        sources=[s['source_key'] for s in manifest['sources']],BM_weld_training_still_retained=True,
        dev64_and_diagnostic24_not_read=True,original_eight_G=496,full_contract_pass=False,
        warmup_forward_backward_without_optimizer_step=True,
        source_sha256=digest(__file__),model_sha256=digest(ROOT/'weld_pipeline/sjg_material_model_859.py'),start_utc=utc())
    write(dest/'registered_protocol.json',protocol);write(dest/'controls.json',controls())
    write(dest/'previous_round_local_review.json',dict(failed_round='run155',scores=read(OUT/'run155/receipt.json')['scores'],
        finite_boundary_diagnosis='run157',training_label_binding=args.data_run,
        change='Train finite face instances, finite boundaries and material owners from native all-point teachers, rather than continuing the failed event-only weight variants.',
        external_consultation_sent=False,full_contract_gates_retained=True))
    try:
        for record in records:
            assert digest(OUT/record['input_file'])==record['input_sha256'] and digest(OUT/record['label_file'])==record['label_sha256']
        model=SurfaceMaterialModel()
        with (OUT/'run26/last.pt').open('rb') as stream:state=torch.load(stream,map_location='cpu')
        model.load_surface(state['model']);del state
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        save_checkpoint(dict(model=initial,updates=0,protocol=protocol),dest/'initial.pt')
        model=model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01);torch.cuda.reset_peak_memory_stats()
        # Exercise forward/backward before the first update, then release this
        # worker's pageable initialization memory under the unchanged guard.
        warm,y=load_record(records[0]);loss,_=objective(model(warm),y);loss.backward();optimizer.zero_grad(set_to_none=True)
        del warm,y,loss;gc.collect();torch.cuda.empty_cache();guard(dest)
        started=time.perf_counter()
        for epoch in range(1,epochs+1):
            model.train();totals=dict(core_points=0,face_points=0,material_points=0);values=[]
            order=np.random.default_rng(966+epoch).permutation(len(records))
            for batch,index in enumerate(order,1):
                guard(dest);record=records[int(index)];block,y=load_record(record)
                optimizer.zero_grad(set_to_none=True);output=model(block);loss,stats=objective(output,y)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite native structural loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();updates+=1
                for k in totals:totals[k]+=stats[k]
                values.append([stats[k] for k in ('face','normal','boundary','material')])
                if batch==1 or batch%20==0 or batch==len(records):
                    row=dict(epoch=epoch,epochs=epochs,block=batch,blocks=len(records),updates=updates,expected_updates=expected,
                        source=record['source_key'],loss=float(loss.detach()),**totals,elapsed_seconds=time.perf_counter()-started)
                    write(dest/'progress.json',row);print('EPOCH',epoch,'/',epochs,'BLOCK',batch,'/',len(records),'UPDATES',updates,'/',expected,'LOSS',round(row['loss'],6),flush=True)
                if batch%200==0:
                    path=dest/'recovery.pending.pt';save_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=updates,
                        epoch=epoch,next_block=batch,totals=totals,torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
                        protocol=protocol,partial_epoch=True),path);path.replace(dest/'recovery.pt')
                del block,y,output,loss
            assert totals['core_points']==data_receipt['core_points']
            assert totals['face_points']==data_receipt['known_face_core'] and totals['material_points']==data_receipt['known_material_core']
            completed=epoch;history.append(dict(epoch=epoch,updates=updates,**totals,mean_losses=np.mean(values,0).tolist()))
            write(dest/'history.json',history)
            path=dest/'last.pending.pt';save_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,
                protocol=protocol,completed_full_epochs=completed,torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all()),path);path.replace(dest/'last.pt')
            print('NATIVE_FULL_EPOCH_COMPLETE',history[-1],flush=True)
        assert updates==expected
        delta=sum(float((v.detach().cpu()-initial[k]).double().square().sum()) for k,v in model.state_dict().items())**.5
        assert delta>0
        write(dest/'receipt.json',dict(status='NATIVE_PTV3_STRUCTURAL_TRAINING_COMPLETE',completed_full_epochs=completed,
            actual_optimizer_updates=updates,parameter_change_l2=delta,history=history,
            checkpoint_sha256=digest(dest/'last.pt'),gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            new_extraction_accuracy_not_measured=True,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',actual_optimizer_updates=updates,completed_full_epochs=completed,
            error=traceback.format_exc(),resources=resources(),full_contract_pass=False,end_utc=utc()));raise

if __name__=='__main__':main()
