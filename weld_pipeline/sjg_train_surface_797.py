"""Train surface validity from pure-XYZ features with offline finite-face labels."""
import argparse
import json
import math
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.sjg_geometry_776 import read, write, digest
from weld_pipeline.sjg_prepare_776 import ROOT, OUT, TRAIN, VALID, ARC
from weld_pipeline.sjg_surface_learning_796 import GEOMETRY,input_features,offline_surface_labels
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_xyz_surface_794 import geometry_support


class SurfaceValidity(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim))
        self.net=nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,x):return self.net((x-self.mean)/self.scale).squeeze(-1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run17');args=ap.parse_args()
    out=OUT/args.run;out.mkdir(exist_ok=False);updates=0;started=time.monotonic()
    try:
        check_resources();torch.set_num_threads(2);torch.manual_seed(2026100797)
        assert (GEOMETRY/'receipt.json').exists(),'Geometry generation must finish first'
        inventory=read(OUT/'inventory.json');data={};feature_freeze=[]
        for si in TRAIN+VALID+ARC:
            print('PREPARE_PURE_XYZ_SURFACE_FEATURES',si,'NO_EPOCH_PREPARATION',flush=True)
            x,binding=input_features(si,inventory);data[si]=x
            file=out/f'source_{si:02d}_features.npy';np.save(file,x)
            feature_freeze.append(dict(source=si,path=file.name,sha256=digest(file),surfaces=len(x),
                input_geometry_freeze_sha256=digest(GEOMETRY/f'source_{si:02d}/input_freeze.json')))
            check_resources()
        write(out/'feature_freeze.json',dict(outputs=feature_freeze,STEP_reads_during_features=0,
            weld_GT_reads_during_features=0,all_features_frozen_before_offline_teacher_labels=True))
        labels={};label_summary={}
        for si in TRAIN+VALID:
            teacher=offline_surface_labels(si,inventory);assert teacher is not None
            labels[si]=teacher['labels'];np.savez_compressed(out/f'source_{si:02d}_teacher.npz',
                labels=teacher['labels'],finite_reference_fraction=teacher['finite_reference_fraction'],
                multiple_material_owners=teacher['multiple_material_owners'])
            label_summary[str(si)]=dict(positive=int((teacher['labels']==1).sum()),
                negative=int((teacher['labels']==0).sum()),unknown=int((teacher['labels']<0).sum()),
                multiple_material_hypotheses=int(teacher['multiple_material_owners'].sum()),
                reference_sha256=teacher['reference_sha256'])
            print('OFFLINE_SURFACE_LABELS',si,label_summary[str(si)],flush=True)
        write(out/'offline_label_summary.json',label_summary)
        x=np.concatenate([data[i][labels[i]>=0] for i in TRAIN])
        y=np.concatenate([labels[i][labels[i]>=0] for i in TRAIN])
        assert y.sum()>0 and (y==0).sum()>0
        batch=128;batches=math.ceil(len(x)/batch);epochs=min(100,2000//batches);assert epochs>=1
        protocol=dict(stage='PURE_XYZ_SURFACE_VALIDITY_PRETRAINING',train=TRAIN,holdout=VALID,
            unlabelled_curve_workpiece=ARC,seed=2026100797,epochs=epochs,batch=batch,batches_per_epoch=batches,
            maximum_updates=2000,training_examples=len(x),positive_examples=int(y.sum()),
            optimizer='AdamW',learning_rate=.001,threshold=.5,feature_dim=x.shape[1],
            features_from_XYZ_only=True,STEP_used_only_as_offline_teacher=True,
            no_source_name_or_absolute_location_features=True,material_identity_not_predicted=True,
            finite_support_fraction_positive=.9,finite_support_fraction_negative=.1,
            mixed_support_excluded_only_from_plane_training_not_weld_denominator=True,
            source_sha256=digest(__file__),feature_source_sha256=digest(ROOT/'weld_pipeline/sjg_surface_learning_796.py'),
            full_contract_pass=False,external_messages=0)
        write(out/'registered_protocol.json',protocol)
        model=SurfaceValidity(x.shape[1]).cuda()
        model.mean.copy_(torch.from_numpy(x.mean(0)).cuda());model.scale.copy_(torch.from_numpy(x.std(0).clip(.01)).cuda())
        xx=torch.from_numpy(x).cuda();yy=torch.from_numpy(y).cuda()
        weight=torch.tensor(float((len(y)-y.sum())/y.sum()),device='cuda')
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        rng=np.random.default_rng(protocol['seed'])
        with (out/'epochs.jsonl').open('w',encoding='utf8') as log:
            for epoch in range(1,epochs+1):
                order=rng.permutation(len(x));loss_sum=0.
                for begin in range(0,len(x),batch):
                    ids=torch.as_tensor(order[begin:begin+batch],dtype=torch.long,device='cuda')
                    loss=F.binary_cross_entropy_with_logits(model(xx[ids]),yy[ids],pos_weight=weight)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite surface loss')
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                    updates+=1;loss_sum+=float(loss.detach())
                status=dict(utc=utc(),epoch=epoch,epochs=epochs,updates=updates,complete_epoch=True,loss=loss_sum/batches)
                log.write(json.dumps(status)+'\n');log.flush();write(out/'progress.json',status)
                print('XYZ_SURFACE_TRAIN Epoch',f'{epoch}/{epochs}','Updates',updates,'Loss',round(status['loss'],6),flush=True)
                if epoch%10==0:check_resources()
        with (out/'surface_validity.pt').open('wb') as f:torch.save(dict(model=model.state_dict(),dim=x.shape[1],protocol=protocol),f)
        model.eval();predictions=[];probabilities={}
        for si in TRAIN+VALID+ARC:
            with torch.no_grad():p=model(torch.from_numpy(data[si]).cuda()).sigmoid().cpu().numpy()
            probabilities[si]=p;file=out/f'source_{si:02d}_probabilities.npy';np.save(file,p)
            predictions.append(dict(source=si,path=file.name,sha256=digest(file)))
        write(out/'prediction_freeze.json',dict(outputs=predictions,all_frozen_before_scoring=True))
        rows=[]
        for si in TRAIN+VALID+ARC:
            p=probabilities[si];keep=p>=.5
            metrics=None
            if si in labels:
                truth=labels[si];known=truth>=0;positive=truth==1;negative=truth==0
                metrics=dict(known=int(known.sum()),unknown=int((~known).sum()),
                    TP=int((keep&positive).sum()),FN=int((~keep&positive).sum()),
                    FP=int((keep&negative).sum()),TN=int((~keep&negative).sum()))
            folder=GEOMETRY/f'source_{si:02d}';binding=read(folder/'input_freeze.json')
            assert digest(folder/'contacts.json')==binding['files']['contacts.json']
            contacts=read(folder/'contacts.json')
            selected=[c for c in contacts if all(keep[j] for j in c['face_indices'])]
            write(out/f'source_{si:02d}_selected_contacts.json',selected)
            for c in selected:c['p0']=np.asarray(c['p0']);c['direction']=np.asarray(c['direction'])
            target=ROOT/inventory[si]['target']['target_path'];assert digest(target)==inventory[si]['target']['target_sha256']
            coverage=geometry_support(selected,read(target));write(out/f'source_{si:02d}_geometry_support_only.json',coverage)
            row=dict(source=si,surface_teacher_metrics=metrics,selected_surfaces=int(keep.sum()),
                all_surfaces=len(p),selected_geometric_candidates=len(selected),
                original_G=coverage['original_G'],geometric_supported=coverage['geometric_supported'])
            rows.append(row);print('SURFACE_MODEL_DIAGNOSTIC',row,flush=True)
        write(out/'receipt.json',dict(status='PURE_XYZ_SURFACE_VALIDITY_TRAIN_AND_TEST_COMPLETE',utc=utc(),
            actual_optimizer_updates=updates,rows=rows,checkpoint_sha256=digest(out/'surface_validity.pt'),
            original_G=sum(r['original_G'] for r in rows),geometric_supported=sum(r['geometric_supported'] for r in rows),
            not_weld_extraction_accuracy=True,weld_extraction_accuracy=None,material_status='UNKNOWN',
            full_contract_pass=False,no_dataset_expansion=True,elapsed_seconds=time.monotonic()-started))
        print('PURE_XYZ_SURFACE_MODEL_COMPLETE | FULL WELD ACCEPTANCE NOT PASSED',flush=True)
    except BaseException as error:
        write(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),actual_optimizer_updates=updates));raise


if __name__=='__main__':main()
