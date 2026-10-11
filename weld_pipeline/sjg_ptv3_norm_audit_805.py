"""Same weights and point blocks: isolate BatchNorm running-statistics effects."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import OUT
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_train_ptv3_804 import DATA,SurfaceModel,prepare_blocks,evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    checkpoint=OUT/'run23/last.pt';original_sha=digest(checkpoint)
    write(dest/'registered_protocol.json',dict(stage='BATCHNORM_SAME_WEIGHT_AUDIT',checkpoint_sha256=original_sha,
        actual_optimizer_updates=0,source=21,no_validation_labels_or_inputs_used=True,contract_pass=False,
        start_utc=utc(),source_sha256=digest(Path(__file__))))
    check_resources();xyz=np.load(DATA/'source_21_xyz_mm.npy')
    with np.load(DATA/'source_21_teacher.npz') as z:labels={k:z[k] for k in ('instance','normal','boundary_target')}
    blocks=prepare_blocks(xyz,labels,dest);model=SurfaceModel().cuda()
    with checkpoint.open('rb') as h:state=torch.load(h,map_location='cuda')
    model.load_state_dict(state['model']);original=copy.deepcopy(model.state_dict())
    weight=torch.tensor(float((labels['boundary_target']==0).sum()/max(1,(labels['boundary_target']==1).sum())),device='cuda')
    baseline=evaluate(model,blocks,weight);print('ORIGINAL_RUNNING_STATS',baseline,flush=True)
    saved=[]
    for module in model.modules():
        if isinstance(module,torch.nn.BatchNorm1d):
            saved.append((module,module.running_mean,module.running_var,module.num_batches_tracked,module.track_running_stats))
            module.running_mean=None;module.running_var=None;module.num_batches_tracked=None;module.track_running_stats=False
    batch=evaluate(model,blocks,weight);repeat=evaluate(model,blocks,weight)
    write(dest/'repeat_comparison.json',dict(first=batch,second=repeat))
    loss_delta=max(abs(batch['loss'][k]-repeat['loss'][k]) for k in batch['loss'])
    normal_delta=max(abs(batch[k]-repeat[k]) for k in ('unoriented_normal_p50_deg','unoriented_normal_p95_deg'))
    counts_equal=batch['boundary']==repeat['boundary']
    repeat_close=loss_delta<=1e-5 and normal_delta<=.01 and counts_equal
    print('SAME_WEIGHTS_CURRENT_INPUT_STATS',batch,flush=True)
    for module,mean,var,count,track in saved:
        module.running_mean=mean;module.running_var=var;module.num_batches_tracked=count;module.track_running_stats=track
    assert all(torch.equal(value,original[key]) for key,value in model.state_dict().items())
    assert digest(checkpoint)==original_sha
    write(dest/'receipt.json',dict(status='BATCHNORM_AUDIT_COMPLETE',end_utc=utc(),baseline=baseline,
        same_weights_current_input_stats=batch,repeat_loss_max_delta=loss_delta,
        repeat_normal_deg_max_delta=normal_delta,repeat_boundary_counts_equal=counts_equal,
        repeat_close=repeat_close,batchnorm_layers=len(saved),
        all_parameters_and_buffers_restored=True,original_checkpoint_unchanged=True,
        optimizer_updates=0,contract_pass=False))


if __name__=='__main__':main()
