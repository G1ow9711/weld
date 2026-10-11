"""Target-free inference for a frozen parent-set model and complete pure inputs."""
import argparse
import gc
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_parent_set_network_1075 import ParentSetModel
from weld_pipeline.sjg_parent_input_1078 import ParentInput
from weld_pipeline.sjg_parent_anchor_bank_1080 import CarrierBank
from weld_pipeline.sjg_parent_stream_encoder_1087 import stream_encode
from weld_pipeline.sjg_parent_set_runtime_1081 import network_choices,decode_choices
from weld_pipeline.sjg_parent_resource_start_1090 import start_cuda


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--train',required=True)
    ap.add_argument('--source',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);start=time.perf_counter()
    try:
        trained=OUT/args.train;receipt=read(trained/'receipt.json')
        recovered=receipt['status']=='PARENT_SET_RECOVERY_COMPLETE_ORIGINAL_BUDGET_FAILED'
        assert (receipt['status']=='PARENT_SET_TRAIN_COMPLETE' or recovered) and receipt['updates']==30
        checkpoint_file=trained/'final.pt'
        if recovered:
            assert receipt['recovery_wall_pass'] and not receipt['original_budget_180s_pass']
            assert receipt['checkpoint_file']=='update_30.pt'
            checkpoint_file=trained/receipt['checkpoint_file']
            watchdog=read(trained/'watchdog_receipt.json')
            assert watchdog['exit_code']==0 and watchdog['termination_reason'] is None and watchdog['wall_seconds']<=120
        assert digest(checkpoint_file)==receipt['checkpoint_sha256']
        torch.set_num_threads(2);torch.manual_seed(1031);reclaim_own_idle_pages();check_resources()
        write(dest/'startup_pages.json',start_cuda())
        with checkpoint_file.open('rb') as f:checkpoint=torch.load(f,map_location='cpu')
        assert digest(trained/'normalization.npz')==checkpoint['normalization_sha256']
        channels=int(checkpoint['protocol'].get('bin_channels',242))
        if channels not in (242,247):raise ValueError('Unregistered input width')
        if checkpoint['protocol'].get('decoder')=='persistent_query_1097':
            from weld_pipeline.sjg_persistent_query_network_1097 import PersistentQueryModel
            model=PersistentQueryModel(bin_channels=channels).cuda()
        else:model=ParentSetModel(bin_channels=channels).cuda()
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        del checkpoint;gc.collect()
        with np.load(trained/'normalization.npz') as z:normalization={k:z[k] for k in z.files}
        if channels==247:
            from weld_pipeline.sjg_roboweldar_pair_support_1093 import PairParentInput
            data=PairParentInput(args.source,normalization)
        else:data=ParentInput(args.source,normalization)
        bank=CarrierBank(args.source)
        binding=data.binding();reclaim_own_idle_pages();check_resources();tick=time.perf_counter()
        with torch.no_grad():
            result=model.forward_encoded(stream_encode(model,data))
            choices=network_choices(model,result,bank);decoded=decode_choices(choices,bank)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        write(dest/'choices.json',choices);write(dest/'prediction.json',decoded)
        probabilities=result['existence'].sigmoid().detach().cpu().numpy()
        write(dest/'receipt.json',dict(status='FROZEN_PARENT_PREDICTION_READY',source=args.source,
            train_run=args.train,checkpoint_sha256=receipt['checkpoint_sha256'],input_binding=binding,
            completed_by_one_recovery=recovered,original_training_budget_pass=not recovered,
            prediction_sha256=digest(dest/'prediction.json'),choices_sha256=digest(dest/'choices.json'),
            complete_profiles=len(data),selected_parent_queries=len(choices),declared_children=decoded['total_prediction_children'],
            probability_min=float(probabilities.min()),probability_max=float(probabilities.max()),
            cached_downstream_inference_seconds=seconds,wall_seconds=time.perf_counter()-start,
            formal_uncached_contract_timing=False,GT_reads=0,STEP_reads=0,GT_count_used=False,utc=utc()))
        print('FROZEN_PREDICTION',args.source,'PARENTS',len(choices),'CHILDREN',decoded['total_prediction_children'],
            'CACHED_SECONDS',round(seconds,3),flush=True)
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),utc=utc()));raise


if __name__=='__main__':main()
