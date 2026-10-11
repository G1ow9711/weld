"""CPU inference of the terminal 26-update checkpoint, explicitly incomplete.

Registered 30-update trial failed its budget. This diagnostic never changes that
status, resumes training, selects an earlier/best epoch, or certifies acceptance.
"""
import gc
import time
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_persistent_query_network_1097 import PersistentQueryModel
from weld_pipeline.sjg_roboweldar_pair_support_1093 import PairParentInput
from weld_pipeline.sjg_parent_anchor_bank_1080 import CarrierBank
from weld_pipeline.sjg_parent_stream_encoder_1087 import stream_encode
from weld_pipeline.sjg_parent_set_runtime_1081 import network_choices,decode_choices


def main():
    dest=OUT/'run282';dest.mkdir(exist_ok=False);started=time.perf_counter()
    source=OUT/'run280';failure=read(source/'failure.json')
    assert failure['actual_updates']==26 and '180-second budget exhausted before update' in failure['error']
    assert not (source/'final.pt').exists() and not (source/'receipt.json').exists()
    sha=digest(source/'interrupted.pt')
    write(dest/'diagnostic_protocol.json',dict(status='REGISTERED_BEFORE_PREDICTION',
        checkpoint_sha256=sha,selection='only terminal interrupted checkpoint; no metric-based selection',
        completed_updates=26,required_updates=30,training_budget_failed=True,
        inference_device='cpu',source='21',original_G=1,all_profiles=True,
        original_existence_threshold=.5,geometry_or_GT_filter=False,
        registered_training_stays_incomplete=True,full_contract_pass=False,utc=utc()))
    torch.set_num_threads(2);check_resources()
    with (source/'interrupted.pt').open('rb') as f:checkpoint=torch.load(f,map_location='cpu')
    assert checkpoint['updates']==26 and checkpoint['status']=='INCOMPLETE'
    model=PersistentQueryModel();model.load_state_dict(checkpoint['state_dict']);model.eval()
    del checkpoint;gc.collect();reclaim_own_idle_pages();check_resources()
    assert digest(source/'normalization.npz')==read(OUT/'run260/input_freeze.json')['normalization_sha256']
    with np.load(source/'normalization.npz') as z:norm={k:z[k] for k in z.files}
    data=PairParentInput('21',norm);bank=CarrierBank('21');binding=data.binding()
    assert binding==read(source/'input_bindings.json')['cases'][0]
    tick=time.perf_counter()
    with torch.no_grad():
        result=model.forward_encoded(stream_encode(model,data))
        choices=network_choices(model,result,bank);decoded=decode_choices(choices,bank)
    seconds=time.perf_counter()-tick
    write(dest/'choices.json',choices);write(dest/'prediction.json',decoded)
    probability=result['existence'].sigmoid()
    assert digest(source/'interrupted.pt')==sha
    write(dest/'receipt.json',dict(status='FROZEN_PARENT_PREDICTION_READY',source='21',train_run='run280',
        incomplete_training=True,completed_training_updates=26,required_training_updates=30,
        registered_final_evaluation=False,checkpoint_sha256=sha,input_binding=binding,
        prediction_sha256=digest(dest/'prediction.json'),choices_sha256=digest(dest/'choices.json'),
        complete_profiles=len(data),selected_parent_queries=len(choices),declared_children=decoded['total_prediction_children'],
        probability_min=float(probability.min()),probability_max=float(probability.max()),
        cached_downstream_inference_seconds=seconds,wall_seconds=time.perf_counter()-started,
        inference_device='cpu',formal_uncached_contract_timing=False,GT_reads=0,STEP_reads=0,
        optimizer_updates=0,GT_count_used=False,utc=utc()))
    print('INTERRUPTED_MODEL_DIAGNOSTIC_FROZEN',len(choices),'PARENTS','26/30 TRAINING UPDATES',flush=True)


if __name__=='__main__':main()
