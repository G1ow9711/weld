"""Zero-fit numerical grouping intervention on all frozen candidate carriers."""
from pathlib import Path
from collections import Counter
import time
import numpy as np
import torch
from weld_pipeline.server_arc_boundary_pointer import ArcBoundaryPointer
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main(model_name='curved_probe_boundary_pointer_v1', output_name='curved_probe_grouped_pointer_all_candidates_v1'):
    started=time.perf_counter();torch.set_num_threads(2);key='new_fcaf2af1abee';source=BASE/model_name;mr=read(source/'receipt.json');assert digest(source/'model.pt')==mr['model_sha256'];checkpoint=torch.load(source/'model.pt',map_location='cpu',weights_only=False);assert checkpoint['protocol_sha256']==digest(source/'protocol.json');folder=BASE/'curved_probe_event_groups_v1'/key;fr=read(folder/'receipt.json');assert digest(folder/'protocol.json')==fr['protocol_sha256'];fp=read(folder/'protocol.json');assert fp['feature_receipt_sha256']==read(source/'protocol.json')['feature_receipt_sha256']
    for name in ['events.npz','mean.npy']:assert digest(folder/name)==fr['files'][name]
    x=np.load(folder/'mean.npy',mmap_mode='r')
    with np.load(folder/'events.npz') as z:offsets=z['offsets'];positions=z['positions_mm']
    model=ArcBoundaryPointer(input_dim=83).cuda().eval();model.load_state_dict(checkpoint['state_dict']);out=BASE/output_name;out.mkdir(exist_ok=False);write(out/'protocol.json',dict(model_sha256=mr['model_sha256'],group_receipt_sha256=digest(folder/'receipt.json'),decoding='Same trained weights/normalization, mean83 features within input-only numerical coincidence groups; all1920 carriers, unchanged pointer STOP. No label filtering or new fit.',GT_reads=0,fit_calls=0));rows=[];counts=Counter()
    for index,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        values=torch.as_tensor(np.clip((x[a:b]-checkpoint['mean'])/checkpoint['std'],-20,20),dtype=torch.float32,device='cuda');q=torch.as_tensor(positions[a:b],dtype=torch.float64,device='cuda');selected=model.decode(values,q);rows.append(dict(profile=index,selected_local_event_indices=selected));counts[len(selected)]+=1
        if index%200==0:print(index,dict(counts),flush=True)
    write(out/'predictions.json',dict(rows=rows));write(out/'receipt.json',dict(status='ZERO_FIT_GROUPED_BOUNDARIES_FROZEN_NOT_PROMOTED',protocol_sha256=digest(out/'protocol.json'),predictions_sha256=digest(out/'predictions.json'),profiles=len(rows),boundary_count_histogram=dict(counts),seconds=time.perf_counter()-started,GT_reads=0,fit_calls=0))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model-name',default='curved_probe_boundary_pointer_v1');p.add_argument('--output-name',default='curved_probe_grouped_pointer_all_candidates_v1');a=p.parse_args();main(a.model_name,a.output_name)
