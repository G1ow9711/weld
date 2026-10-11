"""All-input pointer outputs frozen before optional TRAIN geometry scoring."""
from pathlib import Path
from collections import Counter
import time
import numpy as np
import torch
from weld_pipeline.server_arc_boundary_pointer import ArcBoundaryPointer
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main(model_name='curved_probe_boundary_pointer_v1', output_name='curved_probe_pointer_all_candidates_v1'):
    started=time.perf_counter();torch.set_num_threads(2);key='new_fcaf2af1abee';source=BASE/model_name;mr=read(source/'receipt.json');assert digest(source/'model.pt')==mr['model_sha256'];protocol=read(source/'protocol.json');checkpoint=torch.load(source/'model.pt',map_location='cpu',weights_only=False);assert checkpoint['protocol_sha256']==digest(source/'protocol.json');features=BASE/'curved_probe_event_features_v1'/key;fr=read(features/'receipt.json');assert digest(features/'receipt.json')==protocol['feature_receipt_sha256'];assert digest(features/'features.npy')==fr['feature_sha256'];events=BASE/'curved_probe_vertex_events_v1'/key;er=read(events/'receipt.json');assert digest(events/'receipt.json')==read(features/'protocol.json')['event_receipt_sha256'];assert digest(events/'events.npz')==er['events_sha256'];model=ArcBoundaryPointer(input_dim=83).cuda().eval();model.load_state_dict(checkpoint['state_dict']);x=np.load(features/'features.npy',mmap_mode='r')
    with np.load(events/'events.npz') as z:offsets=z['offsets'];positions=z['positions_mm']
    out=BASE/output_name;out.mkdir(exist_ok=False);write(out/'protocol.json',dict(model_sha256=mr['model_sha256'],feature_receipt_sha256=digest(features/'receipt.json'),event_receipt_sha256=digest(events/'receipt.json'),decoding='All1920 input carriers; unchanged pointer STOP, no threshold, count cap, reference filtering or endpoint correction.',GT_reads=0))
    rows=[];counts=Counter()
    for index,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        values=torch.as_tensor(np.clip((x[a:b]-checkpoint['mean'])/checkpoint['std'],-20,20),dtype=torch.float32,device='cuda');q=torch.as_tensor(positions[a:b],dtype=torch.float64,device='cuda');selected=model.decode(values,q);rows.append(dict(profile=index,selected_local_event_indices=selected));counts[len(selected)]+=1
        if index%100==0:print(index,dict(counts),flush=True)
    write(out/'predictions.json',dict(rows=rows));write(out/'receipt.json',dict(status='ALL_CURVED_INPUT_BOUNDARIES_FROZEN_NOT_PROMOTED',protocol_sha256=digest(out/'protocol.json'),predictions_sha256=digest(out/'predictions.json'),profiles=len(rows),boundary_count_histogram=dict(counts),seconds=time.perf_counter()-started,GT_reads=0))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model-name',default='curved_probe_boundary_pointer_v1');p.add_argument('--output-name',default='curved_probe_pointer_all_candidates_v1');a=p.parse_args();main(a.model_name,a.output_name)
