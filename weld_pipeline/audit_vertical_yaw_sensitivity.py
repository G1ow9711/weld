"""Diagnostic rigid yaw invariance of the frozen vertical branch only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.model_registry_branch import model_from_checkpoint
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.typed_contact_model import infer_probabilities
from weld_pipeline.interval_decoder import decode_intervals
from weld_pipeline.evaluation import score_welds


def rotate_vertical_input(features, degrees):
    x = np.asarray(features).copy()
    if x.shape[1] != 84 or not (x[:,66]>.5).all():
        raise ValueError('Only original vertical frames supported')
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle),-np.sin(angle),0.],
                         [np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    for lo in [57,60,63,67]:
        x[:,lo:lo+3] = x[:,lo:lo+3] @ rotation.T
    # All 63 local distance features are rigid-motion invariant. Channels
    # 49:57 are disabled by this vertical architecture; no bbox approximation.
    return x


def main():
    ap = argparse.ArgumentParser()
    for key in ['data','checkpoint','outdir']:
        ap.add_argument('--'+key,required=True)
    ap.add_argument('--groups',nargs='+',required=True)
    args = ap.parse_args()
    with Path(args.checkpoint).open('rb') as f:state=torch.load(f,map_location='cpu')
    verify_cached_inference_sources(state,args.checkpoint)
    if state['model_name']!='typed_branch_surface':raise ValueError('Expected frozen typed branches')
    model=model_from_checkpoint(state).cuda().eval();torch.set_num_threads(4)
    root=Path(args.data);split=json.loads((root/'split.json').read_text(encoding='utf-8'));rows=[]
    for name in args.groups:
        folder=(root/split.get('piece_roots',{}).get(name,name)).resolve()
        if sha256(folder/'manifest.json')!=state['data_hashes'][name]:raise ValueError('Cache changed')
        piece=load_piece(folder)
        sequences=[s for s in piece['sequences'] if s['type']=='vertical']
        truth=[w for w in piece['truth'] if w['type']=='vertical']
        reference=infer_probabilities(model,sequences)
        for angle in [-45.,-.5,0.,.5,45.,90.,180.]:
            variants=[{**s,'features':rotate_vertical_input(s['features'],angle)} for s in sequences]
            probability=infer_probabilities(model,variants)
            delta=np.concatenate([np.abs(a-b) for a,b in zip(probability,reference)])
            # Decode onto original sequences: inverse rigid transform to compare
            # identical physical seams in the original coordinate system.
            prediction=decode_intervals(sequences,probability,threshold=.5,weld_penalty=2.,mask_weight=.25)
            score=score_welds(prediction,truth)
            row={'group':name,'degrees':angle,'role':'development' if name in split['validation'] else 'training',
                 'valid':score['valid_welds'],'truth':len(truth),'predictions':len(prediction),
                 'false':score['false_or_duplicate_predictions'],
                 'mean_probability_difference_per_head':delta.mean(0).tolist(),
                 'max_probability_difference_per_head':delta.max(0).tolist()}
            rows.append(row);print(json.dumps(row),flush=True)
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    (out/'receipt.json').write_text(json.dumps({'checkpoint_sha256':sha256(args.checkpoint),
        'source_sha256':sha256(__file__),'rows':rows,
        'scope':'Rigid-yaw sensitivity diagnostic, no angle selected and no prediction repair. Geometry-specific process-plane cutoffs are masked for vertical XY in this model.'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
