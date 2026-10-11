"""Shared input-only grid candidate scoring, before any occupancy gate.

This is an additional proposal branch, not an output decoder or physical contact
classifier. The downstream parent selector must be allowed to reject all pairs.
"""
import numpy as np
import torch
from .server_grid_feature_stream import grid_feature_chunks
from .server_grid_endpoint_selector import StreamingEndpointTopK

@torch.no_grad()
def score_grid_profile(model, query_features, query_mm, length_mm, spans_mm,
                       feature_rows, raw_features, mean, std, chunk_size=2048,
                       candidate_policy="topk"):
    device=next(model.parameters()).device
    tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
    queries=np.asarray(query_features);qm=np.asarray(query_mm,dtype=np.float64)
    if queries.ndim!=2 or queries.shape[1]!=457 or qm.shape!=(len(queries),):
        raise ValueError('Query features and finite arclengths required')
    if not np.isfinite(qm).all() or np.any(qm<0) or np.any(qm>length_mm):
        raise ValueError('Query outside finite carrier')
    if not np.all(np.asarray(std)>0):raise ValueError('Positive normalization scale required')
    qx=tensor(np.clip((queries-mean)/std,-20,20));qr=tensor(qm/length_mm)
    qpos=torch.as_tensor(qm,device=device,dtype=torch.float64)
    lengths=tensor(np.full(len(qm),length_mm))
    if candidate_policy == "topk":
        stream=StreamingEndpointTopK(8)
    elif candidate_policy == "local_peaks":
        from .server_grid_peak_candidates import PeakEndpointCandidates
        stream=PeakEndpointCandidates(8)
    else:
        raise ValueError('Unknown fixed candidate policy')
    count=0
    for part in grid_feature_chunks(length_mm,spans_mm,feature_rows,raw_features,chunk_size=chunk_size):
        features=tensor(np.clip((part['features']-mean)/std,-20,20))[None].expand(len(qm),-1,-1)
        positions=torch.as_tensor(part['positions_mm'],device=device,dtype=torch.float64)[None].expand(len(qm),-1)
        ratio=tensor(part['positions_mm']/length_mm)[None].expand(len(qm),-1)
        # No observed-event provenance is asserted for this synthetic branch.
        # Zeros in normalized space plus the explicit grid flag are a new input
        # convention requiring training, not calibrated legacy contact scores.
        provenance=torch.zeros((*ratio.shape,13),device=device)
        flag=torch.ones_like(ratio,dtype=torch.bool)
        logits=model(qx,features,qr,ratio,lengths,provenance,flag)
        legal=torch.stack((positions<=qpos[:,None],positions>=qpos[:,None]),-1)
        stream.add(logits,legal,positions,torch.as_tensor(part['grid_ids'],device=device))
        count+=len(part['grid_ids'])
    result=stream.pairs()
    return dict(pairs=result,event_count=count,provenance='synthetic_grid_unknown_observed_event_fields',
                output_selection_implemented=False,candidate_policy=candidate_policy)
