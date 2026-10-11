"""Peak proposals: boundary/plateau/empty-domain and chunk controls."""
from pathlib import Path
import numpy as np
import torch
from weld_pipeline.server_grid_peak_candidates import peak_first_indices,PeakEndpointCandidates
from weld_pipeline.sjg_local_runtime_904 import write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def oracle(scores,legal,k):
    # Enumerate full constant plateaus and test both neighboring values.
    peaks=[];i=0;n=len(scores)
    values=[float(s) if ok else -float('inf') for s,ok in zip(scores,legal)]
    while i<n:
        end=i+1
        while end<n and values[end]==values[i]:end+=1
        if legal[i] and (i==0 or values[i]>values[i-1]) and (end==n or values[i]>values[end]):peaks.append(i)
        i=end
    ranked=sorted((i for i in range(n) if legal[i]),key=lambda i:(-values[i],i))
    chosen=sorted(peaks,key=lambda i:(-values[i],i))[:k]
    for i in ranked:
        if len(chosen)==k:break
        if i not in chosen:chosen.append(i)
    return sorted(chosen,key=lambda i:(-values[i],i))

def main():
    torch.set_num_threads(2);rng=np.random.default_rng(1031);checks=0
    assert peak_first_indices(np.array([9,8,7,3,8,7,4,5,4,6]),np.ones(10,bool),3).tolist()==[0,4,9]
    assert peak_first_indices(np.ones(9),np.ones(9,bool),3).tolist()==[0,1,2]
    for n in [1,3,8,9,37,120]:
        for _ in range(10):
            scores=rng.integers(-3,4,(3,n,2)).astype(np.float32);legal=rng.random((3,n,2))>.2;legal[0]=False
            # Deliberate legal domain containing flat peaks across chunk edges.
            scores[1,:,0]=2.;legal[1,:,0]=True
            for batch in range(3):
                for side in range(2):
                    got=peak_first_indices(scores[batch,:,side],legal[batch,:,side])
                    assert got.tolist()==oracle(scores[batch,:,side],legal[batch,:,side],8)
                    if len(got):assert scores[batch,got[0],side]==scores[batch,legal[batch,:,side],side].max()
            results=[]
            for chunk in [1,7,2048]:
                stream=PeakEndpointCandidates();pos=torch.arange(n,dtype=torch.float64)[None].expand(3,-1)
                for a in range(0,n,chunk):stream.add(torch.from_numpy(scores[:,a:a+chunk]),torch.from_numpy(legal[:,a:a+chunk]),pos[:,a:a+chunk],torch.arange(a,min(a+chunk,n))*2+10)
                results.append(stream.pairs())
            for name in results[0]:assert all(torch.equal(results[0][name],other[name]) for other in results[1:]),name
            assert results[0]['valid'].shape[1]<=64 and not results[0]['valid'][0].any();checks+=1
    assert PeakEndpointCandidates().pairs() is None
    small=PeakEndpointCandidates(max_events=1)
    try:small.add(torch.zeros(1,2,2),torch.ones(1,2,2,dtype=torch.bool),torch.tensor([[0.,1.]]),torch.arange(2))
    except ValueError:pass
    else:raise AssertionError('Budget must fail instead of truncating')
    out=BASE/'grid_peak_candidates_controls_v1';out.mkdir(exist_ok=False)
    write(out/'receipt.json',dict(status='PASS',fixed_budget=64,oracle_and_chunk_cases=checks,global_endpoint_argmax_preserved=True,flat_peak_first_input_id=True,short_and_empty_domains=True,explicit_event_budget=True,spatial_thresholds=0,GT_reads=0,module_sha256=digest('weld_pipeline/server_grid_peak_candidates.py')))
    print('PASS',checks,flush=True)
if __name__=='__main__':main()
