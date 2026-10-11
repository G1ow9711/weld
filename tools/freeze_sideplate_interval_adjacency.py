"""Freeze exact predicted-interval connections; no grouping decision or GT read."""
from pathlib import Path
import numpy as np
from weld_pipeline.server_exact_interval_adjacency import interval_adjacency
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main():
    # Same coordinate with different provenance is allowed; positive gaps are not.
    p=np.array([1,1,1,2,1,1]);r=np.array([0,0,0,0,1,0])
    intervals=np.array([[0,1],[1,2],[2.01,3],[1,2],[1,2],[1,1]])
    assert np.array_equal(interval_adjacency(p,r,intervals),[[0,1]])
    out=BASE/'sideplate_interval_adjacency_v1';out.mkdir(exist_ok=False);rows=[]
    for arm in ('control8','expanded9'):
        for key in ('new_cfc732e5f068','new_ceeae49d6777'):
            folder=BASE/('sideplate_endpoint_'+arm+'_full_output_v1')/key
            freeze=read(folder/'prediction_freeze.json');path=folder/'compact_predictions.npz';assert digest(path)==freeze['files']['compact_predictions.npz']
            with np.load(path) as compact:
                intervals=compact['interval_mm'];edges=interval_adjacency(compact['profile_ids'],compact['weld_type'],intervals)
                assert np.all(intervals[edges[:,0],1]==intervals[edges[:,1],0])
                assert np.all(intervals[edges[:,0],0]<intervals[edges[:,1],0])
            dest=out/(arm+'_'+key+'.npz');np.savez(dest,edges=edges)
            rows.append(dict(arm=arm,key=key,intervals=len(intervals),edges=len(edges),file=dest.name,sha256=digest(dest),prediction_freeze_sha256=digest(folder/'prediction_freeze.json')))
            print(rows[-1],flush=True)
    write(out/'receipt.json',dict(status='INPUT_ADJACENCY_FROZEN',rows=rows,module_sha256=digest('weld_pipeline/server_exact_interval_adjacency.py'),GT_reads=0,controls='Exact shared coordinates only; different profile/type, positive gaps and empty intervals rejected. Strictly increasing interval start proves directed acyclic graph.',scope='Candidate adjacency only, not parent selection. No coordinates merged, no gap filled, no generated parent output.'))

if __name__=='__main__':main()
