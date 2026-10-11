"""Check chunk-invariant features and all-profile raw-input binding; no labels."""
from pathlib import Path
import hashlib
import numpy as np
from weld_pipeline.server_grid_feature_stream import grid_feature_chunks
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main():
 out=BASE/'grid_feature_stream_controls_v1';out.mkdir(exist_ok=False)
 raw=np.arange(6*152,dtype=float).reshape(6,152)
 def collect(chunk):
  parts=list(grid_feature_chunks(7.,np.array([[0.,2.],[2.,7.]]),[2,0],raw,chunk_size=chunk))
  return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
 a,b=collect(1),collect(2048)
 assert all(np.array_equal(a[k],b[k]) for k in a)
 assert a['feature_rows'].tolist()==[2,2,0,0,0,0,0,0]
 assert np.array_equal(a['features'][:,-1],(a['positions_mm']/7).astype(np.float32))
 results=[]
 for key in ['new_cfc732e5f068','new_ceeae49d6777']:
  inp=FixedQueries(BASE/'frozen_features/legacy_runtime'/('source_'+key),key)
  signature=hashlib.sha256();count=0;peak=0;profiles=0
  for pid in range(len(inp.offsets)-1):
   lo,hi=inp.offsets[pid:pid+2];spans=inp.spans[lo:hi,:2]
   reps=inp.representative[np.arange(lo,hi)];rows=np.searchsorted(inp.canonical,reps)
   assert np.array_equal(inp.canonical[rows],reps)
   for part in grid_feature_chunks(float(spans[-1,1]),spans,rows,inp.raw):
    signature.update(part['features'].tobytes());signature.update(part['feature_rows'].tobytes())
    count+=len(part['positions_mm']);peak=max(peak,part['features'].nbytes)
   profiles+=1
   if profiles%1000==0:print(key,profiles,count,flush=True)
  result=dict(key=key,profiles=profiles,grid_events=count,peak_feature_chunk_bytes=peak,feature_sha256=signature.hexdigest(),input_bindings=inp.bindings)
  write(out/(key+'.json'),result);results.append(result);inp.close();print(result,flush=True)
 write(out/'receipt.json',dict(status='PASS',results=results,chunk_and_alias_controls=True,detector_gate_reads=0,GT_reads=0,STEP_reads=0,fit_calls=0,module_sha256=digest('weld_pipeline/server_grid_feature_stream.py'),scope='All finite input profiles, streamed features only; no presence, endpoint or parent prediction'))
if __name__=='__main__':main()
