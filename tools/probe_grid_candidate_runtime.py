"""Input-selected longest carrier smoke test; no supervision or threshold tuning."""
from pathlib import Path
import numpy as np
import torch
from weld_pipeline.server_grid_candidate_runtime import score_grid_profile
from weld_pipeline.server_grid_endpoint_selector import GridEndpointSelector
from weld_pipeline.server_fixed_fragment_queries import FixedQueries,GRID
from weld_pipeline.sjg_local_runtime_904 import write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 out=BASE/'grid_candidate_runtime_probe_v1';out.mkdir(exist_ok=False);torch.set_num_threads(2)
 checkpoint=BASE/'bounded_raw_selector_v1/model.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 model=GridEndpointSelector().cuda().eval();model.load_raw_state_dict(state['state_dict']);results=[]
 for key in ['new_cfc732e5f068','new_ceeae49d6777']:
  inp=FixedQueries(BASE/'frozen_features/legacy_runtime'/('source_'+key),key)
  lengths=inp.spans[inp.offsets[1:]-1,1];pid=int(lengths.argmax());lo,hi=inp.offsets[pid:pid+2];spans=inp.spans[lo:hi,:2]
  reps=inp.representative[np.arange(lo,hi)];rows=np.searchsorted(inp.canonical,reps);assert np.array_equal(inp.canonical[rows],reps)
  # Fixed query from the middle input fragment; no reference-based selection.
  row=int(rows[len(rows)//2]);ids=4*row+np.arange(4);features=inp.features(ids);span=inp.spans[inp.canonical[row],:2];qm=span[0]+GRID*(span[1]-span[0])
  kwargs=dict(model=model,query_features=features,query_mm=qm,length_mm=lengths[pid],spans_mm=spans,feature_rows=rows,raw_features=inp.raw,mean=state['mean'],std=state['std'])
  a=score_grid_profile(**kwargs,chunk_size=257);b=score_grid_profile(**kwargs,chunk_size=2048)
  # GPU GEMM may round differently across chunk sizes: demand candidate IDs
  # and validity unchanged, measure score difference separately.
  for name in ['left_ids','right_ids','valid']:assert torch.equal(a['pairs'][name],b['pairs'][name]),name
  valid=a['pairs']['valid'];delta=(a['pairs']['base_logits'][valid]-b['pairs']['base_logits'][valid]).abs().max().item()
  np.savez_compressed(out/(key+'.npz'),**{k:v.cpu().numpy() for k,v in a['pairs'].items()})
  results.append(dict(key=key,profile=pid,events=a['event_count'],valid_pairs=int(valid.sum()),chunk_score_max_abs=delta,input_bindings=inp.bindings));inp.close();print(results[-1],flush=True)
 write(out/'receipt.json',dict(status='PASS',results=results,GT_reads=0,STEP_reads=0,fit_calls=0,detector_gate_reads=0,source_checkpoint_sha256=digest(checkpoint),runtime_module_sha256=digest('weld_pipeline/server_grid_candidate_runtime.py'),scope='Two input-selected carriers only; no whole-object or trained-quality claim'))
if __name__=='__main__':main()
