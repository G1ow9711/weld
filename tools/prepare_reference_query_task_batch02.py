"""Separate projected-reference line occupancy task; never physical absence truth."""
from pathlib import Path
import time
import numpy as np
from weld_pipeline.server_fixed_fragment_queries import FixedQueries,GRID
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';DEST=BASE/'reference_query_task_batch02_v1'
KEYS=tuple(c['key'] for c in read(BASE/'additional_batch02_protocol.json')['cases'])
def main():
 DEST.mkdir(exist_ok=False);write(DEST/'protocol.json',dict(training_keys=KEYS,task='Query lies within an original-D3-validated projected REFERENCE line interval on this frozen finite carrier',zero_means='Outside that fixed projected-reference interval set; NOT physical no-weld',physical_status_for_unmatched='UNKNOWN',arc_queries='Excluded from presence objective because arc prefilter is sampled rather than exhaustive',ambiguous_positive_queries='Masked from both presence and endpoint targets, never background',fixed_query_fractions=GRID.tolist(),positive_carriers='All finite line carriers passing endpoint-distance prefilter, individually original D3 checked; no nearest-three truncation',sampling='All positive line queries plus at most32768 uniform reference-background line queries per family, seed1031, inverse inclusion weights to empirical reference task prior',family_weight='equal',new_model_fits=0,DEV_reads=0))
 for key in KEYS:
  start=time.perf_counter();folder=DEST/key;folder.mkdir();data=BASE/'exhaustive_carrier_supervision_v1/positive_query'/key;r=read(data/'receipt.json')
  for name,sha in r['files'].items():assert digest(data/name)==sha
  records=read(data/'records.json');masked=read(data/'masked_queries.json');source=BASE/'frozen_features/legacy_runtime'/('source_'+key);inp=FixedQueries(source,key);profile_line=np.array([p['representation']['kind']=='line' for p in inp.profiles]);valid=np.repeat(profile_line[inp.pid],4);y=np.full(inp.n,-1,np.int8);y[valid]=0
  record_ids=np.array([v['feature_rows'][0]//2*4+v['query_index'] for v in records]);positive=record_ids[valid[record_ids]];y[positive]=1
  ambiguous=np.array([v['feature_row']//2*4+v['query_index'] for v in masked],np.int64);y[ambiguous]=-1
  # Independent interval union on the fixed grid, using every verified line witness.
  witnessdir=BASE/'exhaustive_carrier_supervision_v1/positive_intervals'/key;wr=read(witnessdir/'receipt.json');assert digest(witnessdir/'witnesses.json')==wr['witnesses_sha256'];covered=np.zeros(inp.n,bool)
  for witness in read(witnessdir/'witnesses.json')['positive_witnesses']:
   if witness['kind']!='line':continue
   pid=witness['profile'];lo,hi=witness['interval_mm'];first=np.searchsorted(inp.pid,pid,'left');last=np.searchsorted(inp.pid,pid,'right');fr=np.arange(first,last);span=inp.spans[inp.canonical[fr],:2];query=span[:,0,None]+GRID*(span[:,1]-span[:,0])[:,None];ids=fr[:,None]*4+np.arange(4);covered[ids[(query>=lo)&(query<=hi)]]=True
  assert np.array_equal(covered[y>=0],y[y>=0]==1),'Reference union and positive query table disagree; do not create negatives'
  pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);rng=np.random.default_rng(1031);take=rng.choice(neg,min(32768,len(neg)),replace=False);ids=np.r_[pos,take];weights=np.r_[np.ones(len(pos)),np.full(len(take),len(neg)/len(take))]/(len(pos)+len(neg));assert np.isclose(weights.sum(),1)
  x=inp.features(ids);np.save(folder/'reference_query_labels.npy',y);np.savez(folder/'sampled_presence.npz',features=x,labels=y[ids],query_ids=ids,weights=weights)
  # All supervised positive inputs retain the exact proposal feature definition.
  original=np.load(data/'features.npy')
  for i in range(0,len(record_ids),4096):assert np.array_equal(inp.features(record_ids[i:i+4096]),original[i:i+4096])
  write(folder/'receipt.json',dict(key=key,total_queries=inp.n,line_reference_positive=len(pos),line_reference_background=len(neg),excluded_queries=int((y<0).sum()),sampled_queries=len(ids),all_witness_union_matches_query_labels=True,positive_features_exact=True,physical_negative_labels=False,input_bindings=inp.bindings,positive_receipt_sha256=digest(data/'receipt.json'),witness_receipt_sha256=digest(witnessdir/'receipt.json'),files={p.name:digest(p) for p in folder.glob('*') if p.is_file()},fit_calls=0,DEV_reads=0,seconds=time.perf_counter()-start));inp.close();print(key,len(pos),len(neg),len(ids),flush=True)
 write(DEST/'receipt.json',dict(status='TWO_BATCH02_TRAIN_REFERENCE_TASK_TABLES_COMPLETE',fit_calls=0,DEV_reads=0,source_sha256=digest(__file__)))
if __name__=='__main__':main()
