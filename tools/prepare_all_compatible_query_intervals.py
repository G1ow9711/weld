"""Positive-only whole-child interval labels on a fixed input fragment query grid.

This is a proposal head, not an existence/count classifier. Missing labels do
not become background. Queries are fixed before reading witness intervals.
"""
import argparse,collections,time
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';DEST=BASE/'exhaustive_carrier_supervision_v1/positive_query'
GRID=(.125,.375,.625,.875)
def run(key):
 assert key in {c['key'] for c in read(BASE/'coverage_cohort_protocol.json')['cases']};start=time.perf_counter();dest=DEST/key;dest.mkdir(parents=True,exist_ok=False)
 protocol=dict(schema='SIDE_SYMMETRIC_FRAGMENT152_MEAN_MIN_MAX_QUERY1_V1',columns=457,fixed_fragment_query_fractions=GRID,features='Mean/min/max of two actual unresolved-side 152 features, plus input query arclength / full carrier length',targets='Whole annotated child lo/hi ratios on frozen finite carrier and horizontal/vertical type; proposals only, no existence/count target',ambiguous_query_target='Mask if known target interval/type differs; never average incompatible modes',unknown_queries='No supervised target, no negative/background assignment',GT_only_in_offline_supervision=True,fit_calls=0,DEV_reads=0)
 write(dest/'protocol.json',protocol)
 source=BASE/'frozen_features/legacy_runtime'/('source_'+key);f=source/('run344' if key in ('25','36') else 'run329')/('source_'+key);fr=read(f/'receipt.json');assert digest(f/'features.npy')==fr['features_sha256'];x=np.load(f/'features.npy',mmap_mode='r')
 graphrow=next(r for r in read(source/'run285/all_geometry_frozen_before_labels.json')['cases'] if r['source']==key)
 with np.load(source/'run285'/graphrow['geometry_file']) as z:offsets=z['offsets'];spans=z['spans']
 witness=BASE/'exhaustive_carrier_supervision_v1/positive_intervals'/key;wr=read(witness/'receipt.json');assert digest(witness/'witnesses.json')==wr['witnesses_sha256'];events=read(witness/'witnesses.json')['positive_witnesses'];groups=collections.defaultdict(list)
 for wi,r in enumerate(events):
  for e in r['fragments']:
   fragment=e['canonical_fragment'];pid=int(np.searchsorted(offsets,fragment,side='right')-1);length=float(spans[offsets[pid+1]-1,1]);left,right=spans[fragment,:2];row=e['feature_rows_both_unresolved_sides'][0];assert e['feature_rows_both_unresolved_sides']==[row,row+1] and row%2==0
   for qi,fraction in enumerate(GRID):
    q=float(left+fraction*(right-left))
    if e['positive_overlap_mm'][0]<=q<=e['positive_overlap_mm'][1]:groups[row,qi].append(dict(parent=r['parent'],child=r['child'],witness=wi,target=np.asarray(r['interval_mm'])/length,weld_type=r['weld_type'],query=q/length,profile=pid,length=length,fragment=int(fragment)))
 records=[];vectors=[];targets=[];types=[];masked=[]
 for (row,qi),values in sorted(groups.items()):
  first=values[0];agree=all(v['weld_type']==first['weld_type'] and np.max(abs(v['target']-first['target']))*first['length']<=1e-6 for v in values)
  if not agree:masked.append(dict(feature_row=row,query_index=qi,owners=sorted({(v['parent'],v['child']) for v in values}),reason='INCOMPATIBLE_KNOWN_ENDPOINT_OR_TYPE_MODES'));continue
  xx=np.asarray(x[[row,row+1]],np.float64);vector=np.r_[xx.mean(0),xx.min(0),xx.max(0),first['query']];assert vector.shape==(457,) and np.isfinite(vector).all();lo,hi=first['target'];assert 0<=lo<hi<=1
  vectors.append(vector);targets.append([lo,hi]);types.append(0 if first['weld_type']=='horizontal' else 1);records.append(dict(feature_rows=[row,row+1],query_index=qi,query_fraction=GRID[qi],carrier_query_ratio=first['query'],profile=first['profile'],fragment=first['fragment'],owners=sorted({(v['parent'],v['child']) for v in values})))
 assert vectors,'No positive input-grid queries; do not fit fabricated rows'
 owner_counts=collections.Counter(tuple(o) for r in records for o in r['owners']);weights=np.array([sum(1/owner_counts[tuple(o)] for o in r['owners'])/len(owner_counts) for r in records]);assert np.isclose(weights.sum(),1)
 np.save(dest/'features.npy',np.asarray(vectors,np.float32));np.savez(dest/'targets.npz',interval_ratios=np.asarray(targets),weld_type=np.asarray(types,np.int8),weights=weights);write(dest/'records.json',records);write(dest/'masked_queries.json',masked);x._mmap.close()
 write(dest/'receipt.json',dict(key=key,status='POSITIVE_QUERY_SUPERVISION_FROZEN',rows=len(records),represented_GT_children=len(owner_counts),masked_ambiguous_queries=len(masked),feature_source_sha256=fr['features_sha256'],witness_receipt_sha256=digest(witness/'receipt.json'),target_sha256=wr['target_sha256'],files={n:digest(dest/n) for n in ['features.npy','targets.npz','records.json','masked_queries.json','protocol.json']},fit_calls=0,DEV_reads=0,source_sha256=digest(__file__),seconds=time.perf_counter()-start));print(key,len(records),'rows',len(owner_counts),'GT children',len(masked),'masked',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);run(p.parse_args().key)
