"""Real per-child features from original fragment features and decoded paths.

This new 466-column schema is not an old interval or parent model interface.
No target geometry, STEP, or development test inputs are used.
"""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/best_baseline_restore';DEST=ROOT/'server_rebuild/child_residual_v1'
def main():
 DEST.mkdir(exist_ok=False);reports=[]
 for key in ['21','BM79A_native_local','10','25','36']:
  case=BASE/'legacy_runtime'/('source_'+key);folder=case/'run379'/('test_'+key)
  decoded=read(folder/'decoded.json');metadata=read(folder/'parent_metadata.json');graph=read(folder/'selected_graph.json');nodes=graph['nodes']
  links={(a,b):kind for a,b,kind in graph['selection']['selected_links']}
  source=case/('run344' if key in ('25','36') else 'run329')/('source_'+key);receipt=read(source/'receipt.json')
  assert receipt['columns']==152 and digest(source/'features.npy')==receipt['features_sha256']
  features=np.load(source/'features.npy',mmap_mode='r');parent_u=np.load(folder/'unary.npy');vectors=[];origins=[];parents=[];child_metadata=[];max_error=0.
  for pi,(parent,m) in enumerate(zip(decoded['parents'],metadata)):
   chain=m['graph_nodes'];lengths=np.array([nodes[i]['hi']-nodes[i]['lo'] for i in chain]);assert np.all(lengths>0)
   raw=np.array(features[[nodes[i]['object_row'] for i in chain]],np.float64)
   mean=(raw*lengths[:,None]).sum(0)/lengths.sum();error=float(abs(mean-parent_u[pi,:152]).max());max_error=max(max_error,error);assert error<1e-9
   groups=[]
   for position,node in enumerate(chain):
    if position==0 or links[chain[position-1],node]!=1:groups.append([])
    groups[-1].append(position)
   assert len(groups)==len(parent['segments'])
   for ci,(segment,positions) in enumerate(zip(parent['segments'],groups)):
    ix=np.array(positions);xx=raw[ix];ww=lengths[ix];prob=np.array([nodes[chain[i]]['probability'] for i in positions]);closed=bool(segment.get('is_closed',segment.get('arc_parameters',{}).get('is_closed',False)))
    extra=[prob.min(),ww@prob/ww.sum(),prob.max(),np.log1p(segment['length_mm']),np.log1p(len(ix)),float(segment['kind']=='line'),float(segment['kind']=='arc'),float(segment['weld_type']=='horizontal'),float(segment['weld_type']=='vertical'),float(closed)]
    vectors.append(np.r_[(xx*ww[:,None]).sum(0)/ww.sum(),xx.min(0),xx.max(0),extra]);origins.append(dict(parent=pi,child=ci,graph_nodes=[chain[i] for i in positions]))
    parents.append(dict(child_count=1,is_closed=closed,gaps=[],segments=[segment]));child_metadata.append(dict(graph_nodes=[chain[i] for i in positions],face_pairs=[m['face_pairs'][i] for i in positions],physical_sides=[segment.get('predicted_side','UNKNOWN')],side_known=segment.get('predicted_side','UNKNOWN') not in ('UNKNOWN',None),probability_mean=float(ww@prob/ww.sum()),length_mm=segment['length_mm'],line_fraction=float(segment['kind']=='line')))
  x=np.asarray(vectors,np.float32);assert x.shape==(len(parents),466) and np.isfinite(x).all();features._mmap.close()
  dest=DEST/('source_'+key);dest.mkdir();np.save(dest/'features.npy',x);write(dest/'origins.json',origins);write(dest/'metadata.json',child_metadata);write(dest/'parents.json',dict(parents=parents,invalid_predictions=[],total_prediction_children=len(parents)))
  row=dict(source=key,children=len(parents),columns=466,GT_reads=0,STEP_reads=0,DEV_reads=0,parent_mean_reconstruction_max_abs=max_error,original_features_sha256=receipt['features_sha256'],source_files={name:digest(folder/name) for name in ['decoded.json','parent_metadata.json','selected_graph.json','unary.npy']},files={name:digest(dest/name) for name in ['features.npy','origins.json','metadata.json','parents.json']});write(dest/'receipt.json',row);reports.append(row);print('CHILD_FEATURES',key,x.shape,'parent mean max error',max_error,flush=True)
 write(DEST/'inputs_frozen.json',dict(schema='ORIGINAL_FRAGMENT152_MEAN_MIN_MAX_PLUS_CHILD10_V1',columns=466,layout=dict(weighted_fragment_mean=[0,152],fragment_min=[152,304],fragment_max=[304,456],child_properties=[456,466]),cases=reports,GT_reads=0,STEP_reads=0,DEV_reads=0,fit_calls=0,not_compatible_with_original_526_interval_schema=True))
if __name__=='__main__':main()
