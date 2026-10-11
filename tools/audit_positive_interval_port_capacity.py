"""Offline test: can actual frozen fragment ports represent positive child endpoints?"""
import argparse,collections
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def run(key):
 old=key in ('21','BM79A_native_local','10','25','36');root=BASE/'original_train_query_inputs_v1' if old else BASE;src=root/'frozen_features/legacy_runtime'/('source_'+key);witness=root/'positive_intervals'/key;wr=read(witness/'receipt.json');assert digest(witness/'witnesses.json')==wr['witnesses_sha256'];events=read(witness/'witnesses.json')['positive_witnesses'];profiles=read(src/'run185'/('source_'+key)/'profiles.json');binding=next(r for r in read(src/'run285/all_geometry_frozen_before_labels.json')['cases'] if r['source']==key)
 with np.load(src/'run285'/binding['geometry_file']) as z:offsets=z['offsets'];spans=z['spans']
 rows=[];owners=set();covered=set()
 for e in events:
  owner=(e['parent'],e['child']);owners.add(owner);pid=e['profile'];curve=CurveCarrier(profiles[pid]['representation']);ports=np.unique(spans[offsets[pid]:offsets[pid+1],:2]);q=np.asarray(e['interval_mm']);dist=[];chosen=[]
  for value in q:
   j=int(np.searchsorted(ports,value));ix=np.unique(np.clip([j-1,j],0,len(ports)-1));nearest=int(ix[np.argmin(abs(ports[ix]-value))]);chosen.append(float(ports[nearest]));dist.append(float(abs(ports[nearest]-value)))
  # This compares to the already D3-valid projected interval. It is a necessary
  # port representation diagnostic, not an extra 3mm tolerance on GT geometry.
  ok=max(dist)<=3 and chosen[0]<chosen[1]
  if ok:covered.add(owner)
  rows.append(dict(parent=owner[0],child=owner[1],profile=pid,projected_positive_interval_mm=e['interval_mm'],nearest_input_ports_mm=chosen,port_distance_to_positive_projection_mm=dist,within_3mm_of_projected_endpoints=ok))
 dest=BASE/'positive_query_head_v1/port_capacity';dest.mkdir(exist_ok=True);report=dict(key=key,represented_children=len(owners),children_with_port_pair_near_projected_endpoints=len(covered),missing=sorted(owners-covered),rows=rows,scope='Input-only existing fragment ports, offline GT-positive projection comparison; not original D3 pass or contact proof, no GT ports inserted.',witness_sha256=digest(witness/'witnesses.json'),fit_calls=0,DEV_reads=0);write(dest/(key+'.json'),report);print(key,len(covered),'/',len(owners),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);run(p.parse_args().key)
