"""Offline selected-link annotation audit; never modifies inference outputs."""
from collections import Counter
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
BASE=ROOT/'server_rebuild/connection_annotation_full_probe_v1'
OLD=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime/source_10/run379/test_10'

def main():
 labels=np.load(ROOT/'server_rebuild/connection_annotation_targets_v1/10/labels.npy')
 folder=BASE/'legacy_runtime/source_10/run379/test_10';m=read(folder/'edge_manifest.json')['edge_count']
 assert digest(folder/'edges.i32')==digest(OLD/'edges.i32')
 edges=np.memmap(folder/'edges.i32',dtype='<i4',mode='r',shape=(m,2));n=len(read(folder/'selected_graph.json')['nodes']);codes=edges[:,0].astype(np.int64)*n+edges[:,1];order=np.argsort(codes);sorted_codes=codes[order]
 def describe(selection):
  rows=[];counts=Counter()
  for a,b,k in selection:
   pos=int(np.searchsorted(sorted_codes,a*n+b));assert sorted_codes[pos]==a*n+b
   ix=int(order[pos]);y=int(labels[ix]);category='unknown' if y<0 else ('annotated_no_link' if y==0 else ('compatible_relation' if y==k else 'positive_wrong_relation'))
   rows.append(dict(a=a,b=b,selected_kind=k,annotation=y,category=category));counts[category]+=1
  return dict(counts=dict(counts),rows=rows)
 results={}
 for name,path in [('original',OLD),('guard',ROOT/'server_rebuild/roundoff_gap_baseline_probe_v1/legacy_runtime/source_10/run379/test_10'),('binary_adapter',folder)]:
  assert digest(path/'edges.i32')==digest(OLD/'edges.i32')
  links=read(path/'selected_graph.json')['selection']['selected_links'];all_result=describe(links)
  metadata=read(path/'parent_metadata.json');accepted=read(path/'final_prediction.json')['selected_candidate_indices'];pairs=set()
  for ix in accepted:
   chain=metadata[ix]['graph_nodes'];pairs.update(zip(chain[:-1],chain[1:]))
  retained=[v for v in links if tuple(v[:2]) in pairs]
  results[name]=dict(all_selected=all_result,accepted_parent_links=describe(retained),accepted_parents=len(accepted))
 old=read(OLD/'evaluation.json');new=read(BASE/'evaluation.json')['score'];oldids={v['original_child_id'] for v in old['matches']};newids={v['original_child_id'] for v in new['matches']}
 report=dict(arms=results,gained=sorted(newids-oldids),lost=sorted(oldids-newids),scope='Offline annotation relation compatibility. Unknown is not false. Accepted links map open ordered chains; no runtime GT filtering.',fit_calls=0)
 write(BASE/'selected_link_annotation_audit.json',report)
 for arm,v in results.items():print(arm,{k:r['counts'] for k,r in v.items() if isinstance(r,dict)},flush=True)
 print('GAINED',len(report['gained']),'LOST',len(report['lost']),flush=True)

if __name__=='__main__':main()
