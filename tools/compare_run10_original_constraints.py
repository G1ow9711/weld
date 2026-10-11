"""Compare stored constraints on uniquely corresponding diagnostic graph nodes."""
from collections import Counter
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest
DEST=ROOT/'server_rebuild/run446_uploaded_vs_rebuilt_audit_v1'

def load(folder,mapping):
 manifest=read(folder/'edge_manifest.json')
 for name,r in manifest['files'].items():assert digest(folder/name)==r['sha256']
 n=manifest['edge_count'];edges=np.memmap(folder/'edges.i32',dtype=np.int32,mode='r',shape=(n,2));mapped=mapping[edges];valid=(mapped>=0).all(1);indices=np.flatnonzero(valid);pairs=mapped[valid];codes=pairs[:,0].astype(np.int64)*10000+pairs[:,1];assert len(np.unique(codes))==len(codes)
 probabilities=np.memmap(folder/'probabilities.f64',dtype=np.float64,mode='r',shape=(n,4));legal=np.memmap(folder/'legal.u8',dtype=np.uint8,mode='r',shape=(n,3))
 off=np.load(folder/'resources_offsets.npy');ids=np.load(folder/'resources_indices.npy');resources=Counter();omitted=0
 for a,b in zip(off[:-1],off[1:]):
  row=mapping[ids[a:b]]
  if (row<0).any():omitted+=1
  else:resources[tuple(sorted(row.tolist()))]+=1
 return dict(codes=codes,indices=indices,probabilities=probabilities,legal=legal,resources=resources,total_edges=n,unmapped_edges=int((~valid).sum()),resource_rows=len(off)-1,unmapped_resources=omitted,manifest_sha256=digest(folder/'edge_manifest.json'))

def main():
 corr=read(DEST/'10_graph_node_correspondence.json');oldfolder=OUT/'run379/source_10';newfolder=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime/source_10/run379/test_10'
 oldg=read(oldfolder/'selected_graph.json');testg=read(OUT/'run379/test_10/selected_graph.json');assert oldg['nodes']==testg['nodes'];assert digest(OUT/'run379/test_10/selected_graph.json')==corr['original_graph_sha256'];assert digest(newfolder/'selected_graph.json')==corr['rebuilt_graph_sha256']
 assert oldg['edge_manifest_sha256']==digest(oldfolder/'edge_manifest.json')
 assert read(newfolder/'selected_graph.json')['edge_manifest_sha256']==digest(newfolder/'edge_manifest.json')
 om=np.full(corr['original_nodes'],-1,np.int64);nm=np.full(corr['rebuilt_nodes'],-1,np.int64)
 for i,p in enumerate(corr['pairs']):om[p['original']]=i;nm[p['rebuilt']]=i
 a=load(oldfolder,om);b=load(newfolder,nm);common,ia,ib=np.intersect1d(a['codes'],b['codes'],return_indices=True);ai=a['indices'][ia];bi=b['indices'][ib];delta=np.abs(a['probabilities'][ai]-b['probabilities'][bi]);different=(delta>0).any(1);ld=(a['legal'][ai]!=b['legal'][bi]).any(1)
 np.savez_compressed(DEST/'10_changed_edges.npz',common_codes=common[different|ld],original_rows=ai[different|ld],rebuilt_rows=bi[different|ld],legal_changed=ld[different|ld])
 examples=[]
 for j in np.flatnonzero(different|ld)[:40]:
  examples.append(dict(correspondence_pair=[int(common[j]//10000),int(common[j]%10000)],original_probabilities=a['probabilities'][ai[j]].tolist(),rebuilt_probabilities=b['probabilities'][bi[j]].tolist(),original_legal=a['legal'][ai[j]].tolist(),rebuilt_legal=b['legal'][bi[j]].tolist()))
 report=dict(common_edges=len(common),original_mapped_only_edges=len(a['codes'])-len(common),rebuilt_mapped_only_edges=len(b['codes'])-len(common),exact_probability_differing_edges=int(different.sum()),probability_difference_over_1e_minus8=int((delta.max(1)>1e-8).sum()),maximum_probability_difference=float(delta.max()),legal_differing_edges=int(ld.sum()),legal_differences_by_class=(a['legal'][ai]!=b['legal'][bi]).sum(0).tolist(),original_only_fully_mapped_resource_rows=sum((a['resources']-b['resources']).values()),rebuilt_only_fully_mapped_resource_rows=sum((b['resources']-a['resources']).values()),examples=examples,original={k:a[k] for k in ('total_edges','unmapped_edges','resource_rows','unmapped_resources','manifest_sha256')},rebuilt={k:b[k] for k in ('total_edges','unmapped_edges','resource_rows','unmapped_resources','manifest_sha256')},correspondence_sha256=digest(DEST/'10_graph_node_correspondence.json'),scope='Stored edges/legal/probabilities and resource membership on unique rounded diagnostic node correspondences. Excludes unmapped nodes; not proof of full graph equivalence or solver tie alone.')
 write(DEST/'10_constraint_comparison.json',report);print({k:v for k,v in report.items() if k!='examples'},flush=True)

if __name__=='__main__':main()
