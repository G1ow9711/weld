"""Trace selected links of the observed erroneous extension under both inputs."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
BASE=ROOT/'server_rebuild';NEW=BASE/'roundoff_gap_baseline_probe_v1/legacy_runtime/source_10/run379/test_10';OLD=BASE/'best_baseline_restore/legacy_runtime/source_10/run379/test_10'

def main():
 meta=read(NEW/'parent_metadata.json')[1231];nodes=meta['graph_nodes'];rows=[];stored={}
 for label,folder in [('original',OLD),('guard',NEW)]:
  manifest=read(folder/'edge_manifest.json');n=manifest['edge_count'];e=np.memmap(folder/'edges.i32',dtype='<i4',mode='r',shape=(n,2));p=np.memmap(folder/'probabilities.f64',dtype='<f8',mode='r',shape=(n,4));l=np.memmap(folder/'legal.u8',dtype='u1',mode='r',shape=(n,3));selected={tuple(v) for v in read(folder/'selected_graph.json')['selection']['selected_links']};stored[label]=(e,p,l,selected)
 for a,b in zip(nodes[:-1],nodes[1:]):
  row=dict(source_node=a,destination_node=b)
  for label,(e,p,l,selected) in stored.items():
   ix=np.flatnonzero((e[:,0]==a)&(e[:,1]==b));assert len(ix)==1;i=int(ix[0]);prob=p[i];clipped=np.clip(prob,1e-6,1-1e-6)
   row[label]=dict(probabilities=prob.tolist(),legal=l[i].tolist(),log_odds_vs_no_link=np.log(clipped[1:]/clipped[0]).tolist(),selected_kinds=[k for k in (1,2,3) if (a,b,k) in selected])
  rows.append(row)
 report=dict(candidate_parent=1231,output_parent=66,graph_nodes=nodes,face_pairs=meta['face_pairs'],links=rows,scope='Frozen input link changes for offline error attribution. Faces and geometry alone do not prove physical parent identity; reference labels never modify output.',original_manifest_sha256=digest(OLD/'edge_manifest.json'),guard_manifest_sha256=digest(NEW/'edge_manifest.json'));write(BASE/'roundoff_gap_baseline_probe_v1/extension_edge_trace.json',report);print(rows,flush=True)

if __name__=='__main__':main()
