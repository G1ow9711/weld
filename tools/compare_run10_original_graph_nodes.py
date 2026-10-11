"""Diagnostic node correspondence; rounding is not runtime merging."""
import json
from collections import defaultdict
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest
from weld_pipeline.sjg_stream_profile_geometry_1149 import read_profile_geometry

def rounded(v):
 if isinstance(v,float):return round(v,6)
 if isinstance(v,dict):return {k:rounded(x) for k,x in v.items()}
 if isinstance(v,list):return [rounded(x) for x in v]
 return v

def load(base):
 path=base/'run379/test_10/selected_graph.json';graph=read(path);profiles=read_profile_geometry(base/'run185/source_10/profiles.json')
 with np.load(base/'run285/source_10_geometry.npz') as z:offsets=z['offsets']
 groups=defaultdict(list)
 for i,n in enumerate(graph['nodes']):
  pid=int(np.searchsorted(offsets,n['fragment'],side='right')-1)
  identity=dict(profile=profiles[pid],node={k:v for k,v in n.items() if k not in ('object_row','fragment','probability')})
  groups[json.dumps(rounded(identity),sort_keys=True,separators=(',',':'))].append(i)
 return graph,groups,digest(path)

def main():
 old,og,oh=load(OUT);new,ng,nh=load(ROOT/'server_rebuild/best_baseline_restore/legacy_runtime/source_10')
 unique=[k for k in og.keys()&ng.keys() if len(og[k])==len(ng[k])==1];a=set(old['selection']['selected_nodes']);b=set(new['selection']['selected_nodes']);pairs=[]
 for k in unique:
  i=og[k][0];j=ng[k][0];pairs.append(dict(original=i,rebuilt=j,original_selected=i in a,rebuilt_selected=j in b,probability_absolute_difference=abs(old['nodes'][i]['probability']-new['nodes'][j]['probability'])))
 out=ROOT/'server_rebuild/run446_uploaded_vs_rebuilt_audit_v1'
 report=dict(original_nodes=len(old['nodes']),rebuilt_nodes=len(new['nodes']),unique_diagnostic_correspondences=len(pairs),original_unpaired=len(old['nodes'])-len(pairs),rebuilt_unpaired=len(new['nodes'])-len(pairs),selected_status_differences=sum(r['original_selected']!=r['rebuilt_selected'] for r in pairs),max_probability_difference=max((r['probability_absolute_difference'] for r in pairs),default=0),pairs=pairs,original_graph_sha256=oh,rebuilt_graph_sha256=nh,scope='1e-6 decimal rounded profile geometry/face IDs and node interval/side flags. Diagnostic correspondence only, no runtime merging or proof of full optimization equivalence.')
 write(out/'10_graph_node_correspondence.json',report);print({k:v for k,v in report.items() if k!='pairs'},flush=True)

if __name__=='__main__':main()
