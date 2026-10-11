"""Measure original geometry at stored changed legal gap edges."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest
from weld_pipeline.sjg_stream_profile_geometry_1149 import read_profile_geometry
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.sjg_joint_link_runtime_1107 import node_geometry
DEST=ROOT/'server_rebuild/run446_uploaded_vs_rebuilt_audit_v1'

def geometry(base):
 profiles=read_profile_geometry(base/'run185/source_10/profiles.json',retain_fields=('support_intervals_mm','unobserved_gaps_mm'))
 with np.load(base/'run285/source_10_geometry.npz') as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
 nodes=read(base/'run379/test_10/selected_graph.json')['nodes']
 return graph,nodes

def main():
 changes=np.load(DEST/'10_changed_edges.npz');codes=changes['common_codes'][changes['legal_changed']];pairs=read(DEST/'10_graph_node_correspondence.json')['pairs'];measurements=[]
 for label,base in [('original',OUT),('rebuilt',ROOT/'server_rebuild/best_baseline_restore/legacy_runtime/source_10')]:
  graph,nodes=geometry(base);ends={};distances=[]
  for code in codes:
   a,b=int(code//10000),int(code%10000);i=pairs[a][label];j=pairs[b][label]
   for n in (i,j):
    if n not in ends:ends[n]=node_geometry(nodes[n],graph)[0]
   distances.append(float(np.linalg.norm(ends[i][-1]-ends[j][0])))
  measurements.append(np.array(distances));del graph,nodes
 rows=[dict(correspondence_pair=[int(c//10000),int(c%10000)],original_distance_mm=float(a),rebuilt_distance_mm=float(b)) for c,a,b in zip(codes,*measurements)]
 a,b=measurements;report=dict(changed_gap_edges=len(rows),zero_vs_nonzero=int(((a==0)!=(b==0)).sum()),both_distance_le1e_minus6=int(((a<=1e-6)&(b<=1e-6)).sum()),max_original_distance_mm=float(a.max()),max_rebuilt_distance_mm=float(b.max()),rows=rows,scope='Distances from unchanged original node_geometry on corresponding nodes; no tolerance or stored legality modifications.',changed_edge_sha256=digest(DEST/'10_changed_edges.npz'))
 write(DEST/'10_changed_gap_geometry.json',report);print({k:v for k,v in report.items() if k!='rows'},flush=True)

if __name__=='__main__':main()
