"""Verify runtime/training interval context parity on all frozen known rows."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
from weld_pipeline.sjg_stream_profile_geometry_1149 import read_profile_geometry
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.server_connection_interval_context import node_context,interval_features,ContextConnectionAdapter


def main():
 reports=[];data=ROOT/'server_rebuild/connection_annotation_features_v1';context=ROOT/'server_rebuild/connection_interval_context_v1'
 for key in ('21','10','25','36'):
  r=read(data/key/'receipt.json');cr=read(context/key/'receipt.json');assert digest(context/key/'context.npy')==cr['context_sha256']
  base=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime'/('source_'+key);folder=base/'run379'/('test_'+key);nodes=read(folder/'selected_graph.json')['nodes']
  profiles=read_profile_geometry(base/'run185'/('source_'+key)/'profiles.json',retain_fields=('support_intervals_mm','unobserved_gaps_mm'))
  with np.load(base/'run285'/('source_'+key+'_geometry.npz')) as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
  state=node_context(nodes,graph);expected=np.load(context/key/'context.npy',mmap_mode='r');ix=np.memmap(data/key/'edge_indices.i64',dtype='<i8',mode='r',shape=(r['training_rows'],));edges=np.memmap(folder/'edges.i32',dtype='<i4',mode='r',shape=(r['original_edges'],2))
  for first in range(0,len(ix),65536):
   a,b=edges[ix[first:first+65536]].T;actual=interval_features(a,b,*state);assert np.array_equal(actual,expected[first:first+len(a)])
  reports.append(dict(source=key,rows=len(ix),bitwise_equal=True));print(reports[-1],flush=True)
 class Original:
  n_features_in_=32;classes_=np.arange(4)
  def predict_proba(self,x):assert x.shape[1]==32;return np.tile([.6,.1,.2,.1],(len(x),1))
 class Gate:
  n_features_in_=44;classes_=np.array([0,1])
  def predict_proba(self,x):assert x.shape[1]==44;return np.tile([.2,.8],(len(x),1))
 x=np.zeros((2,44));x[0,[9,11]]=1;p=ContextConnectionAdapter(Original(),Gate()).predict_proba(x)
 assert np.allclose(p[0],[.2,.2,.4,.2]) and np.allclose(p[1],[.6,.1,.2,.1])
 write(context/'runtime_parity.json',dict(status='PASS',cases=reports,module_sha256=digest(ROOT/'weld_pipeline/server_connection_interval_context.py'),mass_and_arc_controls_pass=True,fit_calls=0,protected_DEV_reads=0))

if __name__=='__main__':main()
