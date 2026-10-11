"""Pure-input interval context for frozen known-edge training tables."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
from weld_pipeline.sjg_stream_profile_geometry_1149 import read_profile_geometry
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.sjg_joint_link_runtime_1107 import node_geometry
D=ROOT/'server_rebuild/connection_annotation_features_v1';OUT=ROOT/'server_rebuild/connection_interval_context_v1'

def features(a,b,ends,length,ratios,weld):
 distances=np.linalg.norm(ends[a,:,None,:]-ends[b,None,:,:],axis=3).reshape(-1,4)
 ratio=np.minimum(length[a],length[b])/np.maximum(length[a],length[b])
 return np.column_stack([length[a],length[b],ratios[a],ratios[b],distances,ratio,weld[a]==weld[b]]).astype(np.float32)

def controls():
 rng=np.random.default_rng(1031);ends=rng.normal(size=(4,2,3));length=np.linalg.norm(ends[:,1]-ends[:,0],axis=1);ratios=rng.uniform(size=(4,2));weld=np.array(['a','a','b','b']);a=np.array([0,1]);b=np.array([2,3]);q,_=np.linalg.qr(rng.normal(size=(3,3)))
 x=features(a,b,ends,length,ratios,weld);z=features(a,b,ends@q+np.array([100.,-30.,45.]),length,ratios,weld);assert np.allclose(x,z,atol=1e-6)
 assert x.shape==(2,12)

def main():
 controls();OUT.mkdir(exist_ok=False)
 write(OUT/'protocol.json',dict(columns=['length_a_mm','length_b_mm','a_lo_over_carrier_length','a_hi_over_carrier_length','b_lo_over_carrier_length','b_hi_over_carrier_length','distance_start_start_mm','distance_start_end_mm','distance_end_start_mm','distance_end_end_mm','min_max_length_ratio','same_predicted_weld_type'],runtime_inputs_only=True,rigid_transform_control_pass=True,unknown_mask_unchanged=True))
 reports=[]
 for key in ('21','10','25','36'):
  receipt=read(D/key/'receipt.json');n=receipt['training_rows'];assert digest(D/key/'edge_indices.i64')==receipt['files']['edge_indices.i64']
  base=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime'/('source_'+key);folder=base/'run379'/('test_'+key);nodes=read(folder/'selected_graph.json')['nodes']
  pp=base/'run185'/('source_'+key)/'profiles.json';gp=base/'run285'/('source_'+key+'_geometry.npz');profiles=read_profile_geometry(pp,retain_fields=('support_intervals_mm','unobserved_gaps_mm'))
  with np.load(gp) as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
  ends=np.array([node_geometry(node,graph)[0] for node in nodes]);length=np.array([abs(node['hi']-node['lo']) for node in nodes]);carrier=np.array([graph.curve(graph.profile(node['fragment'])).length for node in nodes]);assert np.all(length>0)&np.all(carrier>0)
  ratios=np.array([[node['lo'],node['hi']] for node in nodes])/carrier[:,None];weld=np.array([node['weld_type'] for node in nodes]);indices=np.memmap(D/key/'edge_indices.i64',dtype='<i8',mode='r',shape=(n,));edges=np.memmap(folder/'edges.i32',dtype='<i4',mode='r',shape=(receipt['original_edges'],2))
  out=OUT/key;out.mkdir();f=out/'context.npy';dest=np.lib.format.open_memmap(f,mode='w+',dtype=np.float32,shape=(n,12))
  for first in range(0,n,65536):
   a,b=edges[indices[first:first+65536]].T;dest[first:first+len(a)]=features(a,b,ends,length,ratios,weld)
  assert np.isfinite(dest).all();dest.flush();del dest
  report=dict(source=key,rows=n,columns=12,context_sha256=digest(f),edge_indices_sha256=digest(D/key/'edge_indices.i64'),geometry_sha256=digest(gp),profiles_sha256=digest(pp),nodes_sha256=digest(folder/'selected_graph.json'));write(out/'receipt.json',report);reports.append(report);print(report,flush=True)
 write(OUT/'receipt.json',dict(status='PURE_INPUT_CONTEXT_READY',cases=reports,fit_calls=0,protected_DEV_reads=0))

if __name__=='__main__':main()
