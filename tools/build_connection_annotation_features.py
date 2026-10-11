"""Freeze numerically guarded link features aligned to annotation supervision."""
import shutil
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
from weld_pipeline.sjg_stream_profile_geometry_1149 import read_profile_geometry
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.server_roundoff_gap_links import guarded_relative_link_blocks
from tools.prepare_connection_annotation_targets import KEYS, DEST as TARGETS

DEST=ROOT/'server_rebuild/connection_annotation_features_v1'

def main():
 DEST.mkdir(exist_ok=False);reports=[]
 write(DEST/'protocol.json',dict(feature_columns=32,guard_sha256=digest(ROOT/'weld_pipeline/server_roundoff_gap_links.py'),target_receipt_sha256=digest(TARGETS/'receipt.json'),unknown_masked=True,positive_illegal_masked=True,scope='Frozen candidate annotation compatibility features; no fit or inference truth filtering. Physical side identity not certified.'))
 for key in KEYS:
  assert shutil.disk_usage(DEST).free>5.5*1024**3
  base=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime'/('source_'+key);folder=base/'run379'/('test_'+key)
  tr=read(TARGETS/key/'receipt.json');lp=TARGETS/key/'labels.npy';assert digest(lp)==tr['labels_sha256'] and digest(folder/'edges.i32')==tr['edges_sha256']
  labels=np.load(lp);m=len(labels);old=np.memmap(folder/'edges.i32',dtype='<i4',mode='r',shape=(m,2));nodes=read(folder/'selected_graph.json')['nodes'];n=len(nodes)
  codes=np.array(old[:,0],np.int64)*n+old[:,1];order=np.argsort(codes);sorted_codes=codes[order];assert np.all(np.diff(sorted_codes)>0)
  profiles=read_profile_geometry(base/'run185'/('source_'+key)/'profiles.json',retain_fields=('support_intervals_mm','unobserved_gaps_mm'))
  with np.load(base/'run285'/('source_'+key+'_geometry.npz')) as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
  case=dict(graph=graph,surfaces=read(base/'run185'/('source_'+key)/'surfaces.json'),frame_cache={})
  out=DEST/key;out.mkdir();counts=np.zeros(4,np.int64);excluded=0;seen=0
  with (out/'features.f32').open('xb') as fx,(out/'labels.i8').open('xb') as fy,(out/'edge_indices.i64').open('xb') as fi:
   for edges,x,legal in guarded_relative_link_blocks(nodes,case):
    c=edges[:,0].astype(np.int64)*n+edges[:,1];ix=np.searchsorted(sorted_codes,c);assert np.all(ix<m) and np.array_equal(sorted_codes[ix],c)
    original=order[ix];y=labels[original];known=y>=0;positive=np.flatnonzero(y>0);bad=positive[~legal[positive,y[positive]-1]];known[bad]=False;excluded+=len(bad)
    fx.write(x[known].astype('<f4').tobytes());fy.write(y[known].tobytes());fi.write(original[known].astype('<i8').tobytes());counts+=np.bincount(y[known],minlength=4);seen+=len(y)
  old._mmap.close();report=dict(source=key,regenerated_edges=seen,original_edges=m,training_rows=int(counts.sum()),class_counts=counts.tolist(),positive_illegal_masked=excluded,files={p.name:digest(p) for p in out.iterdir()})
  write(out/'receipt.json',report);reports.append(report);print(report,flush=True)
 write(DEST/'receipt.json',dict(status='ANNOTATION_FEATURES_FROZEN',cases=reports,fit_calls=0,protected_DEV_reads=0,promoted=False))

if __name__=='__main__':main()
