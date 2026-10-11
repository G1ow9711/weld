from pathlib import Path
import numpy as np
from weld_pipeline.server_local_spacing_graph import local_spacing_graph as original
from weld_pipeline.server_union_nearest_graph import local_spacing_graph
from weld_pipeline.sjg_local_runtime_904 import write,digest

def main():
    p=np.column_stack((np.array([0.,.1,.2,1.2,2.2]),np.zeros(5)))
    v,e,s=local_spacing_graph(p);ov,oe,os=original(p)
    assert np.array_equal(v,ov) and np.array_equal(s,os)
    assert (2,3) not in {tuple(x) for x in oe} and (2,3) in {tuple(x) for x in e}
    assert {tuple(x) for x in oe}<={tuple(x) for x in e}
    p=np.column_stack((np.r_[np.arange(11),np.arange(30,41)],np.zeros(22)))
    v,e,s=local_spacing_graph(p);assert not np.any((v[e[:,0],0]<=10)&(v[e[:,1],0]>=30))
    p=np.random.default_rng(1031).normal(size=(100,2));v,e,s=local_spacing_graph(p);ov,oe,_=original(p)
    d=np.linalg.norm(v[:,None]-v[None],axis=2)
    expected={tuple(x) for x in oe}|{tuple(sorted((i,j))) for i,j in np.argwhere((d<=s[:,None]*(1+1e-12)) & (d>0))}
    assert {tuple(x) for x in e}==expected
    theta=.37;r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    mv,me,_=local_spacing_graph(p@r.T+3)
    pairs=lambda vs,es:{tuple(sorted(tuple(np.round(t,8)) for t in pair)) for pair in vs[es]}
    assert pairs((mv-3)@r,me)==pairs(v,e)
    out=Path('server_rebuild/expansion_v1/union_nearest_graph_controls_v1');out.mkdir(exist_ok=False)
    write(out/'receipt.json',dict(status='PASS',checks=['density_transition','original_graph_superset','large_gap_disconnected','all_pairs_nearest_oracle','rigid_transform'],module_sha256=digest('weld_pipeline/server_union_nearest_graph.py'),base_module_sha256=digest('weld_pipeline/server_local_spacing_graph.py'),scope='Candidate graph only. Sparse isolated pairs can connect; not material/contact proof.'))
    print('PASS')

if __name__=='__main__':main()
