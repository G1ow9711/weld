from pathlib import Path
import numpy as np
from weld_pipeline.server_local_spacing_graph import local_spacing_graph
from weld_pipeline.sjg_local_runtime_904 import write, digest

def pair_coordinates(v, e):
    return {tuple(sorted(tuple(np.round(p, 8)) for p in pair)) for pair in v[e]}

def main():
    angle = np.r_[np.linspace(0, 1, 81), np.linspace(1.04, 2, 25)]
    p = 61 * np.column_stack((np.cos(angle), np.sin(angle)))
    v, e, scale = local_spacing_graph(p)
    pairs = pair_coordinates(v, e)
    assert all(tuple(sorted((tuple(np.round(a,8)), tuple(np.round(b,8))))) in pairs for a,b in zip(p[:-1],p[1:]) if np.linalg.norm(a-b) <= 2*min(np.linalg.norm(p-a,axis=1)[np.linalg.norm(p-a,axis=1)>0].min(),np.linalg.norm(p-b,axis=1)[np.linalg.norm(p-b,axis=1)>0].min()))
    # Same-density strings separated by a much larger unsampled gap.
    gap = np.column_stack((np.r_[np.arange(11), np.arange(30,41)], np.zeros(22)))
    gv, ge, _ = local_spacing_graph(gap)
    assert not np.any((gv[ge[:,0],0] <= 10) & (gv[ge[:,1],0] >= 30))
    theta=.731; r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    moved=p@r.T+np.array([37.,-12.]); mv,me,_=local_spacing_graph(moved)
    assert pair_coordinates((mv-np.array([37.,-12.]))@r,me)==pairs
    rv,re,_=local_spacing_graph(p[::-1]); assert pair_coordinates(rv,re)==pairs
    dv,de,_=local_spacing_graph(np.concatenate((p,p))); assert pair_coordinates(dv,de)==pairs
    # Direct all-pairs definition on irregular points is an independent oracle.
    q=np.random.default_rng(1031).normal(size=(100,2));qv,qe,qs=local_spacing_graph(q)
    distances=np.linalg.norm(qv[:,None]-qv[None],axis=2)
    expected=np.argwhere(np.triu(distances<=2*np.minimum(qs[:,None],qs[None,:])*(1+1e-12),1))
    assert {tuple(x) for x in qe}=={tuple(x) for x in expected}
    out=Path('server_rebuild/expansion_v1/local_spacing_graph_controls_v1');out.mkdir(exist_ok=False)
    write(out/'receipt.json',dict(status='PASS',checks=['irregular_circle_admissible_adjacency','large_gap_disconnected','rigid_transform','input_reversal','exact_duplicates','brute_force_oracle'],module_sha256=digest('weld_pipeline/server_local_spacing_graph.py'),scope='Candidate graph only; isolated sparse pairs may connect and edges are not material/contact proof.'))
    print('PASS')

if __name__=='__main__': main()
