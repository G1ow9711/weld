from pathlib import Path
import numpy as np
from weld_pipeline.server_local_spacing_graph import local_spacing_graph
from weld_pipeline.server_local_graph_arc_candidates import prune_and_trace, fit_chain_arc
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import write,digest

def fits(p):
    v,e,_=local_spacing_graph(p);chains,kept=prune_and_trace(v,e)
    candidates=[fit_chain_arc(v,c,np.zeros(3),np.eye(3)[:,:2]) for c in chains]
    return [c for c in candidates if c],chains,kept

def main():
    t=np.linspace(.2,1.7,81);p=np.column_stack((61*np.cos(t),61*np.sin(t)))+np.array([12,-7])
    result,chains,kept=fits(p);assert len(result)==1
    arc=CircularArc.from_parameters(result[0]['arc_parameters']);assert abs(arc.radius_mm-61)<1e-8
    assert np.linalg.norm(arc.center_mm-[12,-7,0])<1e-8
    assert abs(abs(arc.sweep_rad)-1.5)<1e-8
    circle=np.column_stack((np.cos(np.arange(80)*2*np.pi/80),np.sin(np.arange(80)*2*np.pi/80)))
    result,_,_=fits(circle);assert len(result)==1 and result[0]['arc_parameters']['is_closed']
    result,_,_=fits(np.column_stack((np.arange(20),np.zeros(20))));assert not result
    result,_,_=fits(np.concatenate((p[:30],p[50:])));assert len(result)==2
    out=Path('server_rebuild/expansion_v1/local_graph_arc_controls_v1');out.mkdir(exist_ok=False)
    write(out/'receipt.json',dict(status='PASS',checks=['known_open_arc','closed_circle','collinear_rejected','gap_two_arcs'],module_sha256=digest('weld_pipeline/server_local_graph_arc_candidates.py'),scope='Analytic candidate construction only; no weld or contact classification.'))
    print('PASS')

if __name__=='__main__':main()
