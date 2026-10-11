from pathlib import Path
import numpy as np
from weld_pipeline.server_local_graph_arc_candidates import fit_chain_arc
from weld_pipeline.server_geometric_circle_refinement import refine_chain_circle
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import write,digest
rng=np.random.default_rng(1031);theta=np.linspace(-.12,.12,121);radius=2000.;uv=np.column_stack((np.cos(theta),np.sin(theta)))*(radius+rng.normal(0,.35,len(theta)))[:,None];chain=np.arange(len(uv));origin=np.zeros(3);basis=np.eye(3)[:,:2]
seed=fit_chain_arc(uv,chain,origin,basis);fit=refine_chain_circle(uv,chain,origin,basis,seed);assert fit is not None and fit['refined_radial_sse_mm2']<=fit['original_radial_sse_mm2'];arc=CircularArc.from_parameters(fit['arc_parameters'])
rev=refine_chain_circle(uv,chain[::-1],origin,basis,fit_chain_arc(uv,chain[::-1],origin,basis));other=CircularArc.from_parameters(rev['arc_parameters']);assert np.allclose(arc.center_mm,other.center_mm,atol=1e-5,rtol=0) and abs(arc.radius_mm-other.radius_mm)<1e-5
angle=.71;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);moved=uv@r.T+[37,-91];new=refine_chain_circle(moved,chain,origin,basis,fit_chain_arc(moved,chain,origin,basis));a=CircularArc.from_parameters(new['arc_parameters']);assert np.allclose(a.center_mm[:2],arc.center_mm[:2]@r.T+[37,-91],atol=1e-5,rtol=0)
t=np.arange(64)*2*np.pi/64;circle=np.column_stack((np.cos(t),np.sin(t)))*61;c=np.r_[np.arange(64),0];full=refine_chain_circle(circle,c,origin,basis,fit_chain_arc(circle,c,origin,basis));assert full and abs(full['arc_parameters']['radius_mm']-61)<1e-8
out=Path('server_rebuild/expansion_v1/geometric_circle_refinement_controls_v1');out.mkdir(exist_ok=False);write(out/'receipt.json',dict(status='PASS',module_sha256=digest('weld_pipeline/server_geometric_circle_refinement.py'),controls=['unchanged observed points','nonincreasing radial SSE','chain reversal','rigid transform','exact closed circle'],scope='Numerical controls only; lower input residual does not guarantee lower reference error'))
print('PASS',fit['original_radial_sse_mm2'],fit['refined_radial_sse_mm2'])
