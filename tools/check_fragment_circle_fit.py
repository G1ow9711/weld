"""Fixed-budget synthetic estimator check, not end-to-end surface inference."""
from pathlib import Path
import numpy as np
from weld_pipeline.server_fragment_circle_fit import fit_fragment_circle
from weld_pipeline.sjg_local_runtime_904 import write,digest


def make(theta,radius=2000.):
    return radius*np.column_stack((np.cos(theta),np.sin(theta)))


def main():
    out=Path('server_rebuild/expansion_v1/fragment_circle_fit_controls_v1');out.mkdir(exist_ok=False)
    fragments=[make(np.linspace(a,a+.035,30)) for a in [-.2,-.0175,.165]]
    original=fit_fragment_circle(fragments);assert original is not None
    assert np.linalg.norm(original['center'])<1e-6 and abs(original['radius']-2000)<1e-6
    reversed_fit=fit_fragment_circle([p[::-1] for p in fragments[::-1]])
    duplicates=fit_fragment_circle([np.repeat(p,3,axis=0) for p in fragments])
    angle=.7;q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    shift=np.array([1234.,-5678.]);rigid=fit_fragment_circle([p@q.T+shift for p in fragments])
    assert np.allclose(reversed_fit['center'],original['center'],atol=1e-6)
    assert np.allclose(duplicates['center'],original['center'],atol=1e-6)
    assert np.allclose(rigid['center'],np.asarray(original['center'])@q.T+shift,atol=1e-6)
    assert abs(rigid['radius']-original['radius'])<1e-6
    rows=[]
    # Equal 90 observed points and 3 local fragments in both configurations.
    # Fixed random seeds/noise levels; truth used solely for error measurement.
    for noise in [.05,.5]:
        for spread,starts in [('compact',[-.0525,-.0175,.0175]),('separated',[-.2,-.0175,.165])]:
            for seed in range(30):
                rng=np.random.default_rng(1031+seed)
                clean=[make(np.linspace(a,a+.035,30)) for a in starts]
                observed=[p+rng.normal(0,noise,p.shape) for p in clean]
                joint=fit_fragment_circle(observed);single=[fit_fragment_circle([p]) for p in observed]
                def metrics(f,truth):
                    if f is None:return None
                    error=np.abs(np.linalg.norm(truth-np.array(f['center']),axis=1)-f['radius'])
                    return dict(radius_error_mm=abs(f['radius']-2000),support_position_max_mm=float(error.max()))
                rows.append(dict(noise_mm=noise,layout=spread,seed=1031+seed,joint=metrics(joint,np.concatenate(clean)),independent=[metrics(f,p) for f,p in zip(single,clean)]))
    # Nearby distinct concentric contours can have low residual after wrong merge.
    angles=np.linspace(-.15,.15,90)
    mixed=fit_fragment_circle([make(angles,1999),make(angles,2001)])
    assert mixed is not None and mixed['fragments'][0]['mean_radial_error_mm']<0<mixed['fragments'][1]['mean_radial_error_mm']
    write(out/'report.json',dict(rows=rows,mixed_contour_fit=mixed,controls='PASS analytic circle, fragment/order reversal, exact duplicate density, rigid transform, mixed contour opposite radial bias',module_sha256=digest('weld_pipeline/server_fragment_circle_fit.py'),scope='Known analytic contour fragments supplied; NOT raw XYZ surface boundary extraction or identity inference. No support gaps filled; no finite arcs emitted. No real labels read.',GT_reads=0))
    for noise in [.05,.5]:
        for layout in ['compact','separated']:
            records=[r for r in rows if r['noise_mm']==noise and r['layout']==layout]
            values=[r['joint']['support_position_max_mm'] for r in records if r['joint']]
            baseline=[f['support_position_max_mm'] for r in records for f in r['independent'] if f]
            print(noise,layout,'joint successes',len(values),'median support error',float(np.median(values)),'single successes',len(baseline),'median error',float(np.median(baseline)),flush=True)
    print('PASS; mixed nearby contour residual does not prove physical identity',flush=True)

if __name__=='__main__':main()
