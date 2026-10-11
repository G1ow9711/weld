"""Controls for numerical gap ambiguity, not physical-gap/contact validation."""
import numpy as np
from scipy.spatial.transform import Rotation
from weld_pipeline.server_roundoff_gap_links import resolved_positive_gap
from weld_pipeline.sjg_local_runtime_904 import ROOT,write,digest

def main():
    a=np.array([1000.,2000.,-500.]);length=3000.
    offsets=np.array([0.,1e-12,1e-8,1e-5,.01,1.]);b=a+offsets[:,None]*np.array([1.,0,0])
    actual,tol=resolved_positive_gap(a,b,length);expected=np.array([False,False,True,True,True,True]);assert np.array_equal(actual,expected)
    reverse,_=resolved_positive_gap(b,a,length);assert np.array_equal(reverse,actual)
    # Rotations and moderate representable translations: real micrometre gaps
    # remain distinct while sub-roundoff differences remain unresolved.
    rng=np.random.default_rng(1031)
    for _ in range(40):
        r=Rotation.random(random_state=rng).as_matrix();shift=rng.uniform(-10000,10000,3)
        result,_=resolved_positive_gap(a@r.T+shift,b@r.T+shift,length)
        assert np.array_equal(result,expected)
    zero,_=resolved_positive_gap(a,a,length);assert not zero
    dest=ROOT/'server_rebuild/roundoff_gap_controls_v2';dest.mkdir(exist_ok=False)
    write(dest/'receipt.json',dict(status='PASS',controls=['exact shared','sub-roundoff1e-12mm unresolved','resolved1e-8mm through1mm retained','endpoint reversal','40 rigid transforms'],module_sha256=digest(ROOT/'weld_pipeline/server_roundoff_gap_links.py'),scope='Numerical predicate only. No model accuracy, full-link invariance or physical contact proof.'))
    print('PASS',actual.tolist(),flush=True)

if __name__=='__main__':main()
