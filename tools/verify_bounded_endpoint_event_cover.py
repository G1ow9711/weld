from pathlib import Path
import numpy as np
from weld_pipeline.server_bounded_endpoint_event_cover import bounded_event_cover
from weld_pipeline.sjg_local_runtime_904 import write,digest

def main():
 rng=np.random.default_rng(1031);reports=[]
 for length in [0.1,1.,30.,37.123,24000.]:
  q=np.unique(np.r_[rng.uniform(0,length,12000),np.linspace(0,length,int(np.ceil(length))+1)]);anchors=q[::311];got=bounded_event_cover(q,length,anchors);assert np.isin(got,q).all() and np.isin(anchors,got).all();i=np.searchsorted(got,q);near=np.minimum(abs(q-got[np.minimum(i,len(got)-1)]),abs(q-got[np.maximum(i-1,0)]));assert near.max()<=.5+1e-8;rev=bounded_event_cover(length-q[::-1],length,length-anchors[::-1]);np.testing.assert_allclose(rev,length-got[::-1],atol=1e-8,rtol=0);reports.append(dict(length_mm=length,original=len(q),kept=len(got),max_nearest_kept_mm=float(near.max())))
 out=Path('server_rebuild/expansion_v1/bounded_event_cover_controls_v1');out.mkdir(exist_ok=False);write(out/'receipt.json',dict(status='PASS',cases=reports,anchors_exactly_preserved=True,no_new_coordinates=True,conditional_reversal=True,module_sha256=digest('weld_pipeline/server_bounded_endpoint_event_cover.py'),scope='Input event cover only; no guarantee original3mm reference membership preserved'))
if __name__=='__main__':main()
