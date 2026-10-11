from pathlib import Path
import numpy as np
from weld_pipeline.server_raw_tube_endpoint_events import RawTubeEndpointEvents
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_runtime_904 import write,digest

def curve(a,b):return CurveCarrier(dict(kind='line',endpoints_mm=[a,b]))
def main():
 rng=np.random.default_rng(1031);xyz=rng.uniform([-2,-5,-5],[32,5,5],(2000,3));c=curve([0,0,0],[30,0,0]);q,ids=RawTubeEndpointEvents(xyz).events(c);delta=xyz-c.ends[0];t=delta@c.direction;radial=np.linalg.norm(delta-t[:,None]*c.direction,axis=1);expected=np.unique(t[(t>=0)&(t<=c.length)&(radial<=3)]);np.testing.assert_array_equal(q,expected);np.testing.assert_array_equal(q,t[ids]);R,_=np.linalg.qr(rng.normal(size=(3,3)));shift=np.array([31.,-14.,6.]);moved=xyz@R.T+shift;transformed=curve((c.ends[0]@R.T+shift).tolist(),(c.ends[1]@R.T+shift).tolist());rotq,_=RawTubeEndpointEvents(moved).events(transformed);np.testing.assert_allclose(rotq,q,atol=1e-10);revq,_=RawTubeEndpointEvents(xyz).events(curve([30,0,0],[0,0,0]));np.testing.assert_allclose(revq,30-q[::-1],atol=1e-10)
 out=Path('server_rebuild/expansion_v1/raw_tube_event_controls_v1');out.mkdir(exist_ok=False);write(out/'receipt.json',dict(status='PASS',brute_force_exact=True,raw_projection_exact=True,rigid_transform=True,carrier_reversal=True,random_points=len(xyz),selected_events=len(q),module_sha256=digest('weld_pipeline/server_raw_tube_endpoint_events.py'),scope='Input candidate generation only; not full-pipeline invariance or weldability'))
if __name__=='__main__':main()
