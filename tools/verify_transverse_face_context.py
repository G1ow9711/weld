from pathlib import Path
from types import SimpleNamespace
import numpy as np
from weld_pipeline.server_transverse_face_context import transverse_face_descriptor,transverse_endpoint_context
from weld_pipeline.sjg_local_runtime_904 import write,digest

def main():
 ring=np.array([[0.,0.],[20.,0.],[20.,30.],[0.,30.]]);edges=np.stack((ring,np.roll(ring,-1,axis=0)),1);origin=np.zeros(3);basis=np.eye(3)[:,:2];boundary=(origin,basis,edges);p=np.array([10.,0.,0.]);t=np.array([1.,0.,0.]);n=np.array([0.,0.,1.]);reference=transverse_face_descriptor(p,t,n,0,boundary);assert reference[3]==1 and reference[4]==1;np.testing.assert_allclose(reference[-2:],[.03,.03],atol=1e-12)
 for tn,nn in [(t,-n),(-t,n),(-t,-n)]:np.testing.assert_allclose(transverse_face_descriptor(p,tn,nn,0,boundary),reference,atol=1e-10)
 rng=np.random.default_rng(1031);R,_=np.linalg.qr(rng.normal(size=(3,3)));shift=np.array([111.,-50.,31.]);moved=(origin@R.T+shift,R@basis,edges);got=transverse_face_descriptor(p@R.T+shift,t@R.T,n@R.T,float((n@R.T)@shift),moved);np.testing.assert_allclose(got,reference,atol=1e-9)
 hole=np.array([[8.,10.],[12.,10.],[12.,20.],[8.,20.]]);hole_edges=np.stack((hole,np.roll(hole,-1,axis=0)),1);with_hole=transverse_face_descriptor(p,t,n,0,(origin,basis,np.concatenate((edges,hole_edges))));np.testing.assert_allclose(with_hole[-3:],[np.log(4),.01,.03],atol=1e-12)
 collinear=transverse_face_descriptor(np.zeros(3),t,n,0,boundary);assert collinear[5]==1;np.testing.assert_allclose(collinear[-2:],[.03,.03],atol=1e-12)
 far=transverse_face_descriptor(p+[0,0,4],t,n,0,boundary);assert far[3]==0
 g=SimpleNamespace(normals=np.array([n,-n]),offsets=np.zeros(2),boundaries={0:boundary,1:boundary});np.testing.assert_array_equal(transverse_endpoint_context(p,t,[0,1],g),transverse_endpoint_context(p,t,[1,0],g))
 out=Path('server_rebuild/expansion_v1/transverse_face_context_controls_v1');out.mkdir(exist_ok=True);write(out/'receipt.json',dict(status='PASS',rectangle_distance_mm=30.,hole_crossings=True,collinear_boundary=True,normal_sign=True,tangent_reversal=True,rigid_transform=True,face_permutation=True,remote_plane_mask=True,module_sha256=digest('weld_pipeline/server_transverse_face_context.py'),scope='Observed polygon-ray input descriptors only, not physical plate height/hole truth or full pipeline invariance'))
if __name__=='__main__':main()
