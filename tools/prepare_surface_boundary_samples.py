"""XYZ-only local curvature enrichment for offline surface supervision."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'server_rebuild/surface_boundary_samples_v1'
def main():
 OUT.mkdir(exist_ok=False);reports=[]
 for key in ['10','25','36']:
  rawpath=ROOT/'server_rebuild/inputs_v1'/key/'xyz_mm.npy';xyz=np.load(rawpath,mmap_mode='r');rng=np.random.default_rng(2032);anchors=rng.choice(len(xyz),min(4096,len(xyz)),replace=False);tree=cKDTree(xyz);distance,neighbors=tree.query(xyz[anchors],k=128,workers=2);points=np.asarray(xyz[neighbors[:,:32]],np.float64);centered=points-points.mean(1,keepdims=True);eig=np.maximum(np.linalg.eigvalsh(np.einsum('bki,bkj->bij',centered,centered)),0);curvature=eig[:,0]/np.maximum(eig.sum(1),1e-20);rank=np.argsort(-curvature,kind='stable');high=rank[:512];remaining=np.setdiff1d(np.arange(len(anchors)),high);uniform=rng.choice(remaining,512,replace=False);selected=np.concatenate((high,uniform));cols=np.stack([rng.choice(128,8,replace=False) for _ in selected]);ids=np.unique(np.concatenate((anchors[selected],neighbors[selected[:,None],cols].ravel())));dest=OUT/key;dest.mkdir();np.savez(dest/'samples.npz',point_indices=ids,anchor_indices=anchors[selected],anchor_curvature=curvature[selected]);r=dict(key=key,samples=len(ids),curvature_anchors=512,uniform_anchors=512,raw_sha256=digest(rawpath),samples_sha256=digest(dest/'samples.npz'));reports.append(r);print(r,flush=True);xyz._mmap.close()
 write(OUT/'receipt.json',dict(cases=reports,seed=2032,GT_reads=0,STEP_reads=0,scope='Input-only curriculum sampling, not a contact detector or validation holdout. Enrich high local covariance curvature while retaining random surface anchors.',source_sha256=digest(__file__)))
if __name__=='__main__':main()
