"""Input-only local dimensionality of every projected candidate base plane."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    key='new_fcaf2af1abee';source=BASE/'curved_upright_probe_v3'/key;r=read(source/'receipt.json');assert digest(source/'profiles.json')==r['profiles_sha256'];profiles=read(source/'profiles.json');selected=sorted({i for p in profiles for i in p['face_indices']});folder=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);binding=read(folder/'input_freeze.json');path=BASE/'inputs'/key/'xyz_mm.npy';assert digest(path)==r['input_sha256'];xyz=np.load(path,mmap_mode='r')
    for name in ['surfaces.json','surface_support.npz']:assert digest(folder/name)==binding['files'][name]
    faces=read(folder/'surfaces.json');rows=[]
    with np.load(folder/'surface_support.npz') as support:
        for face in selected:
            f=faces[face];points=xyz[support[f['point_ids_key']]];uv=np.unique((points-np.asarray(f['center']))@np.asarray(f['basis']),axis=0);tree=cKDTree(uv);ratios=[];radii=[]
            for start in range(0,len(uv),4096):
                distance,ids=tree.query(uv[start:start+4096],k=min(16,len(uv)),workers=2);neighbors=uv[ids];delta=neighbors-neighbors.mean(1,keepdims=True);cov=np.einsum('nki,nkj->nij',delta,delta)/delta.shape[1];eigen=np.linalg.eigvalsh(cov);ratio=np.maximum(eigen[:,0],0)/np.maximum(eigen[:,1],np.finfo(float).tiny);ratios.extend(ratio.tolist());radii.extend(distance[:,-1].tolist())
            rows.append(dict(face=face,points=len(points),unique_uv=len(uv),local_covariance_minor_major_quantiles=np.quantile(ratios,[0,.25,.5,.75,.95,1]).tolist(),neighbor_radius_mm_quantiles=np.quantile(radii,[0,.5,.95,1]).tolist()));print(face,len(uv),rows[-1]['local_covariance_minor_major_quantiles'][2],flush=True)
    out=BASE/'curved_base_plane_support_shape_v1';out.mkdir(exist_ok=False);write(out/'report.json',dict(rows=rows,source_receipt_sha256=digest(source/'receipt.json'),surface_freeze_sha256=digest(folder/'input_freeze.json'),input_sha256=digest(path),script_sha256=digest(__file__),scope='All30 candidate base levels; local16NN 2D covariance on original assigned plane points. Descriptive input evidence, no plane rejection threshold or material/bottom identity asserted.',GT_reads=0))

if __name__=='__main__':main()
