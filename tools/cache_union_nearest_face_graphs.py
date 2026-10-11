"""Freeze all observed face graphs from input alone before reference diagnosis."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_union_nearest_graph import local_spacing_graph
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main(key='new_ac2008a03607'):
    started=time.perf_counter()
    out=BASE/'union_nearest_face_graphs_v1'/key;out.mkdir(parents=True,exist_ok=False)
    folder=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key)
    freeze=read(folder/'input_freeze.json');path=BASE/'inputs'/key/'xyz_mm.npy'
    assert digest(path)==freeze['input_sha256']
    for name in ('surfaces.json','surface_support.npz'):assert digest(folder/name)==freeze['files'][name]
    write(out/'protocol.json',dict(key=key,input_sha256=freeze['input_sha256'],source_freeze_sha256=digest(folder/'input_freeze.json'),module_sha256=digest('weld_pipeline/server_union_nearest_graph.py'),base_module_sha256=digest('weld_pipeline/server_local_spacing_graph.py'),selection='All observed faces with nonzero normal and point support. No label-based face selection.',GT_reads=0))
    xyz=np.load(path,mmap_mode='r');faces=read(folder/'surfaces.json');rows=[]
    with np.load(folder/'surface_support.npz') as support:
        for fi,face in enumerate(faces):
            n=np.asarray(face.get('normal',[0,0,0]),float);norm=np.linalg.norm(n);skey=face.get('point_ids_key')
            if norm==0 or skey not in support:continue
            points=np.asarray(xyz[support[skey]],np.float64)
            if not len(points):continue
            n/=norm;origin=points.mean(0);axis=np.eye(3)[np.argmin(abs(n))];u=axis-n*(axis@n);u/=np.linalg.norm(u);basis=np.column_stack((u,np.cross(n,u)))
            uv,edges,spacing=local_spacing_graph((points-origin)@basis)
            target=out/f'face_{fi}.npz';np.savez_compressed(target,uv=uv,edges=edges,spacing=spacing,origin=origin,basis=basis)
            rows.append(dict(face=fi,vertices=len(uv),edges=len(edges),file=target.name,sha256=digest(target)))
            if fi%50==0:print(fi,'faces',len(rows),'edges',sum(r['edges'] for r in rows),flush=True)
    write(out/'receipt.json',dict(status='INPUT_GRAPHS_FROZEN',faces=rows,protocol_sha256=digest(out/'protocol.json'),GT_reads=0,seconds=time.perf_counter()-started))
    print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--key',choices=['new_ac2008a03607','new_fcaf2af1abee'],default='new_ac2008a03607');args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.key)
