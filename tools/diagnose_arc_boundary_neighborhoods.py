"""TRAIN-only boundary evidence diagnostic; no generated weld predictions."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def angular_gaps(tree,uv,k):
    result=np.empty(len(uv));radius=np.empty(len(uv))
    for start in range(0,len(uv),4096):
        stop=min(start+4096,len(uv));distance,ids=tree.query(uv[start:stop],k=min(k+1,len(uv)),workers=2)
        delta=uv[ids[:,1:]]-uv[start:stop,None];angles=np.sort(np.arctan2(delta[:,:,1],delta[:,:,0]),axis=1)
        gaps=np.diff(np.concatenate((angles,angles[:,:1]+2*np.pi),axis=1),axis=1)
        result[start:stop]=gaps.max(axis=1);radius[start:stop]=distance[:,-1]
    return result,radius


def describe(gap,radius,ids=None):
    if ids is not None:gap=gap[ids];radius=radius[ids]
    return dict(n=len(gap),largest_gap_degrees_quantiles=np.quantile(np.rad2deg(gap),[0,.25,.5,.75,.95,1]).tolist(),neighbor_radius_mm_quantiles=np.quantile(radius,[0,.5,.95,1]).tolist(),one_sided_halfplane_count=int((gap>=np.pi-1e-10).sum()))


def main():
    key='new_fcaf2af1abee';graphs=BASE/'union_nearest_face_graphs_v1'/key;gr=read(graphs/'receipt.json');assert digest(graphs/'protocol.json')==gr['protocol_sha256'];gm={r['face']:r for r in gr['faces']};target=BASE/'labels'/key/'targets.json';truth=read(target);rows=[];stats=[]
    for face in [0,190,268]:
        path=graphs/gm[face]['file'];assert digest(path)==gm[face]['sha256']
        with np.load(path) as g:uv=g['uv'];origin=g['origin'];basis=g['basis']
        tree=cKDTree(uv);banks={k:angular_gaps(tree,uv,k) for k in [8,16,32]}
        stats.append(dict(face=face,vertices=len(uv),scales={str(k):describe(*v) for k,v in banks.items()}))
        for pi,parent in enumerate(truth['parents']):
            for ci,child in enumerate(parent['children']):
                if child['representation']['kind']!='arc':continue
                arc=CircularArc.from_parameters(child['representation']['arc_parameters']);points=arc.point_at(np.linspace(0,arc.length_mm,int(np.ceil(arc.length_mm))+1));query=(points-origin)@basis;dist,ids=tree.query(query,workers=2)
                plane_distance=abs((points-origin)@np.cross(basis[:,0],basis[:,1]))
                # Preserve all rows, including unsupported faces, to avoid silent filtering.
                rows.append(dict(face=face,parent=pi,child=ci,samples=len(ids),unique_nearest_vertices=len(np.unique(ids)),plane_distance_max_mm=float(plane_distance.max()),nearest_uv_distance_quantiles_mm=np.quantile(dist,[0,.5,.95,1]).tolist(),scales={str(k):describe(*v,ids=ids) for k,v in banks.items()}))
        print('face',face,'vertices',len(uv),'all k16 gap median',stats[-1]['scales']['16']['largest_gap_degrees_quantiles'][2],flush=True)
    out=BASE/'arc_boundary_neighborhood_diagnostic_v1';out.mkdir(exist_ok=False)
    write(out/'report.json',dict(faces=stats,rows=rows,graph_receipt_sha256=digest(graphs/'receipt.json'),target_sha256=digest(target),script_sha256=digest(__file__),scope='TRAIN reference-selected three support planes. Neighbor angular gap is surface-boundary evidence only, not contact or weld truth. No threshold selection, no runtime filtering, no model change.',DEV_reads=0))

if __name__=='__main__':main()
