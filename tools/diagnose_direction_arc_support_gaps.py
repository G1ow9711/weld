"""TRAIN diagnosis: distinguish circle error from observed finite support gaps."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.server_circle_capacity_prefilter import possible_circle_indices
from weld_pipeline.server_graph_circle_intervals import supported_circle_intervals
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    key='new_fcaf2af1abee';source=BASE/'direction_union_arc_seeds_v1'/key;graphs=BASE/'union_nearest_face_graphs_v1'/key;sr=read(source/'receipt.json');assert digest(source/'protocol.json')==sr['protocol_sha256'];assert read(source/'protocol.json')['source_receipt_sha256']==digest(graphs/'receipt.json');gm={r['face']:r for r in read(graphs/'receipt.json')['faces']};seeds=[]
    for f in sr['faces']:
        path=source/f['file'];assert digest(path)==f['sha256']
        for i,c in enumerate(read(path)['candidates']):seeds.append((f['face'],i,CircularArc.from_parameters(c['arc_parameters'])))
    centers=np.array([s[2].center_mm for s in seeds]);normals=np.array([s[2].plane_normal for s in seeds]);radii=np.array([s[2].radius_mm for s in seeds]);target=BASE/'labels'/key/'targets.json';rows=[];banks={}
    for pi,parent in enumerate(read(target)['parents']):
        for ci,child in enumerate(parent['children']):
            if child['representation']['kind']!='arc':continue
            truth=CircularArc.from_parameters(child['representation']['arc_parameters']);points=truth.point_at(np.linspace(0,truth.length_mm,int(np.ceil(truth.length_mm))+1));eligible=possible_circle_indices(centers,normals,radii,points);best=None
            for idx in eligible:
                delta=points-centers[idx];height=delta@normals[idx];radial=np.linalg.norm(delta-height[:,None]*normals[idx],axis=1);error=float(np.hypot(height,radial-radii[idx]).max())
                if error<=3 and (best is None or error<best[0]):best=(error,idx)
            if best is None:rows.append(dict(parent=pi,child=ci,circle_compatible=False));continue
            error,idx=best;face,index,arc=seeds[idx]
            if face not in banks:
                path=graphs/gm[face]['file'];assert digest(path)==gm[face]['sha256']
                with np.load(path) as g:banks[face]={n:g[n] for n in ['uv','edges','origin','basis']}
                banks[face]['tree']=cKDTree(banks[face]['uv'])
            g=banks[face];center=(arc.center_mm-g['origin'])@g['basis'];delta=(points-g['origin'])@g['basis']-center;theta=np.mod(np.arctan2(delta[:,1],delta[:,0]),2*np.pi);candidate_points=center+arc.radius_mm*np.column_stack((np.cos(theta),np.sin(theta)));intervals=supported_circle_intervals(g['uv'],g['edges'],center,arc.radius_mm);covered=np.zeros(len(theta),bool)
            for lo,hi in intervals:covered|=((theta>=lo)&(theta<=hi))|((theta+2*np.pi>=lo)&(theta+2*np.pi<=hi))
            distance=g['tree'].query(candidate_points,workers=2)[0];missing=~covered
            rows.append(dict(parent=pi,child=ci,circle_compatible=True,selected_circle_face=face,selected_circle_candidate=index,infinite_circle_error_mm=error,samples=len(points),uncovered_samples=int(missing.sum()),uncovered_with_vertex_within3mm=int((missing&(distance<=3)).sum()),nearest_vertex_distance_quantiles_mm=np.quantile(distance,[0,.5,.95,1]).tolist(),uncovered_nearest_vertex_quantiles_mm=np.quantile(distance[missing],[0,.5,1]).tolist() if missing.any() else [],scope='Reference picks best compatible infinite circle solely for diagnosis. No reference-selected carrier is emitted by inference.'))
            print(pi,'missing',int(missing.sum()),'near input vertex',int((missing&(distance<=3)).sum()),flush=True)
    out=BASE/'direction_arc_support_gap_diagnostic_v1';out.mkdir(exist_ok=False);write(out/'report.json',dict(rows=rows,candidate_receipt_sha256=digest(source/'receipt.json'),graph_receipt_sha256=digest(graphs/'receipt.json'),target_sha256=digest(target),scope='TRAIN diagnostic only; <=1mm reference samples. Near vertices do not prove weld/contact or connected support. No model/candidate changes.',DEV_reads=0))


if __name__=='__main__':main()
