"""TRAIN-only trace reference support into existing graph chain topology."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.server_local_graph_arc_candidates import prune_and_trace,fit_chain_arc
from weld_pipeline.server_direction_continuous_chains import trace_direction_continuous
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    key='new_fcaf2af1abee';source=BASE/'union_nearest_face_graphs_v1'/key;receipt=read(source/'receipt.json');assert digest(source/'protocol.json')==receipt['protocol_sha256'];mapping={r['face']:r for r in receipt['faces']}
    membership=BASE/'training_arc_support_membership_fc_diagnostic_v1/report.json';mr=read(membership);target=BASE/'labels'/key/'targets.json';assert digest(target)==mr['target_sha256'];truth=read(target)
    # Faces selected by prior TRAIN diagnostic; never used as runtime candidate filter.
    selected=[0,190,268];rows=[];face_stats=[]
    for face in selected:
        path=source/mapping[face]['file'];assert digest(path)==mapping[face]['sha256']
        with np.load(path) as z:uv=z['uv'];origin=z['origin'];basis=z['basis'];edges=z['edges']
        _,kept=prune_and_trace(uv,edges);chains=trace_direction_continuous(uv,kept);degree=np.bincount(kept.ravel(),minlength=len(uv));max_chain=np.zeros(len(uv),np.int64);fitted=np.zeros(len(uv),bool);lengths=[];fit_count=0
        for chain in chains:
            np.maximum.at(max_chain,chain,len(chain));lengths.append(len(chain))
            if len(chain)>=5 and fit_chain_arc(uv,chain,origin,basis) is not None:fitted[chain]=True;fit_count+=1
        face_stats.append(dict(face=face,vertices=len(uv),edges=len(kept),chains=len(chains),chain_size_quantiles=np.quantile(lengths,[0,.5,.9,.99,1]).tolist(),fitted_chains=fit_count))
        tree=cKDTree(uv)
        for pi,parent in enumerate(truth['parents']):
            for ci,child in enumerate(parent['children']):
                if child['representation']['kind']!='arc':continue
                arc=CircularArc.from_parameters(child['representation']['arc_parameters']);points=arc.point_at(np.linspace(0,arc.length_mm,int(np.ceil(arc.length_mm))+1));distance,ids=tree.query((points-origin)@basis,workers=2)
                row=dict(face=face,parent=pi,child=ci,samples=len(ids),nearest_projected_vertex_distance_quantiles_mm=np.quantile(distance,[0,.5,1]).tolist(),degree_counts={str(int(d)):int(n) for d,n in zip(*np.unique(degree[ids],return_counts=True))},max_chain_size_at_sample_quantiles=np.quantile(max_chain[ids],[0,.5,.9,1]).tolist(),samples_in_fitted_chain=int(fitted[ids].sum()),samples_in_chain_ge5=int((max_chain[ids]>=5).sum()))
                rows.append(row)
        print('face',face,face_stats[-1],flush=True)
    out=BASE/'fc_direction_chain_topology_diagnostic_v1';out.mkdir(exist_ok=False)
    write(out/'report.json',dict(rows=rows,faces=face_stats,graph_receipt_sha256=digest(source/'receipt.json'),membership_report_sha256=digest(membership),target_sha256=digest(target),scope='TRAIN reference-guided diagnosis on three previously identified support planes. Direction-continuous chains. Nearest projected graph vertex is a diagnostic association, not runtime selection or weld proof.',DEV_reads=0,fit_calls=0))


if __name__=='__main__':main()
