"""Input-only bounded joint-circle proposal experiment; no identity assertion."""
from pathlib import Path
import time,shutil
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from weld_pipeline.server_fragment_circle_fit import fit_fragment_circle
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def carrier(uv,chain,origin,basis,center,radius):
    chain=np.asarray(chain,int);closed=chain[0]==chain[-1]
    relative=uv[chain]-center;angles=np.unwrap(np.arctan2(relative[:,1],relative[:,0]));sweep=float(angles[-1]-angles[0])
    if abs(sweep)<1e-10 or np.any(np.diff(angles)*np.sign(sweep)<=0):return None
    if closed:
        if not np.isclose(abs(sweep),2*np.pi,atol=1e-10,rtol=0):return None
        sweep=float(np.sign(sweep)*2*np.pi)
    elif abs(sweep)>=2*np.pi:return None
    if np.linalg.norm(relative[0])<=0:return None
    return CircularArc(origin+basis@center,radius,np.cross(basis[:,0],basis[:,1]),basis@(relative[0]/np.linalg.norm(relative[0])),sweep,closed)


def main():
    started=time.perf_counter();key='new_fcaf2af1abee';source=BASE/'direction_union_arc_seeds_v1'/key;graphs=BASE/'union_nearest_face_graphs_v1'/key
    sr=read(source/'receipt.json');assert digest(source/'protocol.json')==sr['protocol_sha256'];assert read(source/'protocol.json')['source_receipt_sha256']==digest(graphs/'receipt.json');gm={r['face']:r for r in read(graphs/'receipt.json')['faces']}
    out=BASE/'joint_fragment_arc_seeds_v1'/key;out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(source_receipt_sha256=digest(source/'receipt.json'),graph_receipt_sha256=digest(graphs/'receipt.json'),module_sha256=digest('weld_pipeline/server_fragment_circle_fit.py'),builder_sha256=digest(__file__),GT_reads=0,selection='Within each input face, reciprocal unique nearest centroids of existing chain seeds; disjoint pairs only, one deterministic joint fit per pair. Every projected finite carrier retains its original chain, never bridges fragments. Joint residual max <=3mm per fragment required. Same face does NOT establish same contour/contact identity; proposals remain unclassified.',budget='At most floor(N/2) pair fits, at most N projected carriers; originals preserved in separate source folder, not concatenated here.',max_optimizer_evaluations=100))
    rows=[]
    for rec in sr['faces']:
        assert shutil.disk_usage(BASE).free>=8*1024**3,'Insufficient disk headroom'
        cp=source/rec['file'];assert digest(cp)==rec['sha256'];gp=graphs/gm[rec['face']]['file'];assert digest(gp)==gm[rec['face']]['sha256'];old=read(cp)['candidates'];candidates=[];pairs=[];fits=0
        with np.load(gp) as g:
            uv,origin,basis=g['uv'],g['origin'],g['basis']
            chains=[np.asarray(c['chain_vertex_indices'],int) for c in old]
            if len(chains)>=2:
                centroids=np.array([uv[c[:-1] if c[0]==c[-1] else c].mean(0) for c in chains]);tree=cKDTree(centroids);distance,index=tree.query(centroids,k=min(3,len(chains)))
                neighbors=np.full(len(chains),-1,int)
                for i in range(len(chains)):
                    # Coincident centroids or ties are ambiguous; don't choose by index.
                    if distance[i,1]<=1e-12:continue
                    if len(chains)>2 and np.isclose(distance[i,1],distance[i,2],rtol=1e-12,atol=1e-12):continue
                    neighbors[i]=index[i,1]
                pairs=[(i,int(j)) for i,j in enumerate(neighbors) if j>i and neighbors[j]==i]
                for first,second in pairs:
                    if np.intersect1d(chains[first],chains[second]).size:continue
                    fragments=[uv[c[:-1] if c[0]==c[-1] else c] for c in [chains[first],chains[second]]]
                    fits+=1;fit=fit_fragment_circle(fragments)
                    if fit is None or max(r['max_radial_error_mm'] for r in fit['fragments'])>3:continue
                    center=np.asarray(fit['center']);radius=fit['radius']
                    for position,seed_index in enumerate([first,second]):
                        arc=carrier(uv,chains[seed_index],origin,basis,center,radius)
                        if arc is None:continue
                        candidates.append(dict(arc_parameters=arc.to_parameters(),chain_vertex_indices=chains[seed_index].tolist(),joint_source_seeds=[first,second],source_seed=seed_index,proposal_source='reciprocal_nearest_fragment_joint_circle',joint_fragment_residuals=fit['fragments'],support_vertices=len(fragments[position]),radial_error_max_mm=fit['fragments'][position]['max_radial_error_mm'],radial_error_rms_mm=fit['fragments'][position]['rms_radial_error_mm'],observed_chain_length_mm=old[seed_index]['observed_chain_length_mm'],physical_contour_identity='unverified_candidate',connects_fragment_gap=False))
        assert len(candidates)<=len(old)
        path=out/rec['file'];write(path,dict(face=rec['face'],candidates=candidates));rows.append(dict(face=rec['face'],source_candidates=len(old),reciprocal_pairs=len(pairs),fit_calls=fits,candidates=len(candidates),file=path.name,sha256=digest(path)))
        if rec['face']%50==0:print(rec['face'],'joint candidates',sum(r['candidates'] for r in rows),flush=True)
    write(out/'receipt.json',dict(status='INPUT_ARC_CANDIDATES_FROZEN',faces=rows,protocol_sha256=digest(out/'protocol.json'),seconds=time.perf_counter()-started,GT_reads=0,no_automatic_promotion=True))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
