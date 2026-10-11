"""Add input-only multiscale chain-window circle fits, retaining all old seeds."""
from pathlib import Path
import time,shutil
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_local_graph_arc_candidates import fit_chain_arc
from weld_pipeline.server_multiscale_chain_windows import chain_windows
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    started=time.perf_counter();key='new_fcaf2af1abee';source=BASE/'direction_union_arc_seeds_v1'/key;graphs=BASE/'union_nearest_face_graphs_v1'/key
    sr=read(source/'receipt.json');assert digest(source/'protocol.json')==sr['protocol_sha256'];sp=read(source/'protocol.json');assert sp['source_receipt_sha256']==digest(graphs/'receipt.json');gm={r['face']:r for r in read(graphs/'receipt.json')['faces']}
    out=BASE/'multiscale_arc_seeds_v1'/key;out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(source_receipt_sha256=digest(source/'receipt.json'),graph_receipt_sha256=digest(graphs/'receipt.json'),module_sha256=digest('weld_pipeline/server_multiscale_chain_windows.py'),fit_module_sha256=digest('weld_pipeline/server_local_graph_arc_candidates.py'),GT_reads=0,selection='All existing direction-chain seeds retained, append original circle fits to dyadic contiguous subpaths of those input chains; open reversal-symmetric starts and all cyclic starts for closed chains. No reference-based cut points. This does not examine original rejected chains.',minimum_window_edges=4))
    rows=[]
    for record in sr['faces']:
        assert shutil.disk_usage(BASE).free>=8*1024**3,'Insufficient disk headroom'
        cp=source/record['file'];assert digest(cp)==record['sha256'];gr=gm[record['face']];gp=graphs/gr['file'];assert digest(gp)==gr['sha256'];old=read(cp)['candidates'];candidates=list(old)
        with np.load(gp) as g:
            uv,origin,basis=g['uv'],g['origin'],g['basis']
            seen={min(tuple(seed['chain_vertex_indices']),tuple(seed['chain_vertex_indices'][::-1])) for seed in old}
            for index,seed in enumerate(old):
                for window in chain_windows(seed['chain_vertex_indices']):
                    identity=min(tuple(window),tuple(window[::-1]))
                    if identity in seen:continue
                    seen.add(identity)
                    fitted=fit_chain_arc(uv,window,origin,basis)
                    if fitted is not None:candidates.append(dict(fitted,chain_vertex_indices=window.tolist(),source_seed=index,proposal_source='observed_dyadic_chain_window'))
        path=out/record['file'];write(path,dict(face=record['face'],candidates=candidates));rows.append(dict(face=record['face'],original_candidates=len(old),window_candidates=len(candidates)-len(old),candidates=len(candidates),file=path.name,sha256=digest(path)))
        if record['face']%50==0:print(record['face'],'windows',sum(r['window_candidates'] for r in rows),flush=True)
    write(out/'receipt.json',dict(status='INPUT_ARC_CANDIDATES_FROZEN',faces=rows,protocol_sha256=digest(out/'protocol.json'),seconds=time.perf_counter()-started,GT_reads=0,no_automatic_promotion=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
