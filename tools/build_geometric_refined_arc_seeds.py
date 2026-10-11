"""Add input-only geometric circle refinements while retaining all seed proposals."""
from pathlib import Path
import time,shutil
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_geometric_circle_refinement import refine_chain_circle
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    started=time.perf_counter();key='new_fcaf2af1abee';source=BASE/'direction_union_arc_seeds_v1'/key;graphs=BASE/'union_nearest_face_graphs_v1'/key
    sr=read(source/'receipt.json');assert digest(source/'protocol.json')==sr['protocol_sha256'];sp=read(source/'protocol.json');assert sp['source_receipt_sha256']==digest(graphs/'receipt.json');gm={r['face']:r for r in read(graphs/'receipt.json')['faces']}
    out=BASE/'geometric_refined_arc_seeds_v1'/key;out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(source_receipt_sha256=digest(source/'receipt.json'),graph_receipt_sha256=digest(graphs/'receipt.json'),module_sha256=digest('weld_pipeline/server_geometric_circle_refinement.py'),GT_reads=0,selection='All existing direction-chain seeds retained, append successful nonincreasing observed radial SSE refinements on exactly the same chain. No reference reads or source-specific radius/coordinate thresholds.',max_optimizer_evaluations=100))
    rows=[]
    for record in sr['faces']:
        assert shutil.disk_usage(BASE).free>=8*1024**3,'Insufficient disk headroom'
        cp=source/record['file'];assert digest(cp)==record['sha256'];gr=gm[record['face']];gp=graphs/gr['file'];assert digest(gp)==gr['sha256'];old=read(cp)['candidates'];candidates=list(old)
        with np.load(gp) as g:
            uv,origin,basis=g['uv'],g['origin'],g['basis']
            for index,seed in enumerate(old):
                refined=refine_chain_circle(uv,seed['chain_vertex_indices'],origin,basis,seed)
                if refined is not None:candidates.append(dict(refined,source_seed=index,proposal_source='same_chain_geometric_refinement'))
        path=out/record['file'];write(path,dict(face=record['face'],candidates=candidates));rows.append(dict(face=record['face'],original_candidates=len(old),refined_candidates=len(candidates)-len(old),candidates=len(candidates),file=path.name,sha256=digest(path)))
        if record['face']%50==0:print(record['face'],'refined',sum(r['refined_candidates'] for r in rows),flush=True)
    write(out/'receipt.json',dict(status='INPUT_ARC_CANDIDATES_FROZEN',faces=rows,protocol_sha256=digest(out/'protocol.json'),seconds=time.perf_counter()-started,GT_reads=0,no_automatic_promotion=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
