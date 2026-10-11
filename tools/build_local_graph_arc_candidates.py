"""Generate all unbranched-chain circle candidates from frozen input graphs."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_local_graph_arc_candidates import prune_and_trace,fit_chain_arc
from weld_pipeline.server_direction_continuous_chains import trace_direction_continuous
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main(key='new_ac2008a03607', graph='mutual', trace='unbranched'):
    started=time.perf_counter();source=BASE/('local_spacing_face_graphs_v1' if graph=='mutual' else 'union_nearest_face_graphs_v1')/key
    receipt=read(source/'receipt.json');assert receipt['status']=='INPUT_GRAPHS_FROZEN'
    assert digest(source/'protocol.json')==receipt['protocol_sha256']
    out=BASE/(('local_graph_arc_candidates_v1' if graph=='mutual' else 'union_graph_arc_seeds_v1') if trace=='unbranched' else 'direction_'+graph+'_arc_seeds_v1')/key;out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(source_receipt_sha256=digest(source/'receipt.json'),module_sha256=digest('weld_pipeline/server_local_graph_arc_candidates.py'),GT_reads=0,trace_method=trace,trace_module_sha256=digest('weld_pipeline/server_direction_continuous_chains.py') if trace=='direction' else None,selection='Every frozen input face. All monotone nondegenerate least-squares circle fits to selected-method chains with >=5 distinct vertices; residuals retained, no label-based filtering.'))
    rows=[]
    for record in receipt['faces']:
        path=source/record['file'];assert digest(path)==record['sha256']
        with np.load(path) as graph:
            uv=graph['uv'];chains,edges=prune_and_trace(uv,graph['edges']);candidates=[]
            if trace=='direction':chains=trace_direction_continuous(uv,edges)
            for chain in chains:
                fit=fit_chain_arc(uv,chain,graph['origin'],graph['basis'])
                if fit is not None:candidates.append(dict(fit,chain_vertex_indices=chain.tolist()))
        target=out/f"face_{record['face']}.json";write(target,dict(face=record['face'],source_graph_sha256=record['sha256'],candidates=candidates))
        rows.append(dict(face=record['face'],kept_graph_edges=len(edges),chains=len(chains),candidates=len(candidates),file=target.name,sha256=digest(target)))
        if record['face']%50==0:print(record['face'],'candidates',sum(r['candidates'] for r in rows),flush=True)
    write(out/'receipt.json',dict(status='INPUT_ARC_CANDIDATES_FROZEN',faces=rows,protocol_sha256=digest(out/'protocol.json'),seconds=time.perf_counter()-started,GT_reads=0,no_automatic_promotion=True))
    print('COMPLETE',sum(r['candidates'] for r in rows),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--key',choices=['new_ac2008a03607','new_fcaf2af1abee'],default='new_ac2008a03607');parser.add_argument('--graph',choices=['mutual','union'],default='mutual');parser.add_argument('--trace',choices=['unbranched','direction'],default='unbranched');args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.key,args.graph,args.trace)
