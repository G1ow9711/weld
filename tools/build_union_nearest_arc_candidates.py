"""Extend frozen circle candidates only along observed annular graph support."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_graph_circle_intervals import supported_circle_intervals
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main(key='new_ac2008a03607', seeds='mutual'):
    started=time.perf_counter()
    source=BASE/('local_graph_arc_candidates_v1' if seeds=='mutual' else 'direction_union_arc_seeds_v1')/key;graphs=BASE/'union_nearest_face_graphs_v1'/key;original_graphs=BASE/('local_spacing_face_graphs_v1' if seeds=='mutual' else 'union_nearest_face_graphs_v1')/key
    receipt=read(source/'receipt.json');graph_receipt=read(graphs/'receipt.json')
    assert receipt['status']=='INPUT_ARC_CANDIDATES_FROZEN'
    protocol=read(source/'protocol.json');assert digest(source/'protocol.json')==receipt['protocol_sha256']
    assert protocol['source_receipt_sha256']==digest(original_graphs/'receipt.json')
    original_graph_receipt=read(original_graphs/'receipt.json');original_graph_map={r['face']:r for r in original_graph_receipt['faces']}
    assert graph_receipt['protocol_sha256']==digest(graphs/'protocol.json')
    old_graph_protocol=read(original_graphs/'protocol.json');new_graph_protocol=read(graphs/'protocol.json')
    for name in ('input_sha256','source_freeze_sha256'):assert old_graph_protocol[name]==new_graph_protocol[name]
    graph_map={r['face']:r for r in graph_receipt['faces']}
    out=BASE/('union_nearest_arc_candidates_v1' if seeds=='mutual' else 'direction_union_arc_candidates_v1')/key;out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(source_receipt_sha256=digest(source/'receipt.json'),graph_receipt_sha256=digest(graphs/'receipt.json'),module_sha256=digest('weld_pipeline/server_graph_circle_intervals.py'),original_graph_receipt_sha256=digest(original_graphs/'receipt.json'),changed_factor=('Mutual local graph plus symmetric nearest-neighbor union; same original circles and annulus interval rule' if seeds=='mutual' else 'Direction-continuous circle seeds, same union graph and annular interval support rule'),band_mm=3.,GT_reads=0,selection='All frozen circle candidates retained; add each connected angular interval supported by whole input graph edges within radius +/-3mm. No positive gaps joined; no circle refitting; candidate geometry only.'))
    rows=[]
    for record in receipt['faces']:
        path=source/record['file'];assert digest(path)==record['sha256']
        graph_record=graph_map[record['face']];gp=graphs/graph_record['file'];assert digest(gp)==graph_record['sha256']
        original=read(path)['candidates'];candidates=list(original)
        original_record=original_graph_map[record['face']];original_path=original_graphs/original_record['file'];assert digest(original_path)==original_record['sha256']
        with np.load(gp) as g, np.load(original_path) as old_graph:
            for name in ('uv','origin','basis','spacing'):assert np.array_equal(g[name],old_graph[name]),name
            n=len(g['uv']);old_codes=old_graph['edges'][:,0]*n+old_graph['edges'][:,1];new_codes=g['edges'][:,0]*n+g['edges'][:,1]
            assert np.isin(old_codes,new_codes).all(), 'Original graph edges lost'
            uv=g['uv'];edges=g['edges'];origin=g['origin'];basis=g['basis'];normal=np.cross(basis[:,0],basis[:,1])
            for index,candidate in enumerate(original):
                circle=CircularArc.from_parameters(candidate['arc_parameters']);center=(circle.center_mm-origin)@basis
                for lo,hi in supported_circle_intervals(uv,edges,center,circle.radius_mm):
                    sweep=hi-lo;closed=lo==0. and hi==2*np.pi
                    arc=CircularArc(circle.center_mm,circle.radius_mm,normal,basis@np.array([np.cos(lo),np.sin(lo)]),sweep,closed)
                    candidates.append(dict(arc_parameters=arc.to_parameters(),source_circle_candidate=index,angular_support_interval_rad=[lo,hi],proposal_source='observed_annular_graph_interval'))
        target=out/record['file'];write(target,dict(face=record['face'],candidates=candidates))
        rows.append(dict(face=record['face'],original_candidates=len(original),candidates=len(candidates),file=target.name,sha256=digest(target)))
        if record['face']%50==0:print(record['face'],'candidates',sum(r['candidates'] for r in rows),flush=True)
    write(out/'receipt.json',dict(status='INPUT_ARC_CANDIDATES_FROZEN',faces=rows,protocol_sha256=digest(out/'protocol.json'),GT_reads=0,seconds=time.perf_counter()-started,no_automatic_promotion=True))
    print('COMPLETE',sum(r['candidates'] for r in rows),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--key',choices=['new_ac2008a03607','new_fcaf2af1abee'],default='new_ac2008a03607');parser.add_argument('--seeds',choices=['mutual','direction'],default='mutual');args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.key,args.seeds)
