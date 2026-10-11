"""Extend frozen circle candidates only along observed annular graph support."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_graph_circle_intervals import supported_circle_intervals
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main():
    started=time.perf_counter();key='new_ac2008a03607'
    source=BASE/'local_graph_arc_candidates_v1'/key;graphs=BASE/'local_spacing_face_graphs_v1'/key
    receipt=read(source/'receipt.json');graph_receipt=read(graphs/'receipt.json')
    assert receipt['status']=='INPUT_ARC_CANDIDATES_FROZEN'
    protocol=read(source/'protocol.json');assert digest(source/'protocol.json')==receipt['protocol_sha256']
    assert protocol['source_receipt_sha256']==digest(graphs/'receipt.json')
    graph_map={r['face']:r for r in graph_receipt['faces']}
    out=BASE/'local_graph_arc_extended_candidates_v1'/key;out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(source_receipt_sha256=digest(source/'receipt.json'),graph_receipt_sha256=digest(graphs/'receipt.json'),module_sha256=digest('weld_pipeline/server_graph_circle_intervals.py'),band_mm=3.,GT_reads=0,selection='All frozen circle candidates retained; add each connected angular interval supported by whole input graph edges within radius +/-3mm. No positive gaps joined; no circle refitting; candidate geometry only.'))
    rows=[]
    for record in receipt['faces']:
        path=source/record['file'];assert digest(path)==record['sha256']
        graph_record=graph_map[record['face']];gp=graphs/graph_record['file'];assert digest(gp)==graph_record['sha256']
        original=read(path)['candidates'];candidates=list(original)
        with np.load(gp) as g:
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
    with threadpool_limits(limits=2):main()
