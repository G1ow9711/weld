"""Freeze endpoint candidates for all input arcs, without reference reads."""
from pathlib import Path
import time, shutil
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_arc_vertex_events import arc_vertex_events
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main(key='new_ac2008a03607'):
    # Finite arc excludes unrelated angular vertices, retaining observed ones.
    arc=CircularArc([0,0,0],10,[0,0,1],[1,0,0],1.)
    theta=np.array([.2,.7,1.5]);uv=10*np.column_stack((np.cos(theta),np.sin(theta)))
    q,_=arc_vertex_events(arc,uv,np.zeros(3),np.eye(3)[:,:2]);assert np.allclose(q,[0,2,7,10])
    reversed_arc=CircularArc([0,0,0],10,[0,0,1],[np.cos(1),np.sin(1),0],-1.)
    reverse,_=arc_vertex_events(reversed_arc,uv,np.zeros(3),np.eye(3)[:,:2]);assert np.allclose(reverse,10-q[::-1])
    distant=uv*2;q,_=arc_vertex_events(arc,distant,np.zeros(3),np.eye(3)[:,:2]);assert np.array_equal(q,[0,10])
    started=time.perf_counter();source=BASE/'union_nearest_arc_candidates_v1'/key;graphs=BASE/'union_nearest_face_graphs_v1'/key
    receipt=read(source/'receipt.json');assert receipt['status']=='INPUT_ARC_CANDIDATES_FROZEN'
    protocol=read(source/'protocol.json');assert digest(source/'protocol.json')==receipt['protocol_sha256'];assert digest(graphs/'receipt.json')==protocol['graph_receipt_sha256']
    graph_map={r['face']:r for r in read(graphs/'receipt.json')['faces']}
    out=BASE/'training_arc_vertex_events_v1'/key;out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(candidate_receipt_sha256=digest(source/'receipt.json'),graph_receipt_sha256=digest(graphs/'receipt.json'),module_sha256=digest('weld_pipeline/server_arc_vertex_events.py'),band_mm=3.,GT_reads=0,controls='Finite angular range, reversed carrier, remote radial points PASS',events='All graph vertices within3mm radial band projected onto finite candidate arc plus original carrier endpoints. Exact unique positions; no GT insertion. Projection is a candidate, not measured weld endpoint.'))
    rows=[]
    for record in receipt['faces']:
        assert shutil.disk_usage(BASE).free >= 8*1024**3, 'Insufficient cache disk headroom'
        cp=source/record['file'];assert digest(cp)==record['sha256'];candidates=read(cp)['candidates'];gr=graph_map[record['face']];gp=graphs/gr['file'];assert digest(gp)==gr['sha256']
        offsets=[0];positions=[];vertices=[]
        with np.load(gp) as g:
            uv=g['uv'];origin=g['origin'];basis=g['basis']
            for c in candidates:
                q,ids=arc_vertex_events(CircularArc.from_parameters(c['arc_parameters']),uv,origin,basis)
                positions.append(q);vertices.append(ids);offsets.append(offsets[-1]+len(q))
        dest=out/f"face_{record['face']}.npz";np.savez_compressed(dest,offsets=np.asarray(offsets,np.int64),positions_mm=np.concatenate(positions) if positions else np.empty(0),vertex_indices=np.concatenate(vertices) if vertices else np.empty(0,np.int64))
        rows.append(dict(face=record['face'],candidates=len(candidates),events=offsets[-1],file=dest.name,sha256=digest(dest)))
        if record['face']%50==0:print(record['face'],'events',sum(r['events'] for r in rows),flush=True)
    write(out/'receipt.json',dict(status='INPUT_ARC_EVENTS_FROZEN',faces=rows,protocol_sha256=digest(out/'protocol.json'),GT_reads=0,seconds=time.perf_counter()-started))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--key', choices=['new_ac2008a03607','new_fcaf2af1abee'], default='new_ac2008a03607');args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.key)
