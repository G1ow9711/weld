"""Reference-sampled distance to frozen input graphs, not weld extraction score."""
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from tools.diagnose_training_arc_boundary_support import segment_distances

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main():
    key='new_ac2008a03607';source=BASE/'local_spacing_face_graphs_v1'/key
    receipt=read(source/'receipt.json');assert receipt['status']=='INPUT_GRAPHS_FROZEN'
    assert digest(source/'protocol.json')==receipt['protocol_sha256']
    for face in receipt['faces']:assert digest(source/face['file'])==face['sha256']
    # All candidate graphs are complete and validated before any reference read.
    target=BASE/'labels'/key/'targets.json';truth=read(target);arcs=[]
    for pi,parent in enumerate(truth['parents']):
        for ci,child in enumerate(parent['children']):
            if child['representation']['kind']!='arc':continue
            arc=CircularArc.from_parameters(child['representation']['arc_parameters'])
            points=arc.point_at(np.linspace(0,arc.length_mm,int(np.ceil(arc.length_mm))+1))
            arcs.append(dict(parent=pi,child=ci,points=points,distance=np.full(len(points),np.inf)))
    for fi,face in enumerate(receipt['faces']):
        with np.load(source/face['file']) as graph:
            if not len(graph['edges']):continue
            segments=graph['origin']+graph['uv'][graph['edges']]@graph['basis'].T
        for arc in arcs:
            arc['distance']=np.minimum(arc['distance'],segment_distances(arc['points'],segments))
        if fi%50==0:print('faces',fi,flush=True)
    rows=[dict(parent=a['parent'],child=a['child'],samples=len(a['points']),distance_quantiles_mm=np.quantile(a['distance'],[0,.5,.95,1]).tolist(),samples_le3=int((a['distance']<=3).sum())) for a in arcs]
    out=BASE/'local_spacing_arc_support_evaluation_v1';out.mkdir(exist_ok=False)
    write(out/'report.json',dict(rows=rows,input_graph_receipt_sha256=digest(source/'receipt.json'),target_sha256=digest(target),scope='TRAIN reference sampling <=1mm. Union may span disconnected or interior graph edges; no analytic arc fit, D3 weld prediction, material boundary, or contact proof.',fit_calls=0,DEV_reads=0,heldout_reads=0))
    print(rows,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
