from pathlib import Path
import numpy as np
from weld_pipeline.server_graph_circle_intervals import supported_circle_intervals
from weld_pipeline.sjg_local_runtime_904 import write,digest

def main():
    angles=np.array([-.2,-.1,0.,.1,.2,.5,.6,.7]);p=61*np.column_stack((np.cos(angles),np.sin(angles)))
    edges=np.array([[0,1],[1,2],[2,3],[3,4],[5,6],[6,7]])
    intervals=supported_circle_intervals(p,edges,np.zeros(2),61)
    assert len(intervals)==2,intervals
    assert np.allclose(sorted(b-a for a,b in intervals),[.2,.4])
    reverse=supported_circle_intervals(p,edges[:,::-1],np.zeros(2),61)
    assert np.allclose(sorted(intervals),sorted(reverse))
    # Endpoint support alone must not admit a chord crossing the inner disk.
    assert supported_circle_intervals(np.array([[61.,0],[-61.,0]]),np.array([[0,1]]),np.zeros(2),61)==[]
    assert supported_circle_intervals(p+1000,edges,np.zeros(2),61)==[]
    out=Path('server_rebuild/expansion_v1/graph_circle_interval_controls_v1');out.mkdir(exist_ok=False)
    write(out/'receipt.json',dict(status='PASS',checks=['wrap_interval','positive_gap_preserved','edge_reversal','interior_chord_rejected','remote_edges_rejected'],module_sha256=digest('weld_pipeline/server_graph_circle_intervals.py'),scope='Analytic interval union only, not contact or extraction.'))
    print('PASS')

if __name__=='__main__':main()
