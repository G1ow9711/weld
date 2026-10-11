"""Wait for the registered input graph producer, then freeze and score arcs."""
from pathlib import Path
import sys,time,subprocess,traceback
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'union_nearest_arc_comparison_followup'

def main():
    OUT.mkdir(exist_ok=False);key='new_ac2008a03607'
    write(OUT/'protocol.json',dict(key=key,source_circle_receipt_sha256=digest(BASE/'local_graph_arc_candidates_v1'/key/'receipt.json'),unchanged_circle_fits=True,unchanged_annulus_band_mm=3.,unchanged_original_D3=True,GT_in_candidate_generation=False,DEV_reads=0,no_automatic_promotion=True))
    ready=BASE/'union_nearest_face_graphs_v1'/key/'receipt.json';proc=Path('/proc/809582');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None
    while not ready.exists():
        if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Graph producer stopped without receipt')
        time.sleep(5)
    jobs=[('candidates','build_union_nearest_arc_candidates.py',[]),('capacity','evaluate_local_graph_arc_capacity.py',['--key',key,'--folder-name','union_nearest_arc_candidates_v1'])]
    for name,script,args in jobs:
        print('START',name,flush=True)
        with (OUT/(name+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',name,flush=True)
    folder=BASE/'union_nearest_arc_candidates_v1'/key;score=read(folder/'d3_interval_capacity.json')
    write(OUT/'receipt.json',dict(status='INPUT_GRAPH_COMPARISON_CAPACITY_COMPLETE',candidate_receipt_sha256=digest(folder/'receipt.json'),score_sha256=digest(folder/'d3_interval_capacity.json'),projected_interval_children=score['children_with_original_D3_witness'],full_candidate_geometry_children=score['children_with_full_candidate_geometry_witness'],candidate_count=score['candidate_count'],no_automatic_promotion=True,DEV_reads=0))

if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
