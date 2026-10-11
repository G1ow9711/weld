"""Apply the frozen graph/circle construction to the second existing TRAIN case."""
from pathlib import Path
import sys,subprocess,traceback
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'second_training_arc_followup'

def main():
    key='new_fcaf2af1abee';OUT.mkdir(exist_ok=False)
    source=BASE/'local_spacing_face_graphs_v1'/key/'receipt.json';assert read(source)['status']=='INPUT_GRAPHS_FROZEN'
    write(OUT/'protocol.json',dict(key=key,role='Existing TRAIN arc family',input_graph_receipt_sha256=digest(source),same_graph_circle_and_interval_rules_as_first_TRAIN=True,no_parameter_search=True,DEV_reads=0,heldout_reads=0))
    jobs=[('circle_seeds','build_local_graph_arc_candidates.py',['--key',key]),('union_graph','cache_union_nearest_face_graphs.py',['--key',key]),('extended_candidates','build_union_nearest_arc_candidates.py',['--key',key]),('capacity','evaluate_local_graph_arc_capacity.py',['--key',key,'--folder-name','union_nearest_arc_candidates_v1'])]
    for name,script,args in jobs:
        print('START',name,flush=True)
        with (OUT/(name+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools'/script),*args],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
        print('COMPLETE',name,flush=True)
    folder=BASE/'union_nearest_arc_candidates_v1'/key
    write(OUT/'receipt.json',dict(status='SECOND_TRAIN_ARC_CAPACITY_COMPLETE',score_sha256=digest(folder/'d3_interval_capacity.json'),no_automatic_promotion=True,DEV_reads=0))

if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
