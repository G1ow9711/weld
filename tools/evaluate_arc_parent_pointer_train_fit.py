"""Original D3 evaluation of frozen conditional TRAIN pointer predictions."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.contract_geometry_d3_409 import score_source

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main(model_folder='arc_parent_boundary_pointer_v1'):
    model=BASE/model_folder;receipt=read(model/'receipt.json');assert digest(model/'conditional_train_predictions.json')==receipt['prediction_sha256']
    key='new_ac2008a03607';source=BASE/'union_nearest_arc_candidates_v1'/key;events=BASE/'training_arc_vertex_events_v1'/key
    protocol=read(model/'protocol.json');labeldir=BASE/'training_arc_parent_boundaries_v1';assert digest(labeldir/'receipt.json')==protocol['label_receipt_sha256'];labels=read(labeldir/'labels.json');assert labels['event_receipt_sha256']==digest(events/'receipt.json')
    targetpath=BASE/'labels'/key/'targets.json';assert digest(targetpath)==labels['target_sha256'];truth=native_targets(read(targetpath));banks={};rows=[];parents=[];invalid=[]
    for row in read(model/'conditional_train_predictions.json')['rows']:
        face=row['face'];index=row['candidate'];target=truth['parents'][row['parent']]
        if face not in banks:
            with np.load(events/f'face_{face}.npz') as z:banks[face]=(z['offsets'],z['positions_mm'],read(source/f'face_{face}.json')['candidates'])
        offsets,positions,candidates=banks[face];a,b=offsets[index:index+2];q=positions[a:b];path=row['selected_local_event_indices'];assert all(0<=i<len(q) for i in path);assert all(q[j]>q[i] for i,j in zip(path,path[1:]))
        roles={c['weld_type'] for c in target['children']};assert len(roles)==1;role=next(iter(roles));arc=CircularArc.from_parameters(candidates[index]['arc_parameters']);segments=[]
        for i,j in zip(path,path[1:]):
            segment=sample_arc(arc.subarc(float(q[i]),float(q[j])));segment['weld_type']=role;segments.append(segment)
        parent=dict(child_count=len(segments),is_closed=False,gaps=['SHARED']*max(0,len(segments)-1),segments=segments)
        if segments:parents.append(parent)
        elif path:invalid.append(dict(counted_false_children=1,predicted_weld_types=[0 if role=='horizontal' else 1],predicted_kinds=[2],reason='ONLY_ONE_SELECTED_BOUNDARY'))
        localtruth=dict(truth,parents=[target],original_children=len(target['children']),original_parents=1)
        score=score_source(dict(parents=[parent] if segments else [],invalid_predictions=[],total_prediction_children=len(segments)),localtruth,[False])
        rows.append(dict(row,selected_boundary_count=len(path),predicted_children=len(segments),conditional_geometry=score['categories']['overall']))
    full=score_source(dict(parents=parents,invalid_predictions=invalid,total_prediction_children=sum(p['child_count'] for p in parents)+len(invalid)),truth,[False]*len(truth['parents']))
    write(model/'conditional_train_geometry.json',dict(rows=rows,complete_conditioned_carriers=sum(r['conditional_geometry']['TP']==r['conditional_geometry']['G'] and r['conditional_geometry']['FP']==0 for r in rows),conditioned_carriers=len(rows),all_conditioned_outputs_score=full,prediction_sha256=receipt['prediction_sha256'],scope='Conditional TRAIN fit only: carrier list and weld type supplied from training labels. Pointer-selected positions/count/order unmodified. All conditioned outputs retained including duplicate FP; no runtime detector or generalization claim.',fit_calls=0,DEV_reads=0))
    print('Complete conditioned carriers',sum(r['conditional_geometry']['TP']==r['conditional_geometry']['G'] and r['conditional_geometry']['FP']==0 for r in rows),'/',len(rows),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--model-folder',choices=['arc_parent_boundary_pointer_v1','arc_parent_feedback_fixed_v1','arc_parent_feedback_model_valid_v1'],default='arc_parent_boundary_pointer_v1');args=parser.parse_args();main(args.model_folder)
