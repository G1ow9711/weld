"""TRAIN boundary-set labels and exact-sharing path witnesses on frozen events."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.contract_geometry_d3_409 import child_error,score_source
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main():
    key='new_ac2008a03607';events=BASE/'training_arc_vertex_events_v1'/key;source=BASE/'union_nearest_arc_candidates_v1'/key
    er=read(events/'receipt.json');assert digest(events/'protocol.json')==er['protocol_sha256']
    assert read(events/'protocol.json')['candidate_receipt_sha256']==digest(source/'receipt.json')
    for f in er['faces']:assert digest(events/f['file'])==f['sha256']
    cr=read(source/'receipt.json')
    for f in cr['faces']:assert digest(source/f['file'])==f['sha256']
    reportpath=BASE/'training_arc_event_capacity_v1/report.json';previous=read(reportpath);assert previous['event_receipt_sha256']==digest(events/'receipt.json')
    targetpath=BASE/'labels'/key/'targets.json';truth=native_targets(read(targetpath));by_parent={};banks={};rows=[];decoded=[]
    for row in previous['rows']:by_parent.setdefault(row['parent'],{})[row['child']]=row
    for pi,child_rows in by_parent.items():
        parent=truth['parents'][pi];children=parent['children'];assert not parent['is_closed'] and all(g=='SHARED' for g in parent['gaps'])
        refs=[reference_segment(c) for c in children];common=None
        for ci in range(len(children)):
            compatible={(w['face'],w['candidate']) for w in child_rows[ci]['witnesses']};common=compatible if common is None else common&compatible
        labelled=[];witness=None
        for face,index in sorted(common or []):
            if face not in banks:
                with np.load(events/f'face_{face}.npz') as z:banks[face]=(z['offsets'],z['positions_mm'],read(source/f'face_{face}.json')['candidates'])
            offsets,positions,candidates=banks[face];start,stop=offsets[index:index+2];q=positions[start:stop];arc=CircularArc.from_parameters(candidates[index]['arc_parameters']);points=arc.point_at(q);masks=[]
            for boundary in range(len(children)+1):
                targets=[]
                if boundary>0:targets.append(refs[boundary-1]['points'][-1])
                if boundary<len(children):targets.append(refs[boundary]['points'][0])
                distance=np.linalg.norm(points[:,None]-np.asarray(targets)[None],axis=2).max(1);ids=np.flatnonzero(distance<=3);masks.append(ids[np.argsort(distance[ids],kind='stable')])
            record=dict(face=face,candidate=index,boundary_event_indices=[(m+start).tolist() for m in masks],missing_boundaries=[len(m)==0 for m in masks],paths=[])
            for sign in (1,-1):
                reachable={int(i):[int(i)] for i in masks[0]}
                for ci,child in enumerate(children):
                    following={}
                    for end in masks[ci+1]:
                        for begin,path in reachable.items():
                            if sign*(q[end]-q[begin])<=0:continue
                            lo,hi=sorted((float(q[begin]),float(q[end])));segment=sample_arc(arc.subarc(lo,hi));segment['weld_type']=child['weld_type']
                            if sign<0:segment=dict(segment,**reverse_segment(segment))
                            if child_error(segment,child,refs[ci],False,False) is not None:following[int(end)]=path+[int(end)];break
                    reachable=following
                    if not reachable:break
                if reachable:
                    path=next(iter(reachable.values()));record['paths'].append(dict(direction_sign=sign,event_indices=(np.asarray(path)+start).tolist()))
                    if witness is None:
                        segments=[]
                        for ci,(begin,end) in enumerate(zip(path,path[1:])):
                            lo,hi=sorted((float(q[begin]),float(q[end])));segment=sample_arc(arc.subarc(lo,hi));segment['weld_type']=children[ci]['weld_type']
                            if sign<0:segment=dict(segment,**reverse_segment(segment))
                            segments.append(segment)
                        witness=dict(face=face,candidate=index,event_indices=(np.asarray(path)+start).tolist(),direction_sign=sign)
                        decoded.append(dict(child_count=len(children),is_closed=False,gaps=['SHARED']*(len(children)-1),segments=segments))
            labelled.append(record)
        rows.append(dict(parent=pi,child_count=len(children),common_carriers=len(common or []),labels=labelled,witness=witness));print(pi,'carriers',len(common or []),'joint',witness is not None,flush=True)
    output=dict(parents=decoded,invalid_predictions=[],total_prediction_children=sum(p['child_count'] for p in decoded))
    score=score_source(output,truth,[False]*len(truth['parents']))
    out=BASE/'training_arc_parent_boundaries_v1';out.mkdir(exist_ok=False)
    write(out/'labels.json',dict(rows=rows,event_receipt_sha256=digest(events/'receipt.json'),target_sha256=digest(targetpath),source_capacity_sha256=digest(reportpath),scope='TRAIN supervision only: carrier selection and endpoint label sets use references after input freeze. Exact shared event indices and one monotone carrier direction throughout parent; no GT coordinates inserted. No runtime predictor trained.',DEV_reads=0))
    write(out/'witness_d3_score.json',score)
    write(out/'receipt.json',dict(status='TRAIN_PARENT_BOUNDARY_LABELS_READY',parents_with_joint_witness=sum(r['witness'] is not None for r in rows),arc_parents=len(rows),labels_sha256=digest(out/'labels.json'),witness_score_sha256=digest(out/'witness_d3_score.json'),geometry_only_reference_selected=True,fit_calls=0,DEV_reads=0))

if __name__=='__main__':main()
