"""Freeze input-only parent descriptors, then score TRAIN reference compatibility."""
from pathlib import Path
import time
import numpy as np
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc,CurveGeometryError
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
from weld_pipeline.contract_geometry_d3_409 import variants,child_error
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def describe(x,q,path,arc):
    lo,hi=path[0],path[-1];patch=x[lo:hi+1];ends=x[[lo,hi]];lengths=np.diff(q[path]);margin=sorted([q[lo]/arc.length_mm,(arc.length_mm-q[hi])/arc.length_mm]);scalars=[np.log1p(max(1,len(path)-1)),np.log1p(q[hi]-q[lo]),np.log1p(arc.length_mm),np.log1p(arc.radius_mm),*margin,np.log1p(lengths.min() if len(lengths) else 0),np.log1p(lengths.max() if len(lengths) else 0)]
    value=np.r_[patch.mean(0,dtype=np.float64),patch.min(0),patch.max(0),ends.mean(0),abs(ends[1]-ends[0]),scalars].astype(np.float32);assert value.shape==(423,) and np.isfinite(value).all();return value


def output_record(arc,q,path):
    if len(path)==1:return dict(invalid=dict(counted_false_children=1,predicted_weld_types=[0],predicted_kinds=[2],reason='ONE_BOUNDARY'))
    segments=[]
    try:
        for i,j in zip(path,path[1:]):
            s=sample_arc(arc.subarc(float(q[i]),float(q[j])));s['weld_type']='horizontal';reverse_segment(s);segments.append(s)
    except CurveGeometryError as error:return dict(invalid=dict(counted_false_children=len(path)-1,predicted_weld_types=[0]*(len(path)-1),predicted_kinds=[2]*(len(path)-1),reason=str(error)))
    return dict(parent=dict(child_count=len(segments),is_closed=False,gaps=['SHARED']*(len(segments)-1),segments=segments))


def main():
    started=time.perf_counter();out=BASE/'joint_arc_output_quality_data_v1';out.mkdir(exist_ok=False);records=[];features=[];bindings={};family_keys={'first':'new_ac2008a03607','second':'new_fcaf2af1abee'}
    # All descriptor/output construction happens before opening target files.
    for family,key in family_keys.items():
        prediction=BASE/('joint_arc_pointer_'+family+'_all_candidates_v1');pr=read(prediction/'receipt.json');bindings[family]=dict(prediction_receipt_sha256=digest(prediction/'receipt.json'))
        if family=='first':
            source=BASE/'union_nearest_arc_candidates_v1'/key;sr=read(source/'receipt.json');sm={r['face']:r for r in sr['faces']};ef=BASE/'training_arc_event_features_v1'/key;er=read(ef/'receipt.json');fm={r['face']:r for r in er['faces']};ev=BASE/'training_arc_vertex_events_v1'/key
            for face in pr['faces']:
                pid=face['face'];assert digest(prediction/face['file'])==face['sha256'];assert digest(source/sm[pid]['file'])==sm[pid]['sha256'];assert digest(ef/fm[pid]['file'])==fm[pid]['sha256'];eventpath=ev/f'face_{pid}.npz';assert digest(eventpath)==fm[pid]['event_file_sha256'];arcs=read(source/sm[pid]['file'])['candidates'];x=np.load(ef/fm[pid]['file'],mmap_mode='r')
                with np.load(eventpath) as z:offsets=z['offsets'];positions=z['positions_mm'];ids=z['vertex_indices']
                for row in read(prediction/face['file'])['rows']:
                    path=row['selected_local_event_indices']
                    if not path:continue
                    index=row['candidate'];a,b=offsets[index:index+2];q=positions[a:b];arc=CircularArc.from_parameters(arcs[index]['arc_parameters']);extra=np.column_stack((np.zeros((b-a,2)),ids[a:b]>=0,np.ones(b-a),np.log1p(q),np.log1p(q[-1]-q),np.zeros(b-a)));values=np.column_stack((x[a:b],extra));features.append(describe(values,q,path,arc));records.append(dict(family=family,face=pid,candidate=index,**output_record(arc,q,path)))
            bindings[family].update(feature_receipt_sha256=digest(ef/'receipt.json'),event_receipt_sha256=digest(ev/'receipt.json'),candidate_receipt_sha256=digest(source/'receipt.json'))
        else:
            source=BASE/'curved_upright_probe_v3'/key;assert digest(source/'profiles.json')==read(source/'receipt.json')['profiles_sha256'];profiles=read(source/'profiles.json');group=BASE/'curved_probe_event_groups_v1'/key;gr=read(group/'receipt.json');assert digest(group/'mean.npy')==gr['files']['mean.npy'];assert digest(group/'events.npz')==gr['files']['events.npz'];assert digest(prediction/'predictions.json')==pr['predictions_sha256'];x=np.load(group/'mean.npy',mmap_mode='r')
            with np.load(group/'events.npz') as z:offsets=z['offsets'];positions=z['positions_mm']
            for row in read(prediction/'predictions.json')['rows']:
                index=row['profile'];a,b=offsets[index:index+2];q=positions[a:b];path=row['selected_local_event_indices'];arc=CircularArc.from_parameters(profiles[index]['representation']['arc_parameters'])
                paths=[[i] for i in path] if len(path)==1 else [[i,j] for i,j in zip(path,path[1:])]
                for segment_index,part in enumerate(paths):features.append(describe(x[a:b],q,part,arc));records.append(dict(family=family,profile=index,interval_index=segment_index,**output_record(arc,q,part)))
            bindings[family].update(group_receipt_sha256=digest(group/'receipt.json'),candidate_receipt_sha256=digest(source/'receipt.json'))
    np.save(out/'features.npy',np.stack(features));write(out/'outputs.json',dict(rows=records));write(out/'input_receipt.json',dict(feature_sha256=digest(out/'features.npy'),outputs_sha256=digest(out/'outputs.json'),bindings=bindings,dimensions=423,rows=len(records),scope='Input-only actual-output geometry and event descriptor pooling. No family/candidate IDs in model features; fixed-H and branch parent conventions inherited. Frozen before target reads.',GT_reads=0))
    target_bank={};target_hashes={}
    for family,key in family_keys.items():
        path=BASE/'labels'/key/'targets.json';truth=native_targets(read(path));target_hashes[family]=digest(path);target_bank[family]=[(p,[reference_segment(c) for c in p['children']]) for p in truth['parents']]
    labels=[]
    for index,r in enumerate(records):
        if 'invalid' in r:labels.append(dict(row=index,compatible_children=0,children=r['invalid']['counted_false_children'],compatible_parents=[],quality=0.,kind='numerically_invalid_export'));continue
        parent=r['parent'];best=0;compatible=[]
        for pi,(truth,refs) in enumerate(target_bank[r['family']]):
            if parent['child_count']!=truth['original_child_count'] or parent['is_closed']!=truth['is_closed']:continue
            for view in variants(parent,True):
                if view['gaps']!=truth['gaps']:continue
                count=sum(child_error(s,truth['children'][ci],refs[ci],truth['is_closed'] and len(refs)==1,False) is not None for ci,(_,s) in enumerate(view['segments']));best=max(best,count)
                if count==parent['child_count']:compatible.append(pi);break
        labels.append(dict(row=index,compatible_children=best,children=parent['child_count'],compatible_parents=compatible,quality=best/parent['child_count'],kind='reference_geometry_quality_not_physical_presence'))
    write(out/'labels.json',dict(rows=labels,target_sha256=target_hashes,scope='TRAIN-only output geometric compatibility with provided references. Zero quality means this export fails to match reference geometry, NOT proof no physical weld/contact exists. Duplicates remain positive if geometrically compatible; set-level deduplication is separate. Shared upstream was trained on both families.'));write(out/'receipt.json',dict(status='OUTPUT_QUALITY_DATA_FROZEN',input_receipt_sha256=digest(out/'input_receipt.json'),labels_sha256=digest(out/'labels.json'),rows=len(records),seconds=time.perf_counter()-started,DEV_reads=0));print({f:dict(rows=sum(r['family']==f for r in records),positive_rows=sum(r['family']==f and l['quality']>0 for r,l in zip(records,labels))) for f in family_keys},flush=True)

if __name__=='__main__':main()
