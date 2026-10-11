"""Align quality units to children, preserving frozen parent outputs for scoring."""
from pathlib import Path
import numpy as np
from tools.prepare_joint_arc_output_quality import describe
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    source=BASE/'joint_arc_output_quality_data_v1';sr=read(source/'receipt.json');assert digest(source/'input_receipt.json')==sr['input_receipt_sha256'];inp=read(source/'input_receipt.json');assert digest(source/'outputs.json')==inp['outputs_sha256'];assert digest(source/'features.npy')==inp['feature_sha256'];parents=read(source/'outputs.json')['rows'];first_prediction=BASE/'joint_arc_pointer_first_all_candidates_v1';second_prediction=BASE/'joint_arc_pointer_second_all_candidates_v1';assert digest(first_prediction/'receipt.json')==inp['bindings']['first']['prediction_receipt_sha256'];assert digest(second_prediction/'receipt.json')==inp['bindings']['second']['prediction_receipt_sha256'];pm={r['face']:r for r in read(first_prediction/'receipt.json')['faces']};assert digest(second_prediction/'predictions.json')==read(second_prediction/'receipt.json')['predictions_sha256'];second_paths=read(second_prediction/'predictions.json')['rows'];oldx=np.load(source/'features.npy');key='new_ac2008a03607';features=BASE/'training_arc_event_features_v1'/key;fr=read(features/'receipt.json');fm={r['face']:r for r in fr['faces']};events=BASE/'training_arc_vertex_events_v1'/key;arcs=BASE/'union_nearest_arc_candidates_v1'/key;am={r['face']:r for r in read(arcs/'receipt.json')['faces']};banks={};rows=[];values=[]
    key2='new_fcaf2af1abee';group=BASE/'curved_probe_event_groups_v1'/key2;gr=read(group/'receipt.json');assert digest(group/'mean.npy')==gr['files']['mean.npy'];assert digest(group/'events.npz')==gr['files']['events.npz'];second_x=np.load(group/'mean.npy',mmap_mode='r');profilepath=BASE/'curved_upright_probe_v3'/key2/'profiles.json';assert digest(profilepath)==read(profilepath.parent/'receipt.json')['profiles_sha256'];profiles=read(profilepath)
    with np.load(group/'events.npz') as z:second_q=z['positions_mm'];second_offsets=z['offsets']
    for parent_row,parent in enumerate(parents):
        if 'invalid' in parent:
            value=oldx[parent_row].copy();value[415]=np.log1p(1);values.append(value);rows.append(dict(parent_row=parent_row,child_index=None,family=parent['family'],invalid=True));continue
        if parent['family']=='first':
            face=parent['face'];candidate=parent['candidate']
            if face not in banks:
                ep=events/f'face_{face}.npz';assert digest(ep)==fm[face]['event_file_sha256'];xp=features/fm[face]['file'];assert digest(xp)==fm[face]['sha256'];ap=arcs/am[face]['file'];assert digest(ap)==am[face]['sha256']
                assert digest(first_prediction/pm[face]['file'])==pm[face]['sha256']
                with np.load(ep) as z:banks[face]=(np.load(xp,mmap_mode='r'),z['positions_mm'],z['offsets'],z['vertex_indices'],read(ap)['candidates'],read(first_prediction/pm[face]['file'])['rows'])
            x,allq,offsets,ids,records,predictions=banks[face];original_path=predictions[candidate]['selected_local_event_indices'];a,b=offsets[candidate:candidate+2];q=allq[a:b];extra=np.column_stack((np.zeros((b-a,2)),ids[a:b]>=0,np.ones(b-a),np.log1p(q),np.log1p(q[-1]-q),np.zeros(b-a)));x=np.column_stack((x[a:b],extra));arc=CircularArc.from_parameters(records[candidate]['arc_parameters'])
        else:
            candidate=parent['profile'];original_path=second_paths[candidate]['selected_local_event_indices'][parent['interval_index']:parent['interval_index']+2];a,b=second_offsets[candidate:candidate+2];q=second_q[a:b];x=second_x[a:b];arc=CircularArc.from_parameters(profiles[candidate]['representation']['arc_parameters'])
        points=arc.point_at(q)
        for child_index,segment in enumerate(parent['parent']['segments']):
            ends=np.asarray(segment['points'])[[0,-1]];distance=np.linalg.norm(points[:,None]-ends[None],axis=2);path=original_path[child_index:child_index+2];assert len(path)==2 and q[path[1]]>q[path[0]];assert np.max(distance[path,np.arange(2)])<=1e-6;values.append(describe(x,q,path,arc));rows.append(dict(parent_row=parent_row,child_index=child_index,family=parent['family'],invalid=False))
    out=BASE/'joint_arc_child_quality_data_v1';out.mkdir(exist_ok=False);np.save(out/'features.npy',np.stack(values));write(out/'rows.json',dict(rows=rows));write(out/'input_receipt.json',dict(parent_data_receipt_sha256=digest(source/'receipt.json'),feature_sha256=digest(out/'features.npy'),rows_sha256=digest(out/'rows.json'),dimensions=423,scope='All frozen output children, same descriptor semantics and source inputs; parent grouping retained by parent_row. Invalid parents use one quality row and retain original counted false children. Input extraction before target reads.',GT_reads=0))
    refs={};target_hashes={}
    for family,key in [('first','new_ac2008a03607'),('second','new_fcaf2af1abee')]:
        p=BASE/'labels'/key/'targets.json';target_hashes[family]=digest(p);truth=native_targets(read(p));refs[family]=[(c,reference_segment(c)) for parent in truth['parents'] for c in parent['children']]
    labels=[]
    for row in rows:
        matches=[]
        if not row['invalid']:
            segment=parents[row['parent_row']]['parent']['segments'][row['child_index']];reverse=dict(segment,**reverse_segment(segment))
            for index,(child,ref) in enumerate(refs[row['family']]):
                if child_error(segment,child,ref,False,False) is not None or child_error(reverse,child,ref,False,False) is not None:matches.append(index)
        labels.append(dict(quality=float(bool(matches)),compatible_reference_children=matches))
    write(out/'labels.json',dict(rows=labels,target_sha256=target_hashes,scope='Individual child reference geometry quality; no claim of physical absence for unmatched. Whole original parents retained unchanged at evaluation; individual reversal here does not authorize childwise flipping of outputs.'));write(out/'receipt.json',dict(input_receipt_sha256=digest(out/'input_receipt.json'),labels_sha256=digest(out/'labels.json'),rows=len(rows),positive_rows=sum(bool(r['quality']) for r in labels),DEV_reads=0));print('rows',len(rows),'positive',sum(bool(r['quality']) for r in labels),flush=True)

if __name__=='__main__':main()
