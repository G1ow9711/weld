"""Offline TRAIN adapter for sealed fresh factories and complete supervision.

Runtime must construct the same features from its current original PCD. This
module is never imported by the free predictor. Missing representations and
unknown whole-path fields stay in the original denominators.
"""
from pathlib import Path
import numpy as np
from weld_pipeline.contract_field_acceptance_297 import FIELDS
from weld_pipeline.legacy_incremental_supervision_388 import load_fresh_training_interfaces,read,digest
from weld_pipeline.legacy_incremental_features_395 import prepare_candidate_inputs
from weld_pipeline.legacy_incremental_loss_396 import canonical_target
from weld_pipeline.legacy_incremental_correspondence_414 import canonical_child_fields
from weld_pipeline.world_set_evaluation_285 import reference_segment

STATES=('unknown','value','no_object','out_of_range','mixed')

def sample_features(sequence,positions_mm):
    f=np.asarray(sequence['features'],np.float32)
    at=np.asarray(positions_mm,np.float64)/float(sequence['sample_step_mm'])
    assert np.isfinite(at).all() and np.all(at>=-1e-7) and np.all(at<=len(f)-1+1e-7)
    # Interpolation indices are representation coordinates, not GT fitting.
    at=np.clip(at,0.,len(f)-1);lo=np.floor(at).astype(int);hi=np.minimum(lo+1,len(f)-1)
    fraction=(at-lo).astype(np.float32)
    return (f[lo]*(1.-fraction[:,None])+f[hi]*fraction[:,None]).astype(np.float32)

def empty_attributes(count):
    return dict(state_class=np.zeros((count,16),np.int64),state_known=np.zeros((count,16),bool),
        scalar_values=np.full((count,16),np.nan,np.float32),scalar_known=np.zeros((count,16),bool),
        signed_normals=np.full((count,2,3),np.nan,np.float32),normal_known=np.zeros((count,2),bool),
        grade_weight=np.ones((count,16),np.float32))

def ordinary_attributes(children):
    target=empty_attributes(len(children))
    for i,child in enumerate(children):
        assert set(child['fields'])==set(FIELDS)
        for j,name in enumerate(FIELDS):
            row=child['fields'][name]
            if not row['known_payload_available']:continue
            state=row['status'];assert state in STATES and state!='unknown'
            target['state_class'][i,j]=STATES.index(state);target['state_known'][i,j]=True
            if state!='value':continue
            value=row['value']
            if name.startswith('normal_'):
                side=int(name[-1])-1;target['signed_normals'][i,side]=value;target['normal_known'][i,side]=True
            else:
                assert name not in ('hole_edge','segment_reason')
                target['scalar_values'][i,j]=float(value);target['scalar_known'][i,j]=True
    return target

def local_attributes(samples):
    target=empty_attributes(len(samples))
    target['grade_weight'][:]=.5
    for i,sample in enumerate(samples):
        assert not sample['formal_E_truth_certified']
        if not sample['pointwise_payload_valid']:continue
        for side in (0,1):
            j=FIELDS.index('thickness_'+str(side+1));n=FIELDS.index('normal_'+str(side+1))
            target['state_class'][i,[j,n]]=1;target['state_known'][i,[j,n]]=True
            target['scalar_values'][i,j]=sample['plate_'+str(side+1)+'_thickness_mm'];target['scalar_known'][i,j]=True
            target['signed_normals'][i,side]=sample['plate_'+str(side+1)+'_signed_normal'];target['normal_known'][i,side]=True
    return target

def pack_source(root,free_directory,mapping_path,expected_mapping_sha,plate_heads,q,c,correct_teacher_ids=(),H_mapping_path=None,H_mapping_sha=None):
    root=Path(root);free_directory=Path(free_directory);mapping_path=Path(mapping_path)
    assert digest(mapping_path)==expected_mapping_sha
    mapping=read(mapping_path)
    directory=(root/mapping['fresh_capture_manifest_path']).parent
    fresh=load_fresh_training_interfaces(directory,mapping['fresh_capture_manifest_sha256'])
    sequences=fresh['sequences'];arrays=fresh['arrays'];rows=fresh['manifest']['sequences']
    probs=[arrays['original_parent_probabilities'][r['start']:r['stop']] for r in rows]
    pair=read(free_directory/'receipt.json');route=pair['routes']['hybrid']
    diagnostic_path=free_directory/'hybrid/vertical_diagnostic.json'
    fingerprints={name.replace('\\','/'):sha for name,sha in route['frozen_outputs_sha256'].items()}
    assert digest(diagnostic_path)==fingerprints['hybrid/vertical_diagnostic.json']
    x,base,overflow=prepare_candidate_inputs(sequences,probs,arrays['structural'],arrays['process'],
        read(diagnostic_path),plate_heads['split_fractions'],q,c)
    assert not overflow,'Increase the one global capacity before registration; never discard frozen teacher children'
    truth_path=root/mapping['geometry_targets_path'];assert digest(truth_path)==mapping['geometry_targets_sha256']
    truth=read(truth_path)
    canonical_path=root/mapping['E16_canonical_payload_path'];assert digest(canonical_path)==mapping['E16_canonical_payload_sha256']
    canonical=read(canonical_path)
    children=canonical['children'];by_id={child['original_child_id']:child for child in children}
    geometry_by_id={child['original_child_id']:child for parent in truth['parents'] for child in parent['children']}
    assert set(by_id)==set(geometry_by_id) and len(by_id)==mapping['original_G']
    mapped_by_id={cid:(r,ci) for r in mapping['mapped_parents'] for ci,cid in enumerate(r['original_child_ids'])}
    parents=[[] for _ in sequences]
    correct_teacher_ids=set(correct_teacher_ids)
    for record in mapping['mapped_parents']:
        record=dict(record);si=record['sequence_index'];n=record['original_child_count']
        seq=sequences[si]
        endpoints=np.asarray([np.asarray(reference_segment(geometry_by_id[cid])['points'])[[0,-1]] for cid in record['original_child_ids']])
        if record['parent_chart_direction']==-1:endpoints=endpoints[::-1,::-1]
        record.update(canonical_original_world_endpoints_mm=endpoints.tolist(),direction_semantics_confirmed=None,
            candidate_geometry=dict(kind='arc',curve=seq['curve']) if seq.get('curve_kind')=='arc' else
             dict(kind='line',origin_mm=np.asarray(seq['xyz'][0]).tolist(),direction=np.asarray(seq['direction']).tolist()))
        mask=[cid in correct_teacher_ids for cid in record['original_child_ids']]
        ordered,negative=canonical_target(record);eligible=[]
        if any(mask):
            for qi in range(q):
                if int(base['count_logits'][si,qi].argmax())!=n or base['presence_logits'][si,qi]<0.:continue
                residual=np.abs(base['positions_mm'][si,qi,:n]-ordered)
                if float(residual.max())<=3.:eligible.append((float(residual.max()),qi))
        if eligible:
            _,qi=min(eligible);record.update(teacher_correct_child_mask=mask,teacher_correct_query_index=qi,
                keep_teacher_applies_only_to_previously_correct_children=True)
        parents[si].append(record)
    # Unmapped positives cannot be taught as negative background. Mask the
    # affected type for this source until candidate coverage is resolved.
    missed_types={geometry_by_id[cid]['weld_type'] for row in mapping['representation_misses'] for cid in row['original_child_ids']}
    background_known=np.array([seq['type'] not in missed_types for seq in sequences],bool)
    groups=np.array([2 if seq.get('curve_kind')=='arc' else 1 if seq['type']=='vertical' else 0 for seq in sequences],np.int64)
    path_x=np.zeros((len(children),17*84),np.float32);path_available=np.zeros(len(children),bool)
    local_x=[];local_samples=[];local_available=[];local_keys=[]
    canonical_children=[]
    for child_index,child in enumerate(children):
        cid=child['original_child_id'];attached=mapped_by_id.get(cid)
        if attached:
            record,ci=attached;seq=sequences[record['sequence_index']];a,z=record['child_parameter_endpoints_mm'][ci]
            path_x[child_index]=sample_features(seq,np.linspace(min(a,z),max(a,z),17)).reshape(-1);path_available[child_index]=True
            canonical_children.append(dict(child,fields=canonical_child_fields(child['fields'],z<a)))
        else:canonical_children.append(child)
        for sample in child['ordered_auxiliary_material_samples']:
            available=bool(attached and sample['pointwise_payload_valid'])
            if available:
                value=a+float(sample['fraction'])*(z-a);feature=sample_features(seq,[value])[0]
            else:feature=np.zeros(84,np.float32)
            local_x.append(feature);local_samples.append(sample);local_available.append(available);local_keys.append(sample['key'])
    path_target=ordinary_attributes(canonical_children)
    for key in ('state_known','scalar_known','normal_known'):path_target[key][~path_available]=False
    local_target=local_attributes(local_samples);local_available=np.asarray(local_available,bool)
    for key in ('state_known','scalar_known','normal_known'):local_target[key][~local_available]=False
    shadow=[]
    H_sequences=sequences;H_probs=probs;H_parents=parents
    if H_mapping_path is not None:
        H_mapping_path=Path(H_mapping_path);assert digest(H_mapping_path)==H_mapping_sha
        hm=read(H_mapping_path);hdir=(root/hm['fresh_capture_manifest_path']).parent
        assert digest(hdir/'manifest.json')==hm['fresh_capture_manifest_sha256'];hmanifest=read(hdir/'manifest.json')
        identical=(hmanifest['sequences']==fresh['manifest']['sequences'] and all(hmanifest['files'][name]['array_payload_sha256']==fresh['manifest']['files'][name]['array_payload_sha256'] for name in ('xyz.npy','features.npy')))
        if identical:
            # Actual immutable payloads match; share feature pages within this
            # TRAIN source while retaining the original115 probabilities.
            H_sequences=sequences
            pp=hdir/'original_parent_probabilities.npy';assert digest(pp)==hmanifest['files'][pp.name]['file_sha256']
            hp=np.load(pp,mmap_mode='r',allow_pickle=False)
            H_probs=[hp[r['start']:r['stop']] for r in hmanifest['sequences']]
        else:
            hf=load_fresh_training_interfaces(hdir,hm['fresh_capture_manifest_sha256'])
            H_sequences=hf['sequences'];H_probs=[hf['arrays']['original_parent_probabilities'][r['start']:r['stop']] for r in hf['manifest']['sequences']]
        H_parents=[[] for _ in H_sequences]
        for record in hm['mapped_parents']:H_parents[record['sequence_index']].append(record)
    for si,seq in enumerate(H_sequences):
        if seq['type']!='horizontal':continue
        step=float(seq['sample_step_mm']);n=len(seq['features']);target=np.zeros((n,4),np.float32)
        at=np.arange(n)*step
        for record in H_parents[si]:
            ordered,_=canonical_target(record)
            for left,right in ordered:target[:,0]=np.maximum(target[:,0],((at>=left)&(at<=right)).astype(np.float32))
            for value,channel in [(float(ordered[:,0].min()),1),(float(ordered[:,1].max()),2)]+[(float(p[1]),3) for p in ordered[:-1]]:
                target[:,channel]=np.maximum(target[:,channel],np.exp(-.5*((at-value)/1.5)**2))
        shadow.append(dict(sequence_index=si,features=seq['features'],frozen_probabilities=H_probs[si],targets=target,
            background_known=seq['type'] not in missed_types,has_supervised_positive=bool(H_parents[si])))
    assert len(local_samples)==sum(len(ch['ordered_auxiliary_material_samples']) for ch in children)
    return dict(features=x,base=base,parents=parents,background_known=background_known,group_ids=groups,
        path_features=path_x,path_targets=path_target,path_available=path_available,
        local_features=np.asarray(local_x,np.float32).reshape(-1,84),local_targets=local_target,
        local_available=local_available,local_reference_keys=local_keys,shadow=shadow,
        source_index_training_metadata_only=mapping['source_index_training_metadata_only'],
        original_G=mapping['original_G'],original_E16_records=len(children)*16,original_local_points=len(local_samples),
        representation_miss_children=mapping['representation_miss_children'],
        original_child_ids=[child['original_child_id'] for child in children],
        masked_background_candidate_count=int((~background_known).sum()),
        whole_path_fields_and_features_use_one_positive_chart_correspondence=True,
        local_scope_not_expanded_to_whole_curve=True,official_E_truth_records=0)
