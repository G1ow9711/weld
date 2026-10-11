"""Offline correction feedback; never imported by the raw predictor."""
import numpy as np
from weld_pipeline.legacy_incremental_training_data_400 import pack_source as original_pack_source
from weld_pipeline.legacy_incremental_supervision_388 import read,digest


def pack_source(*args,feedback_path,feedback_sha256,**kwargs):
    assert digest(feedback_path)==feedback_sha256
    feedback=read(feedback_path);source=original_pack_source(*args,**kwargs)
    assert source['source_index_training_metadata_only']==feedback['source_index_training_metadata_only']
    b,q=source['base']['presence_logits'].shape
    original_masked=int((~source['background_known']).sum())
    source['background_known']=np.asarray(feedback['event_background_known'],bool)
    assert len(source['background_known'])==b
    source['masked_background_candidate_count']=int((~source['background_known']).sum())
    source['prior_whole_type_masked_background_candidate_count']=original_masked
    negative=np.ones((b,q),np.float32)
    for si,qi in feedback['hard_negative_event_queries']:
        assert source['background_known'][si]
        negative[si,qi]=4.
    source['hard_negative_weights']=negative
    refs=feedback['correct_current_V_world_reference_by_original_child_id']
    for records in source['parents']:
        for record in records:
            ids=record['original_child_ids'];mask=[cid in refs for cid in ids]
            if not any(mask):continue
            # Ground-truth orientation is converted once to the existing
            # positive candidate chart; physical coordinates are never moved.
            world=[refs.get(cid) for cid in ids]
            truth=np.asarray(record['canonical_original_world_endpoints_mm'])
            if record['parent_chart_direction']==-1:
                world=[None if p is None else p[::-1] for p in world[::-1]];mask=mask[::-1]
            value=np.asarray([truth[i] if p is None else p for i,p in enumerate(world)],np.float32)
            record.update(corrective_keep_child_mask=mask,corrective_keep_world_endpoints_mm=value)
    for row in source['shadow']:
        n=len(row['features']);step=feedback['H_shadow_sample_step_mm'][str(row['sequence_index'])]
        # Exact sample coordinates are obtained from the sealed source's
        # manifest, not inferred from a file name or a truth coordinate.
        row['background_known']=feedback['H_background_known_by_candidate'][row['sequence_index']]
        regions=feedback['H_shadow_regions'].get(str(row['sequence_index']),{})
        weight=np.ones((n,4),np.float32);keep=np.zeros((n,4),bool)
        for left,right in regions.get('hard_fp_intervals',[]):
            selected=(np.arange(n)*step>=left-3.)&(np.arange(n)*step<=right+3.)
            weight[selected]=4.
        for left,right in regions.get('protected_correct_intervals',[]):
            selected=(np.arange(n)*step>=left-3.)&(np.arange(n)*step<=right+3.)
            keep[selected]=True
        row.update(negative_weights=weight,protected_correct_field_mask=keep)
    source['current_correct_V_ids_used_only_in_TRAIN']=len(refs)
    return source
