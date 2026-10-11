"""Semantic controls for partial parents, raw signs and unknown process labels.

These fixtures are deliberately synthetic and never written as model output.
"""
import copy
import hashlib
import json

import numpy as np

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_complete_output_assessment import assess_complete_output, process_reference, FIELDS
from weld_pipeline.world_set_runtime_285 import line_segment
from tools.assess_full_weld_output_fields import pool_freeze_fingerprint


def controls():
    segments = [dict(line_segment([0, 0, 0], [20, 0, 0]), weld_type='horizontal'),
                dict(line_segment([20, 0, 0], [20, 30, 0]), weld_type='horizontal')]
    target = dict(original_children=2, parents=[dict(original_parent_id='p',
        original_child_count=2, is_closed=False, gaps=['SHARED'], children=[
            dict(original_child_id=str(i), original_kind='line', weld_type='horizontal',
                 representation=dict(kind='line', endpoints_mm=s['endpoints_mm']))
            for i, s in enumerate(segments)])])
    records = [dict(parent_id='p', child_id=str(i), child_index=i) for i in range(2)]
    state = np.full((2, 8), -1, np.int8)
    state[0, 0], state[0, 4], state[1, 6] = 0, 2, 1
    numeric = np.full((2, 8), np.nan)
    numeric[0, 0] = 12.
    arrays = dict(state=state, required_mask=np.ones((2, 8), bool), exact_mm=numeric)
    evidence = dict(parents=[dict(parent_id='p', children=[dict(child_id=str(i),
        endpoint_frames={name:dict(known=True, raw_export_axes=np.eye(3).tolist(), translation=point)
                         for name, point in zip(('start', 'end'), s['endpoints_mm'])})
        for i, s in enumerate(segments)])])
    reference = process_reference(target, records, arrays, evidence)
    for i, segment in enumerate(segments):
        segment.update(endpoint_attributes={name:dict(state='EXACT_MM', value_mm=0.) for name in FIELDS},
            raw_export_frame_axes=[np.eye(3).tolist()]*2,
            frame_origins_mm=segment['endpoints_mm'], predicted_side='UNKNOWN',
            frame_business_semantics_certified=False)
    segments[0]['endpoint_attributes']['front_wall']['value_mm'] = 15.
    segments[0]['endpoint_attributes']['start_hole_height'] = dict(state='NO_HOLE', value_mm=None)
    segments[1]['endpoint_attributes']['start_base_edge'] = dict(state='DISTANCE_GT_50', value_mm=None)

    def decoded(parts):
        return dict(parents=[dict(child_count=2, is_closed=False, gaps=['SHARED'], segments=parts)],
                    invalid_predictions=[], total_prediction_children=2)

    before = copy.deepcopy((target, evidence, segments))
    good = assess_complete_output(decoded(segments), target, reference)
    assert good['development_d3_geometry']['overall']['TP'] == 2
    assert good['complete_parent_development_geometry']['fully_matched_parents'] == 1
    assert good['fields']['front_wall']['absolute_error_mm']['maximum'] == 3.
    assert good['fields']['front_wall']['unknown_required'] == 1
    assert good['fields']['start_hole_height']['absolute_error_mm']['count'] == 0
    assert good['fields']['start_base_edge']['absolute_error_mm']['count'] == 0
    assert sum(row['compared'] for row in good['fields'].values()) == 3
    assert good['raw_frames']['business_frame_certified_predictions'] == 0
    assert good['all_output_completeness']['unknown_sides'] == 2
    assert not good['full_contract_pass']
    partial = copy.deepcopy(segments)
    partial[1].update(line_segment([20, 0, 0], [20, 30, 5]))
    result = assess_complete_output(decoded(partial), target, reference)
    assert result['development_d3_geometry']['overall']['TP'] == 1
    assert result['complete_parent_development_geometry']['fully_matched_parents'] == 0
    assert result['complete_parent_development_geometry']['partially_matched_parents'] == 1
    assert result['fields']['start_base_edge']['not_geometry_matched_known_required'] == 1
    flipped = copy.deepcopy(segments)
    flipped[0]['raw_export_frame_axes'] = [np.diag([-1., -1., 1.]).tolist()]*2
    signed = assess_complete_output(decoded(flipped), target, reference)
    assert signed['raw_frames']['signed_axis_errors_deg']['X']['maximum'] == 180.
    assert signed['raw_frames']['signed_axis_errors_deg']['Y']['maximum'] == 180.
    assert signed['raw_frames']['signed_axis_errors_deg']['Z']['maximum'] == 0.
    reversed_parts = copy.deepcopy(segments[::-1])
    for segment in reversed_parts:
        segment.update(line_segment(*segment['endpoints_mm'][::-1]))
    reverse = assess_complete_output(decoded(reversed_parts), target, reference)
    assert reverse['development_d3_geometry']['overall']['TP'] == 2
    assert reverse['raw_ordered_geometry']['overall']['TP'] == 0
    assert sum(row['compared'] for row in reverse['fields'].values()) == 0
    assert reverse['raw_frames']['origin_error_mm']['count'] == 0
    malformed = copy.deepcopy(segments)
    malformed[0]['raw_export_frame_axes'] = [[None]]
    malformed[0]['endpoint_attributes']['front_wall']['value_mm'] = None
    missing = assess_complete_output(decoded(malformed), target, reference)
    assert missing['fields']['front_wall']['numeric_prediction_missing'] == 1
    assert missing['raw_frames']['missing_predicted_axes'] == 2
    assert (target, evidence, segments) == before
    invalid = dict(arrays, state=state.astype(float))
    try:
        process_reference(target, records, invalid, evidence)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonintegral label-state dtype was accepted')
    original = dict(job_key='synthetic', geometry_GT_reads=0, STEP_reads=0)
    original_sha = hashlib.sha256(json.dumps(original, ensure_ascii=False, indent=2).encode()).hexdigest()
    augmented = dict(original, semantic_fingerprints={'synthetic_input': 'a'*64})
    saved_sha = hashlib.sha256(json.dumps(augmented, ensure_ascii=False, indent=2).encode()).hexdigest()
    assert pool_freeze_fingerprint(augmented, saved_sha, 'TRAIN') == original_sha
    assert pool_freeze_fingerprint(augmented, saved_sha, 'MODEL_SELECTION_ONLY') == saved_sha
    assert pool_freeze_fingerprint(dict(augmented, STEP_reads=1), saved_sha, 'TRAIN') != original_sha
    return dict(status='PASS', synthetic_semantic_controls_only=True, optimizer_updates=0,
        partial_parent_not_complete=True, unknown_fields_not_graded=True,
        censored_states_not_distance_values=True, raw_axis_sign_flip_is_180_degrees=True,
        whole_parent_reversal_not_raw_direction_or_frame_pass=True,
        missing_predictions_not_filled=True, reference_and_prediction_unchanged=True,
        train_original_receipt_bytes_still_sha_authenticated=True,
        held_receipt_reconstruction_disallowed=True, changed_pool_binding_rejected=True,
        model_accuracy_not_evaluated=True, full_contract_pass=False)


if __name__ == '__main__':
    print(json.dumps(controls(), ensure_ascii=False))
