"""Free Hline event decoding and attributes in the event's own current chart."""
import torch
from weld_pipeline.legacy_incremental_event_runtime_393 import decode_candidate_events, as_numpy
from weld_pipeline.legacy_incremental_attribute_runtime_404 import feature_path_for_own_prediction, decode_fields
from weld_pipeline.legacy_incremental_correspondence_414 import canonical_child_fields
from weld_pipeline.contract_field_acceptance_297 import FIELDS


def decode_horizontal_line_events(sequences, output, threshold=.5):
    indices = [index for index, sequence in enumerate(sequences)
               if sequence['type'] == 'horizontal' and sequence.get('curve_kind') != 'arc']
    selected = [sequences[index] for index in indices]
    keys = ('presence_logits', 'count_logits', 'positions_mm', 'gap_logits', 'direction_logits', 'closed_logits')
    arrays = {key: as_numpy(output[key])[indices] for key in keys}
    result = decode_candidate_events(selected, arrays, threshold, ('horizontal',))
    for parent in result['predictions']:
        original = indices[parent['original_candidate_sequence_index']]
        parent['original_candidate_sequence_index'] = original
        parent['id'] = 'event_%05d_%03d' % (original, parent['learned_parent_query_index'])
        assert parent['type'] == 'horizontal' and all(segment['kind'] == 'line' for segment in parent['segments'])
    for row in result['invalid_positive_ledger'] + result['parent_relations']:
        row['sequence_index'] = indices[row['sequence_index']]
        assert row['type'] == 'horizontal' and row['kind'] == 'line'
    result.update(version='HLINE_EVENTS_ORIGINAL_COMPLETE_FACTORY_CHART_v1',
                  selected_original_candidate_indices=indices, learned_HArc_events_excluded_before_query_decode=True)
    return result


@torch.no_grad()
def estimate_event_H_attributes(model, predictions, current_mixed_sequences):
    records = []
    for parent in predictions:
        assert parent['type'] == 'horizontal'
        forced = parent['original_candidate_sequence_index']
        assert current_mixed_sequences[forced]['type'] == 'horizontal'
        assert current_mixed_sequences[forced].get('curve_kind') != 'arc'
        for child_index, segment in enumerate(parent['segments']):
            assert segment['kind'] == 'line'
            data = feature_path_for_own_prediction(segment, current_mixed_sequences, forced)
            record = dict(parent_id=parent['id'], child_index=child_index, weld_side='unknown',
                formal_E_truth_certified=False, original_GT_or_STEP_reads=0,
                local_scope_not_expanded_to_whole_curve=True, forced_original_candidate_sequence_index=forced)
            if data is None:
                record.update(fields={name: dict(status='unknown', value=None, source=None,
                    reason='OWN_PREDICTION_HAS_NO_ORIGINAL_FEATURE_CHART') for name in FIELDS}, local_estimates=[])
            else:
                assert data['own_candidate_index'] == forced
                device = next(model.events.parameters()).device
                local = torch.from_numpy(data['features']).to(device)
                canonical = local.flip(0) if data['negative_chart_direction'] else local
                whole, pointwise = model.path_attributes(canonical.reshape(1, -1)), model.attributes(local)
                record.update(fields=canonical_child_fields(decode_fields(whole, 0, scope='WHOLE_PATH',
                    whole_known_fields=model.config.get('whole_path_known_supervision_fields')), data['negative_chart_direction']),
                    own_candidate_index=data['own_candidate_index'],
                    local_estimates=[dict(predicted_child_fraction=float(index / 16),
                        world_xyz_mm=data['world_positions_mm'][index].tolist(), fields=decode_fields(pointwise, index),
                        scope='POINTWISE_ESTIMATE_ONLY') for index in range(17)])
            records.append(record)
    return records
