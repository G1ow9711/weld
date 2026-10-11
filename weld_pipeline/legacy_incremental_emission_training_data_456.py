"""TRAIN-only303 protection and unique complete-parent query anchors."""
import numpy as np
from weld_pipeline.legacy_incremental_corrective_training_data_424 import pack_source as corrective_pack
from weld_pipeline.legacy_incremental_supervision_388 import read, digest
from weld_pipeline.legacy_incremental_loss_396 import world_positions
from weld_pipeline.legacy_incremental_effective_nodes_422 import effective_nodes_numpy
from weld_pipeline.reports.goal_contract_training_20261005.score_legacy_full65_frozen_385 import adapt_predictions


def pack_source(root, free_directory, mapping_path, expected_mapping_sha, plate_heads, q, c,
                correct_teacher_ids, H_mapping_path, H_mapping_sha, *, feedback_path, feedback_sha256,
                first2_feedback_path, first2_feedback_sha256, first2_free_directory, first2_score_path):
    import torch
    from pathlib import Path
    first2 = read(first2_feedback_path)
    assert digest(first2_feedback_path) == first2_feedback_sha256
    refs = first2['correct_current_V_world_reference_by_original_child_id']
    protected = set(correct_teacher_ids) | set(refs)
    source = corrective_pack(root, free_directory, mapping_path, expected_mapping_sha, plate_heads, q, c,
        sorted(protected), H_mapping_path, H_mapping_sha, feedback_path=feedback_path, feedback_sha256=feedback_sha256)
    assert source['source_index_training_metadata_only'] == first2['source_index_training_metadata_only']
    folder = Path(first2_free_directory)
    receipt = read(folder / 'receipt.json')
    assert digest(folder / 'predictions.json') == receipt['frozen_outputs_sha256']['predictions.json']
    prior_predictions = read(folder / 'predictions.json')
    adapted = adapt_predictions(prior_predictions, read(folder / 'invalid_positive_ledger.json'))
    matches = {row['original_child_id']: row for row in read(first2_score_path)['ordered_geometry']['matches']}
    proposals, partial = {}, 0
    for si, records in enumerate(source['parents']):
        for record in records:
            ids, n = record['original_child_ids'], record['original_child_count']
            record['protected_complete_parent'] = all(cid in protected for cid in ids)
            mask = [cid in refs for cid in ids]
            if any(mask):
                world = [refs.get(cid) for cid in ids]
                if record['parent_chart_direction'] == -1:
                    world, mask = [None if p is None else p[::-1] for p in world[::-1]], mask[::-1]
                truth = record['canonical_original_world_endpoints_mm']
                record['corrective_keep_child_mask'] = mask
                record['corrective_keep_world_endpoints_mm'] = [truth[i] if p is None else p for i, p in enumerate(world)]
            if not record['protected_complete_parent']:
                partial += int(any(cid in protected for cid in ids))
                continue
            eligible = []
            if all(cid in refs and cid in matches for cid in ids):
                parent_indices = {matches[cid]['predicted_parent_index'] for cid in ids}
                if len(parent_indices) == 1:
                    parent = prior_predictions[adapted['parents'][parent_indices.pop()]['original_prediction_index']]
                    if parent['type'] == 'vertical' and len(parent['segments']) == n:
                        assert parent['original_candidate_sequence_index'] == si, 'Protected source owner cannot silently change'
                        eligible = [int(parent['learned_parent_query_index'])]
            else:
                # Only exact unique full-count/gap/world correspondence qualifies.
                wanted = np.asarray(record['canonical_original_world_endpoints_mm'])
                gap_status = record['original_gap_status'][::-1] if record['parent_chart_direction'] == -1 else record['original_gap_status']
                for qi in range(q):
                    if source['base']['presence_logits'][si, qi] < 0 or int(source['base']['count_logits'][si, qi].argmax()) != n:
                        continue
                    if bool(source['base']['closed_logits'][si, qi] >= 0.) != record['parent_is_closed']:
                        continue
                    gaps = source['base']['gap_logits'][si, qi, :n - 1] >= 0.
                    if gaps.tolist() != [s == 'POSITIVE_GAP' for s in gap_status]:
                        continue
                    points = effective_nodes_numpy(source['base']['positions_mm'][si, qi, :n], gaps)
                    actual = world_positions(torch.from_numpy(points), record['candidate_geometry']).numpy()
                    if np.linalg.norm(actual - wanted, axis=-1).max() <= 3.:
                        eligible.append(qi)
            if len(eligible) == 1:
                proposals.setdefault((si, eligible[0]), []).append(record)
    anchors = 0
    for (si, qi), records in proposals.items():
        if len(records) != 1:
            continue
        record = records[0]
        assert record['protected_complete_parent']
        assert source['hard_negative_weights'][si, qi] == 1., 'Protected complete parent must not also be classified hard background'
        record['fixed_train_query'] = qi
        anchors += 1
    source.update(current_correct_V_ids_used_only_in_TRAIN=len(refs),
                  protected_complete_train_anchor_count=anchors, protected_partial_parent_count=partial,
                  protected_original_child_ids_training_only=sorted(protected),
                  protected303_source_references_retained=True)
    mapping = read(mapping_path)
    payload = Path(root) / mapping['E16_canonical_payload_path']
    assert digest(payload) == mapping['E16_canonical_payload_sha256']
    source['path_original_child_ids'] = [row['original_child_id'] for row in read(payload)['children']]
    assert len(source['path_original_child_ids']) == len(source['path_features']) == source['original_G']
    return source
