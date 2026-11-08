"""Recheck the unchanged competing-face model gate after all recoveries.

Post-model derived sources have no learned source observations. They remain
UNKNOWN to this gate and are neither rejected nor used as endorsed anchors.
"""
import copy
import time
from . import curveonly_01a1ac7c_paired_face_rules_v407 as previous


def correct(decoded, features, packet, inventory, solids, source_expert):
    start = time.perf_counter()
    model_count = len(packet['candidates'])
    indices = [i for i, p in enumerate(decoded['parents'])
               if 0 <= p['source_candidate'] < model_count]
    # The exact original packet/features supply every expert vote. There are
    # deliberately no appended rows for post-model publication geometry.
    view = dict(decoded, parents=[decoded['parents'][i] for i in indices])
    chosen, audit = previous.inspect(view, features, packet, inventory, solids, source_expert)
    audit = copy.deepcopy(audit)
    for row in audit.get('certificates', []) + audit.get('rejected', []):
        row['parent_index'] = indices[row['parent_index']]
        for anchor in row.get('opposite_face_anchors', []):
            anchor['parent_index'] = indices[anchor['parent_index']]
    removed = {indices[row['parent_index']] for row in chosen}
    result = copy.deepcopy(decoded)
    if removed:
        result['parents'] = [p for i, p in enumerate(result['parents']) if i not in removed]
        for i, parent in enumerate(result['parents']):
            parent['parent_id'] = f'curve_{i:05d}'
        result.update(geometry_units=len(result['parents']),
                      published_segments=sum(len(p['segments']) for p in result['parents']))
    audit.update(stage='FINAL_PUBLICATION_AFTER_ALL_MODEL_RECOVERIES',
        original_gate_version=407, original_gate_parameters_unchanged=True,
        removed_parent_indices=sorted(removed),
        post_model_sources_retained_unknown=[i for i in range(len(decoded['parents'])) if i not in indices],
        no_features_or_source_votes_fabricated=True,
        early_gate_audit_retained=True, retained_fields_exact_except_parent_id=True,
        no_new_fit_or_VALcalibration=True)
    result['final_paired_face_source_rule'] = audit
    result['final_paired_face_source_rule_seconds'] = time.perf_counter() - start
    return result
