"""Whole-source streamed role selection using predicted counts and scores.

No torch or supervision files are read. Every original anchor score is consumed;
chunk size cannot change the selected canonical slots. Ties use original IDs.
"""
import numpy as np

from .source_graph_targets_275 import SLOT_COLUMNS


def select_full_source_slots(score_blocks, anchor_count, predicted_counts):
    if type(anchor_count) is not int or anchor_count < 1:
        raise ValueError("Original source anchor capacity required")
    counts = np.asarray(predicted_counts)
    if counts.shape != (len(SLOT_COLUMNS),) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("Explicit six predicted integer cardinalities required")
    if (counts < 0).any() or (counts > anchor_count).any():
        raise ValueError("Predicted count exceeds unchanged original role capacity")
    best_rows = [np.empty(0, np.int64) for _ in SLOT_COLUMNS]
    best_scores = [np.empty(0, np.float64) for _ in SLOT_COLUMNS]
    visited = 0
    for first, block in score_blocks:
        scores = np.asarray(block)
        if type(first) is not int or first != visited:
            raise ValueError("Every original anchor block must occur once in original order")
        if (scores.ndim != 2 or scores.shape[1] != len(SLOT_COLUMNS) or len(scores) == 0
                or first + len(scores) > anchor_count or not np.isfinite(scores).all()):
            raise ValueError("Whole original role-score block is incomplete or invalid")
        original_rows = np.arange(first, first + len(scores), dtype=np.int64)
        for role, count in enumerate(counts.tolist()):
            if not count:
                continue
            order = np.lexsort((original_rows, -scores[:, role]))[:count]
            rows = np.concatenate((best_rows[role], original_rows[order]))
            values = np.concatenate((best_scores[role], scores[order, role].astype(np.float64)))
            keep = np.lexsort((rows, -values))[:count]
            best_rows[role], best_scores[role] = rows[keep], values[keep]
        visited += len(scores)
    if visited != anchor_count:
        raise ValueError("Runtime selection did not consume the complete original anchor denominator")
    slots = []
    for role, count in enumerate(counts.tolist()):
        if len(best_rows[role]) != count:
            raise ValueError("Selected role count differs from its own neural prediction")
        slots.extend((int(row) + 1, role, SLOT_COLUMNS[role][1]) for row in best_rows[role])
    slots.sort()
    return np.asarray(slots, dtype=np.int64).reshape(-1, 3), dict(
        full_original_anchor_count=anchor_count, full_original_role_slots=anchor_count * len(SLOT_COLUMNS),
        original_anchor_rows_consumed=visited, predicted_counts=counts.tolist(), selected_slots=len(slots),
        truth_count_or_slots_read=False, original_role_capacity_changed=False,
    )


def select_neural_slots(model, source_features, *, chunk_size=1024):
    """Only the pure-source reader and model are runtime arguments."""
    import torch
    from .contract_runtime_heads_284 import discrete_cardinalities

    if type(chunk_size) is not int or chunk_size < 1 or model.training:
        raise ValueError("Explicit chunk size and evaluation mode required")
    capacity = source_features.anchor_count
    context = torch.from_numpy(np.array(source_features.arrays["whole_source_context.npy"], dtype=np.float32, copy=True))
    raw_context = torch.from_numpy(np.array(source_features.raw_triangles.context, dtype=np.float32, copy=True))
    with torch.no_grad():
        predicted = model.predict_source_role_counts(context, raw_context, capacity)
        counts = discrete_cardinalities(predicted, capacity).cpu().numpy()

        def blocks():
            for first in range(0, capacity, chunk_size):
                rows = np.arange(first, min(first + chunk_size, capacity), dtype=np.int64)
                output = model(source_features.batch(rows))
                yield first, output["presence_logits"].detach().cpu().numpy()

        slots, receipt = select_full_source_slots(blocks(), capacity, counts)
    receipt.update(model_role_counts_only=True, original_pcd_sha256_available_in_metadata_only=True,
                   selected_gt_candidate_reads=0, STEP_runtime_reads=0, decoder_complete=False)
    return slots, receipt
