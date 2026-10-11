"""Mine high-response negatives from a frozen TRAIN prediction only."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest
from weld_pipeline.sjg_probability_corrected_queries_1007 import labels_for


def extend_layout(layout, profiles, supervision, prediction_run, source_key, destination):
    cohort = {c['source_key']:c for c in read(OUT/'run98/local_cohort.json')['cases']}
    assert cohort[source_key]['role']=='TRAIN' and source_key not in ('64','24')
    folder = OUT/prediction_run/('source_'+source_key)
    binding = read(folder/'input_output_freeze.json')
    assert binding['input_sha256']==cohort[source_key]['input_sha256']
    for name in ('probability.npy','q_mm.npy','profile_offsets.npy'):
        assert digest(folder/name)==binding['files'][name]
    ptr = np.load(folder/'profile_offsets.npy', allow_pickle=False)
    assert np.array_equal(ptr, layout['ptr'])
    probabilities = np.load(folder/'probability.npy', mmap_mode='r', allow_pickle=False)
    q = np.load(folder/'q_mm.npy', mmap_mode='r', allow_pickle=False)
    parts = [[],[],[]]
    for node in range(len(profiles)):
        start, end = int(ptr[node]), int(ptr[node+1])
        high = probabilities[start:end,:3] >= .5
        if not high.any():continue
        indices = np.flatnonzero(high.any(1))
        labels = labels_for(profiles, supervision, np.full(len(indices),node), q[start+indices])
        for head in range(3):
            selected = high[indices,head] & labels['target_mask'][:,head] & (labels['target'][:,head]<.1)
            if selected.any():parts[head].append(start+indices[selected])
    counts = []
    arrays = {}
    for head, chunks in enumerate(parts):
        indices = np.unique(np.concatenate(chunks)) if chunks else np.empty(0,np.int64)
        counts.append(len(indices));arrays['head_'+str(head)] = indices
        if len(indices):
            layout['event_groups'].append(indices)
            layout['event_keys'].append(('mined_TRAIN_negative',head))
    assert sum(counts)>0
    path = destination/'frozen_TRAIN_hard_query_indices.npz'
    np.savez_compressed(path, **arrays)
    write(destination/'hard_query_mining.json', dict(source_key=source_key, prediction_run=prediction_run,
        frozen_probability_sha256=binding['files']['probability.npy'], hard_query_counts_by_head=counts,
        response_threshold=.5, TRAIN_soft_target_less_than=.1, sample_file=path.name, sample_sha256=digest(path),
        overlap_probabilities_accounted_by_sampler_1010=True, full_background_probability_nonzero=True,
        extraction_threshold_unchanged=True, no_heldout_target_read=True, optimizer_updates=0))
    return layout
