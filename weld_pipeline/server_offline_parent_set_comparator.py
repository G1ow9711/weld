"""TRAIN-only feasible comparison set; not an inference-time oracle."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .server_offline_parent_set_reference import evaluate_reference_subset


def reference_comparator(table, parent_children, model_unary):
    counts = table['child_counts']; reference_count = table['tp'].shape[1]
    if not reference_count or not len(counts):
        return [], evaluate_reference_subset(table, [])
    weight = 2*table['tp'] - counts[:, None]
    preference = np.asarray(model_unary)/(1+abs(np.asarray(model_unary)))
    # Integer reference utility is primary; model score breaks its ties only.
    cost = np.where(weight > 0, -weight-preference[:, None]*1e-6/(reference_count+1), 1e12)
    augmented = np.concatenate([cost.T, np.zeros((reference_count, reference_count))], axis=1)
    refs, candidates = linear_sum_assignment(augmented)
    matches = [(int(parent), int(reference)) for reference, parent in zip(refs, candidates)
               if parent < len(counts) and weight[parent, reference] > 0]
    matches.sort(key=lambda x: (-weight[x[0], x[1]], -model_unary[x[0]], x[0]))
    selected = []; used = set()
    for candidate, _ in matches:
        children = parent_children[candidate]
        if len(set(children)) != len(children) or used.intersection(children):
            continue
        selected.append(candidate); used.update(children)
    # Hungarian utility above is only a proposal. Original D3 costs decide its
    # actual quality; neither global optimality nor identical assignment is assumed.
    return selected, evaluate_reference_subset(table, selected)
