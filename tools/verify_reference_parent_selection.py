"""Decoder controls against a separate dense greedy implementation."""
from pathlib import Path
import numpy as np
from weld_pipeline.server_reference_parent_selection import greedy_nullable_set, set_model_utility, build_competition_graph
from weld_pipeline.server_offline_parent_pair_utility import reference_pair_targets
from weld_pipeline.sjg_local_runtime_904 import write, digest


def main():
    rng = np.random.default_rng(1031)
    for trial in range(50):
        n = trial
        unary = rng.integers(-3, 6, n).astype(float)
        pairs = np.asarray([(i, j) for i in range(n) for j in range(i+1, n)
                            if rng.random() < .2], np.int64).reshape(-1, 2)
        penalty = rng.integers(0, 5, len(pairs)).astype(float)
        children = [[int(rng.integers(0, max(1, n)))] for _ in range(n)]
        actual = greedy_nullable_set(unary, pairs, penalty, children)
        matrix = np.zeros((n, n)); matrix[tuple(pairs.T)] = penalty
        matrix += matrix.T
        selected = []; taken = set(); history = []
        while True:
            gain = unary - (matrix[:, selected].sum(1) if selected else 0.)
            for i in range(n):
                if i in selected or taken.intersection(children[i]):
                    gain[i] = -np.inf
            if not n or gain.max() <= 0:
                break
            index = int(gain.argmax()); selected.append(index); taken.update(children[index])
            history.append(set_model_utility(unary, pairs, penalty, selected))
        assert actual == selected
        assert np.all(np.diff([0.] + history) > 0)
    # A duplicate consumes reference utility, but a distinct reference does not.
    table = dict(costs=np.array([[-1e6, 1e12], [-1e6, 1e12], [1e12, -1e6]]),
                 tp=np.array([[1, 0], [1, 0], [0, 1]]), child_counts=np.ones(3, int), invalid_children=np.asarray(0))
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    unary, costs = reference_pair_targets(table, pairs)
    assert np.array_equal(costs, [1, 0, 0])
    assert greedy_nullable_set(unary, pairs, 2*costs, [[0], [1], [2]]) == [0, 2]
    assert greedy_nullable_set([-1, 0], [], [], [[0], [1]]) == []
    assert greedy_nullable_set([2, 1], [], [], [[0, 0], [1]]) == [1]
    assert build_competition_graph([], [], [])[1].shape == (0, 29)
    root = Path(__file__).resolve().parents[1]
    out = root/'server_rebuild/expansion_v1/reference_parent_selection_controls_v1'
    out.mkdir(exist_ok=False)
    write(out/'receipt.json', dict(status='PASS', dense_decoder_controls=50,
          duplicate_reference_cost=True, null_output=True, repeated_child_rejected=True,
          modules={name: digest(root/'weld_pipeline'/f'{name}.py') for name in
                   ['server_reference_parent_selection', 'server_offline_parent_pair_utility']}, fit_calls=0))
    print('REFERENCE_PARENT_SELECTION_CONTROLS_PASS', flush=True)


if __name__ == '__main__':
    main()
