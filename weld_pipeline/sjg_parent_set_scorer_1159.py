"""Small NumPy parent-set scorer. No labels, GT matching, geometry edits or NMS.

Inputs must use frozen TRAIN-only normalization. Relations describe candidates
against the currently selected set. Output is one logit per candidate and STOP.
"""
from pathlib import Path
import numpy as np

SCHEMA = 'parent-set-mlp-194-32-1-stop-322-v1'
SHAPES = {'W1': (194, 32), 'b1': (32,), 'W2': (32,), 'b2': (1,),
          'Ws': (322,), 'bs': (1,)}


class ParentSetScorer:
    def __init__(self, seed=1031):
        rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.parameters = dict(W1=rng.normal(0., np.sqrt(2./226), SHAPES['W1']),
            b1=np.zeros(32), W2=rng.normal(0., np.sqrt(2./33), 32), b2=np.zeros(1),
            Ws=rng.normal(0., 1./np.sqrt(322), 322), bs=np.zeros(1))
        self.initialization_rng_state = rng.bit_generator.state

    def forward(self, unary, relations, selected):
        unary = np.asarray(unary, dtype=np.float64)
        relations = np.asarray(relations, dtype=np.float64)
        if unary.ndim != 2 or unary.shape[1] != 160 or relations.shape != (len(unary), 32):
            raise ValueError('Expected normalized unary[N,160] and relation[N,32]')
        if not np.isfinite(unary).all() or not np.isfinite(relations).all():
            raise ValueError('Scorer requires finite features')
        selected = tuple(selected)
        if len(set(selected)) != len(selected) or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(unary) for i in selected):
            raise ValueError('Selected indices must be unique existing candidates')
        if not selected and np.any(relations != 0):
            raise ValueError('Empty-set relations must be zero')
        state = np.array([np.log1p(len(selected)), float(not selected)])
        inputs = np.empty((len(unary), 194), dtype=np.float64)
        inputs[:, :160] = unary
        inputs[:, 160:192] = relations
        inputs[:, 192:] = state
        all_mean = unary.mean(axis=0) if len(unary) else np.zeros(160)
        selected_mean = unary[list(selected)].mean(axis=0) if selected else np.zeros(160)
        stop_input = np.concatenate((all_mean, selected_mean, state))
        p = self.parameters
        hidden = np.tanh(inputs @ p['W1'] + p['b1'])
        logits = np.r_[hidden @ p['W2'] + p['b2'][0], stop_input @ p['Ws'] + p['bs'][0]]
        if not np.isfinite(logits).all():
            raise FloatingPointError('Nonfinite model output')
        return logits, (inputs, hidden, stop_input)

    def backward(self, cache, score_gradient):
        inputs, hidden, stop_input = cache
        delta = np.asarray(score_gradient, dtype=np.float64)
        if delta.shape != (len(inputs)+1,) or not np.isfinite(delta).all():
            raise ValueError('Score gradient shape or finiteness invalid')
        child_delta, stop_delta = delta[:-1], delta[-1]
        hidden_delta = (child_delta[:, None] * self.parameters['W2']) * (1.-hidden*hidden)
        return dict(W1=inputs.T @ hidden_delta, b1=hidden_delta.sum(axis=0),
            W2=hidden.T @ child_delta, b2=np.array([child_delta.sum()]),
            Ws=stop_input*stop_delta, bs=np.array([stop_delta]))

    @staticmethod
    def choose_next(logits, selected):
        """Label-free action: only the already selected exact index is masked."""
        values = np.asarray(logits, dtype=np.float64).copy()
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError('Expected finite candidate logits followed by STOP')
        indices = tuple(selected)
        if len(set(indices)) != len(indices) or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(values)-1 for i in indices):
            raise ValueError('Invalid selected candidate indices')
        if indices:
            values[list(indices)] = -np.inf
        # Stable candidate order, then STOP, breaks exact ties without GT.
        action = int(np.argmax(values))
        return None if action == len(values)-1 else action

    def save(self, path):
        from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
        arrays = {name: value for name, value in self.parameters.items()}
        arrays.update(schema=np.array(SCHEMA), seed=np.array(self.seed, dtype=np.int64))
        atomic_binary_write(Path(path), lambda stream: np.savez(stream, **arrays))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(SHAPES) | {'schema', 'seed'} or str(z['schema']) != SCHEMA:
                raise ValueError('Unrecognized scorer schema')
            model = cls(int(z['seed']))
            for name, shape in SHAPES.items():
                array = z[name]
                if array.dtype != np.float64 or array.shape != shape or not np.isfinite(array).all():
                    raise ValueError('Invalid stored parameter: ' + name)
                model.parameters[name] = array.copy()
        return model


class AdamState:
    """One update after per-piece prefix-mean gradients; no hidden extra steps."""
    def __init__(self, model):
        self.step_count = 0
        self.first = {k: np.zeros_like(v) for k, v in model.parameters.items()}
        self.second = {k: np.zeros_like(v) for k, v in model.parameters.items()}

    def update(self, model, gradients, lr=1e-3, weight_decay=1e-4, clip=5.):
        if set(gradients) != set(SHAPES):
            raise ValueError('All parameter gradients are required')
        if not (np.isfinite([lr, weight_decay, clip]).all() and lr > 0 and weight_decay >= 0 and clip > 0):
            raise ValueError('Invalid optimizer settings')
        for k, v in gradients.items():
            if v.shape != SHAPES[k] or not np.isfinite(v).all():
                raise ValueError('Invalid parameter gradient: ' + k)
        norm = float(np.sqrt(sum(np.sum(v*v) for v in gradients.values())))
        scale = min(1., clip/max(norm, np.finfo(float).tiny))
        self.step_count += 1
        for k, p in model.parameters.items():
            # Adam with coupled L2, matching clipping before optimizer decay.
            g = gradients[k]*scale + weight_decay*p
            self.first[k] = .9*self.first[k] + .1*g
            self.second[k] = .999*self.second[k] + .001*g*g
            first = self.first[k]/(1.-.9**self.step_count)
            second = self.second[k]/(1.-.999**self.step_count)
            p -= lr*first/(np.sqrt(second)+1e-8)
            if not np.isfinite(p).all():
                raise FloatingPointError('Nonfinite optimizer output')
        return dict(step=self.step_count, raw_gradient_norm=norm, clip_scale=scale)
