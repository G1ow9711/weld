"""Input source containment context for the finite-carrier interval model.

These four scalars describe a candidate representation, not whether it should
be welded. No label, prediction, workpiece identity or new geometry query is
used. The original 185 pooled observations remain an exact prefix.
"""
import types
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_set_v18 as base
from .curveonly_01a1ac7c_composite_v48 import ENDPOINT_MM

NAMES = ('has_larger_closed_source_parent', 'has_circular_closed_source_parent',
         'minimum_fraction_of_closed_source_parent', 'is_composite_source_path')


def candidate_context(packet):
    candidates = packet['candidates']
    lengths = [float(cad.arclength(c['points'])[-1]) for c in candidates]
    descendants = {}

    def children(ci, stack=()):
        if ci in stack:
            raise ValueError('Cyclic finite source provenance')
        if ci in descendants:
            return descendants[ci]
        result = set()
        for member in candidates[ci].get('source_material_path', []):
            child = int(member['curve'])
            if not 0 <= child < len(candidates):
                raise ValueError('Finite source child outside input packet')
            span = float(member['end_mm'])-float(member['start_mm'])
            if abs(span-lengths[child]) > 1e-7*max(1., lengths[child]):
                raise ValueError('Finite source length mapping mismatch')
            result.add(child)
            result.update(children(child, stack+(ci,)))
        descendants[ci] = result
        return result

    # Validate all source paths, including paths which are not closed parents.
    for ci in range(len(candidates)):
        children(ci)
    out = np.zeros((len(candidates), len(NAMES)), dtype=np.float32)
    fractions = [[] for _ in candidates]
    for ci, candidate in enumerate(candidates):
        path = candidate.get('source_material_path')
        out[ci, 3] = bool(path)
        points = np.asarray(candidate['points'])
        if not (path and candidate.get('STEP') and candidate.get('pose')
                and np.linalg.norm(points[0]-points[-1]) <= 1e-6):
            continue
        for child in descendants[ci]:
            if lengths[ci] <= lengths[child]+ENDPOINT_MM:
                continue
            out[child, 0] = 1.
            out[child, 1] = max(out[child, 1], float(candidate.get('circle') is not None))
            fractions[child].append(lengths[child]/lengths[ci])
    for ci, values in enumerate(fractions):
        if values:
            out[ci, 2] = min(values)
    assert np.isfinite(out).all()
    return out


def pooled(features, context):
    return np.r_[base.pooled(features), np.asarray(context, np.float32)].astype(np.float32)


class GraphCurveSet(base.CurveSet):
    def predict(self, features, packet):
        context = candidate_context(packet)
        rows = iter(context)

        def add_context(values):
            return pooled(values, next(rows))

        # Reuse the frozen decoder verbatim, changing only its pooling encoder.
        decode = types.FunctionType(base.CurveSet.predict.__code__,
            dict(base.__dict__, pooled=add_context))
        out = decode(self, features, packet)
        assert next(rows, None) is None, 'Candidate pooling order mismatch'
        if self.last_timing is not None:
            self.last_timing.update(pooled_features=189, query_features=192,
                candidate_context_features=len(NAMES), context_source='INPUT_FINITE_SOURCE_PATHS',
                new_geometric_queries=0)
        return out
