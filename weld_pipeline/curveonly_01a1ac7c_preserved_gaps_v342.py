"""Retain the frozen model's existing gaps at verified transverse side faces.

The 35mm customer threshold is unchanged. Clearance above35 alone does not
cancel another existing process gap. Both endpoints must agree with the same
finite transverse plate's side-face pair on the actual free welding side.
"""
import types
from . import curveonly_01a1ac7c_owner_rules_v340 as previous

prior, provenance = previous.prior, previous.provenance


def apply(raw, features, packet, inventory, solids):
    old, _ = provenance.select(raw, packet)
    base = previous.previous

    def crossings(ci, packet, inventory, geometry):
        blocks, review = base.crossings(ci, packet, inventory, geometry)
        if not blocks:
            return blocks, review
        ranges = []
        for p in old:
            q = previous.coincident_parameters(p['points'], packet['candidates'][ci]['points'])
            if q is not None:
                s, error = q
                ranges.append((float(s.min()), float(s.max())))
        intervals = base.union(ranges)
        for record in review:
            if record['measured_two_board_passage_mm'] is None:
                continue
            a, b = record['start_mm'], record['end_mm']
            if any(abs(left[1]-a) <= 3. and abs(right[0]-b) <= 3.
                   for left, right in zip(intervals[:-1], intervals[1:])):
                if not any(abs(q['start_mm']-a) <= base.NUMERIC and
                           abs(q['end_mm']-b) <= base.NUMERIC for q in blocks):
                    blocks.append(dict(record, decision='PRESERVED_MODEL_GAP_AT_VERIFIED_TRANSVERSE_SIDE_PAIR',
                        customer35_triggered=record['measured_two_board_passage_mm'] <= base.PASSAGE_MM,
                        no_new35_permission_or_threshold_relaxation=True))
        return blocks, review

    proxy = types.SimpleNamespace(**dict(base.__dict__, crossings=crossings))
    namespace = dict(previous.__dict__, previous=proxy)
    result, audit = types.FunctionType(previous.apply.__code__, namespace)(
        raw, features, packet, inventory, solids)
    audit.update(branch='V342', existing_structural_model_gaps_preserved=True,
        customer35_limit_unchanged=True)
    return result, audit
