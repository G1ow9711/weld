"""Read-only provenance audit of the six unresolved finite-domain pairs.

No fitting, candidate mutation or tolerance changes. The old mesh was not saved
by 950, so a vertex match is explicitly not a complete triangle-domain proof.
Exact rational expressions test whether the identified vertex projection and
convex-hull boundary intersection denote the same parameter on frozen inputs.
"""
import gc
import time
from fractions import Fraction
import numpy as np
from scipy.spatial import ConvexHull
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest, check_resources
from weld_pipeline.sjg_finite_envelope_carriers_941 import clip_line, complement


def rational_vector(a):
    return [Fraction.from_float(float(v)) for v in a]


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def sub(a, b):
    return [x-y for x, y in zip(a, b)]


def cross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def exact_parameters(vertex, edge, face, origin, direction):
    o, d, v = map(rational_vector, (origin, direction, vertex))
    c = rational_vector(face['center'])
    basis = np.asarray(face['basis'])
    axes = [rational_vector(basis[:, i]) for i in range(2)]
    project = lambda p: [dot(sub(p, c), axis) for axis in axes]
    a, b = [project(rational_vector(point)) for point in edge]
    pos = project(o)
    vector = [dot(d, axis) for axis in axes]
    e = sub(b, a)
    denominator = cross2(vector, e)
    if denominator == 0:
        return dict(status='EXACT_PARALLEL_UNPROVED')
    hull_t = cross2(sub(a, pos), e)/denominator
    vertex_t = dot(sub(v, o), d)
    difference = vertex_t-hull_t
    return dict(status='EXACTLY_EQUAL' if difference == 0 else 'DISTINCT_BOUNDARY_EXPRESSIONS',
                delta_parameter_mm=float(difference),
                delta_numerator=str(difference.numerator), delta_denominator=str(difference.denominator),
                arithmetic='rational evaluation of frozen float64 coordinates and plane frame',
                proves_complete_domain_identity=False)


def main():
    start = time.perf_counter()
    dest = OUT/'run310'
    dest.mkdir(exist_ok=False)
    rows = read(OUT/'run309/conflict_provenance.json')['rows']
    cases = {c['source_key']: c for c in read(OUT/'run98/local_cohort.json')['cases']}
    results = []
    write(dest/'registered_protocol.json', dict(new_fit_calls=0, GT_reads=0, STEP_reads=0,
        scope='six conflicting generator pairs only; no full candidate regeneration',
        source_sha256=digest(__file__), input_mutations=0,
        identity_requires_full_domain_and_port_proof=True))
    for key in dict.fromkeys(r['source'] for r in rows):
        check_resources()
        base = OUT/f'run132/source_{key}'
        current = OUT/f'run185/source_{key}'
        faces = read(base/'surfaces.json')
        profiles = read(base/'profiles.json')
        current_profiles = read(current/'profiles.json')
        assert profiles == current_profiles[:len(profiles)]
        assert faces == read(current/'surfaces.json')[:len(faces)]
        assert digest(OUT/cases[key]['input_file']) == cases[key]['input_sha256']
        for folder in (base, current):
            binding = read(folder/'input_freeze.json')
            for name in ('profiles.json', 'surfaces.json', 'surface_support.npz'):
                assert digest(folder/name) == binding['files'][name]
        xyz = np.load(OUT/cases[key]['input_file'], mmap_mode='r')
        cache = {}
        with np.load(base/'surface_support.npz') as z, np.load(current/'surface_support.npz') as current_z:
            for r in (r for r in rows if r['source'] == key):
                p, q = [profiles[r[k]] for k in ('negative_profile', 'positive_profile')]
                ids = p['face_indices']
                assert ids == q['face_indices']
                a, b = [faces[i] for i in ids]
                direction = np.cross(a['normal'], b['normal'])
                direction /= np.linalg.norm(direction)
                if direction[np.argmax(abs(direction))] < 0:
                    direction = -direction
                normals = np.stack([a['normal'], b['normal']])
                origin = normals.T @ np.linalg.solve(normals @ normals.T, [a['offset'], b['offset']])
                ranges, endpoints = [], []
                for fi in ids:
                    f = faces[fi]
                    if fi not in cache:
                        support = z[f['point_ids_key']]
                        assert np.array_equal(support, current_z[f['point_ids_key']])
                        points = np.asarray(xyz[support], dtype=np.float64)
                        uv = (points-f['center']) @ np.asarray(f['basis'])
                        cache[fi] = (support, points, ConvexHull(uv))
                    support, points, hull = cache[fi]
                    basis = np.asarray(f['basis'])
                    pos, vec = (origin-f['center']) @ basis, direction @ basis
                    aa = hull.equations[:, :2] @ vec
                    bb = hull.equations[:, :2] @ pos+hull.equations[:, 2]
                    lo_ids, hi_ids = np.flatnonzero(aa < -1e-12), np.flatnonzero(aa > 1e-12)
                    lo_idx = int(lo_ids[np.argmax(-bb[lo_ids]/aa[lo_ids])])
                    hi_idx = int(hi_ids[np.argmin(-bb[hi_ids]/aa[hi_ids])])
                    interval = clip_line(hull.equations, pos, vec)
                    assert interval is not None
                    ranges.append(interval)
                    endpoints.append((lo_idx, hi_idx))
                lo_face = int(np.argmax([x[0] for x in ranges]))
                hi_face = int(np.argmin([x[1] for x in ranges]))
                lo, hi = ranges[lo_face][0], ranges[hi_face][1]
                envelope = np.stack([origin+lo*direction, origin+hi*direction])
                saved_envelope = np.asarray(p['representation']['endpoints_mm'])
                saved_global = np.asarray(q['representation']['endpoints_mm'])
                provenance = []
                for endpoint, active in enumerate((lo_face, hi_face)):
                    fi = ids[active]
                    f = faces[fi]
                    support, points, hull = cache[fi]
                    # Locate potential global boundary source, preserving uncertainty
                    # about omitted triangle connectivity in the old export.
                    param = (points-origin) @ direction
                    global_t = float((saved_global[endpoint]-origin) @ direction)
                    nearest = int(np.argmin(abs(param-global_t)))
                    edge_indices = hull.simplices[endpoints[active][endpoint]]
                    edge = points[edge_indices]
                    exact = exact_parameters(points[nearest], edge, f, origin, direction)
                    provenance.append(dict(endpoint=endpoint, active_hull_face=fi,
                        hull_edge_point_ids=support[edge_indices].tolist(),
                        nearest_projected_vertex_point_id=int(support[nearest]),
                        nearest_projection_difference_mm=float(abs(param[nearest]-global_t)),
                        vertex_signed_strip_distance_mm=float((points[nearest]-origin) @ np.cross(f['normal'], direction)),
                        mesh_triangle_connectivity_saved=False, vertex_is_proven_mesh_boundary=False,
                        exact_boundary_expression=exact))
                length = hi-lo
                shift = float((saved_global[0]-origin) @ direction-lo)
                intervals = [[max(0., left+shift), min(length, right+shift)] for left, right in q['support_intervals_mm']]
                intervals = [v for v in intervals if v[1] > v[0]]
                gaps = complement(intervals, length)
                replay_support = np.asarray(intervals)
                old_support = np.asarray(p['support_intervals_mm'])
                row = dict(source=key, negative_profile=r['negative_profile'], positive_profile=r['positive_profile'],
                    ordered_faces=ids, status='ALIAS_UNPROVED_DIFFERENT_FINITE_CONSTRAINT_RULES',
                    frozen_input_sha256=cases[key]['input_sha256'],
                    support_sha256=digest(base/'surface_support.npz'),
                    frozen_plane_parameters=[{k:f[k] for k in ('normal','offset','center','basis')} for f in (a,b)],
                    original_candidates_unchanged=True, complete_domain_identity_proved=False,
                    global_domain_rule='union of retained triangle cross sections in +/-0.2 mm strip; interval merging 1e-5 mm',
                    envelope_domain_rule='exact line through convex hull of all observed support projected to plane frame',
                    support_port_rule='global intervals shifted and clipped to envelope; complement kept as unobserved gap',
                    replay_max_envelope_endpoint_difference_mm=float(abs(envelope-saved_envelope).max()),
                    replay_support_shape_equal=replay_support.shape == old_support.shape,
                    replay_max_support_port_difference_mm=float(abs(replay_support-old_support).max()) if replay_support.shape == old_support.shape and replay_support.size else None,
                    replay_gap_count=len(gaps), original_gap_count=len(p['unobserved_gaps_mm']),
                    endpoint_sources=provenance)
                results.append(row)
                write(dest/'partial_results.json', results)
                print('FINITE_SOURCE_PROOF', key, r['negative_profile'], r['positive_profile'], row['status'],
                      'REPLAY_MAX_MM', row['replay_max_envelope_endpoint_difference_mm'], flush=True)
        del cache, xyz, faces, profiles, current_profiles
        gc.collect()
    write(dest/'receipt.json', dict(status='FINITE_ALIAS_IDENTITY_NOT_PROVED_NO_FIT', pairs=results,
        proved_pairs=0, unproved_pairs=len(results), remaining_conflicting_rows=205,
        new_fit_calls=0, GT_reads=0, STEP_reads=0, input_mutations=0,
        extraction_correct_rate=None, extraction_precision_mm=None,
        seconds=time.perf_counter()-start))
    print('FINITE_SOURCE_AUDIT_COMPLETE', len(results), time.perf_counter()-start, flush=True)


if __name__ == '__main__':
    main()
