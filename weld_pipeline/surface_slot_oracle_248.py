"""Read-only complete registry oracle; imports no production slot builder."""
import json
import math
import struct

import numpy as np

VERSION = 'INDEPENDENT_ORIGINAL_TRIANGLE_ANCHOR_REGION_CAPACITY_ORACLE_248_v1'
EXPECTED_ROLE_CAPACITIES = {'parent': 1, 'boundary': 2, 'geometry_span': 1, 'geometry_node': 1, 'child': 1}


def exact_plane_identity(face):
    normal = tuple(map(float, face['normal']))
    offset = float(face['offset_mm'])
    if not all(map(math.isfinite, (*normal, offset))) or abs(sum(x*x for x in normal)-1.) > 2e-9:
        raise ValueError('Invalid computational plane equation')
    sign = 1. if next(x for x in normal if x) > 0 else -1.
    coefficients = [sign*x if x else 0. for x in (*normal, offset)]
    return struct.pack('<dddd', *coefficients)


def original_point_key(raw_id):
    return b'R' + int(raw_id).to_bytes(8, byteorder='little', signed=True)


def original_triangle_key(plane, raw_ids):
    return b'T' + plane + b''.join(int(i).to_bytes(8, byteorder='little', signed=True) for i in sorted(raw_ids))


def verify_face(connection, raw, face, original_representatives, original_triangles):
    face_id = int(face['face_id'])
    descriptor, plane = connection.execute('SELECT descriptor_json,plane_key FROM domains WHERE face_id=?', (face_id,)).fetchone()
    if json.loads(descriptor) != face or plane != exact_plane_identity(face):
        raise ValueError('Original domain descriptor or computational plane identity changed')
    original = np.asarray(original_representatives)[np.asarray(original_triangles)]
    rows = connection.execute('''SELECT triangle_ordinal,raw_a,raw_b,raw_c,centroid_anchor_id
             FROM triangles WHERE face_id=? ORDER BY triangle_ordinal''', (face_id,)).fetchall()
    if len(rows) != len(original):
        raise ValueError('Original triangle count differs')
    if rows:
        actual = np.asarray(rows, dtype=np.int64)
        if not np.array_equal(actual[:, 0], np.arange(len(original))) or not np.array_equal(actual[:, 1:4], original):
            raise ValueError('Original triangle or raw-corner order differs')
    memberships = connection.execute('''SELECT r.triangle_ordinal,r.source_role,r.anchor_id,a.canonical_key,
             a.kind,a.raw_a,a.raw_b,a.raw_c,a.plane_key,a.seed_x,a.seed_y,a.seed_z
             FROM regions AS r JOIN anchors AS a ON a.anchor_id=r.anchor_id
             WHERE r.face_id=? ORDER BY r.triangle_ordinal,r.source_role''', (face_id,)).fetchall()
    if len(memberships) != 4 * len(original):
        raise ValueError('Every original triangle must have exactly four source-region memberships')
    maximum_seed_error = 0.
    for triangle_ordinal, raw_ids in enumerate(original):
        for role in range(4):
            record = memberships[4*triangle_ordinal + role]
            ordinal, actual_role, anchor_id, key, kind, a, b, c, stored_plane, x, y, z = record
            if ordinal != triangle_ordinal or actual_role != role:
                raise ValueError('Original local role sequence is incomplete or duplicated')
            if role < 3:
                representative = int(raw_ids[role])
                if (key != original_point_key(representative) or kind != 'RAW_REPRESENTATIVE'
                        or a != representative or b is not None or c is not None or stored_plane is not None):
                    raise ValueError('Original raw representative anchor identity differs')
                seed = raw[representative]
            else:
                sorted_ids = sorted(map(int, raw_ids))
                if (key != original_triangle_key(plane, sorted_ids) or kind != 'TRIANGLE_CENTROID'
                        or [a, b, c] != sorted_ids or stored_plane != plane or anchor_id != rows[triangle_ordinal][4]):
                    raise ValueError('Original accepted triangle anchor or alias identity differs')
                seed = np.array([math.fsum(float(raw[i, axis]) for i in sorted_ids)/3. for axis in range(3)])
            error = float(np.max(np.abs(seed - [x, y, z])))
            if error > 1e-8:
                raise ValueError('Source anchor seed does not reconstruct from original points')
            maximum_seed_error = max(maximum_seed_error, error)
    return dict(original_triangles=len(original), original_triangle_corners=3*len(original),
                region_memberships=len(memberships), maximum_seed_reconstruction_error_mm=maximum_seed_error)


def verify_global_capacity(connection, counts, spec, expected_faces, expected_triangles):
    if spec['per_anchor_slot_capacities'] != EXPECTED_ROLE_CAPACITIES:
        raise ValueError('Finite per-anchor capacity configuration changed')
    if (spec['max_children_per_parent'] is not None or spec['duplicate_serialization_adds_capacity']
            or spec['approximate_coordinate_clustering_for_identity']
            or not spec['reference_path_features_for_initial_location_forbidden']):
        raise ValueError('Decoder count, alias or initial-input rule changed')
    tables = {table: connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
              for table in ('domains', 'triangles', 'regions', 'anchors')}
    if tables['domains'] != expected_faces or tables['triangles'] != expected_triangles or tables['regions'] != 4*expected_triangles:
        raise ValueError('Complete source domain/triangle/region census differs')
    if connection.execute('SELECT count(*) FROM anchors a WHERE NOT EXISTS(SELECT 1 FROM regions r WHERE r.anchor_id=a.anchor_id)').fetchone()[0]:
        raise ValueError('Anchor has no original triangle region; cannot increase capacity')
    if connection.execute('SELECT count(*) FROM regions r WHERE NOT EXISTS(SELECT 1 FROM anchors a WHERE a.anchor_id=r.anchor_id)').fetchone()[0]:
        raise ValueError('Region references an unknown anchor')
    if connection.execute('SELECT count(*) FROM regions r WHERE NOT EXISTS(SELECT 1 FROM triangles t WHERE t.face_id=r.face_id AND t.triangle_ordinal=r.triangle_ordinal)').fetchone()[0]:
        raise ValueError('Region references an unknown original triangle')
    unique = connection.execute('SELECT count(DISTINCT canonical_key) FROM anchors').fetchone()[0]
    if unique != tables['anchors']:
        raise ValueError('Duplicate serialized anchor increases capacity')
    kinds = dict(connection.execute('SELECT kind,count(*) FROM anchors GROUP BY kind'))
    if set(kinds) != {'RAW_REPRESENTATIVE', 'TRIANGLE_CENTROID'}:
        raise ValueError('Unregistered anchor source kind')
    wanted = dict(domains=tables['domains'], original_triangles=tables['triangles'],
                  original_triangle_corners=3*tables['triangles'], region_memberships=tables['regions'],
                  canonical_anchors=tables['anchors'], canonical_raw_anchors=kinds['RAW_REPRESENTATIVE'],
                  canonical_triangle_anchors=kinds['TRIANGLE_CENTROID'],
                  triangle_alias_reuses=tables['triangles']-kinds['TRIANGLE_CENTROID'],
                  slot_capacities={role: capacity*tables['anchors'] for role, capacity in EXPECTED_ROLE_CAPACITIES.items()})
    if counts != wanted:
        raise ValueError('Declared complete canonical slot capacities differ from actual rows')
    return wanted
