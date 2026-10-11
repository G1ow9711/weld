"""Finite source-only surface anchors and an explicit untrained output space.

This is an input/decoder registry, not a weld detector or a learned locator.
Local regions consist of original accepted triangles. Material states remain
unknown. Ground truth is never used to create an anchor or increase capacity.
"""
from __future__ import annotations

from hashlib import sha256
import json
import sqlite3
import struct

import numpy as np

VERSION = 'FINITE_LOCAL_SOURCE_ANCHOR_DECODER_OUTPUT_SLOTS_247_v1'
CAPACITIES = {'parent': 1, 'boundary': 2, 'geometry_span': 1, 'geometry_node': 1, 'child': 1}


def decoder_spec():
    return dict(
        version=VERSION, status='FROZEN_UNTRAINED_OUTPUT_SPACE_NOT_EXPRESSION_OR_MODEL_PASS',
        anchor_sources=['ACCEPTED_ORIGINAL_REPRESENTATIVE_ROW', 'ACCEPTED_TRIANGLE_CENTROID'],
        per_anchor_slot_capacities=dict(CAPACITIES),
        total_capacity_rule='PER_ANCHOR_CAPACITY_TIMES_CANONICAL_SOURCE_ANCHOR_COUNT',
        max_children_per_parent=None,
        parent_child_count_rule='SELECT_FINITE_GLOBAL_CHILD_SLOTS_WITH_CONTINUE_END_OUTPUTS',
        slot_identity='ANCHOR_ID_ROLE_LOCAL_ORDINAL',
        duplicate_serialization_adds_capacity=False,
        approximate_coordinate_clustering_for_identity=False,
        local_position_permission='UNION_OF_ORIGINAL_TRIANGLES_EXPLICITLY_LINKED_TO_ANCHOR_AND_FACE',
        initial_feature_input='ORIGINAL_PCD_LOCAL_ANCHOR_DOMAIN_AND_WHOLE_SOURCE_CONTEXT',
        reference_path_features_for_initial_location_forbidden=True,
        predicted_path_refinement_input='FEATURES_ON_ACTUAL_PREDICTED_GEOMETRY_ONLY',
        independent_initial_and_refinement_feature_normalizers=True,
        domain_types=['FINITE_SAMPLED_FACE_UNION', 'ORDERED_FINITE_CURVE_CARRIER'],
        current_manifest_domain_binding='FINITE_SAMPLED_FACE_UNION_ONLY',
        current_inter_domain_transition_permissions='SAME_FACE_ONLY',
        additional_1D_carrier_and_cross_domain_binding_status='NOT_BOUND_PENDING_COMPLETE_PROVIDER',
        parent_slot=dict(start_boundary_region='OWN_ANCHOR_LOCAL_REGION', existence='PREDICTED',
                         child_slots='FINITE_GLOBAL_REFERENCES', direction='EXPLICIT',
                         periodic_phase='EXPLICIT_CONTINUOUS_PARAMETER', winding='EXPLICIT_SIGNED_TRAVERSAL'),
        boundary_slot=dict(position='CONTINUOUS_WITHIN_OWN_ANCHOR_REGION_IN_SELECTED_PERMITTED_DOMAIN',
                           logical_roles=['START', 'END', 'CUT', 'GAP_LEFT', 'GAP_RIGHT'],
                           prediction_identity='ONE_COORDINATE_PER_BOUNDARY_SLOT'),
        geometry_span=dict(allowed_types=['line', 'arc', 'polyline'],
                           endpoints='REFERENCES_TO_BOUNDARY_SLOTS',
                           parameter_frame='DECLARED_DOMAIN_UV_OR_ORDERED_CARRIER_PHYSICAL_Q',
                           line='CONTINUOUS_START_END_BOUNDARY_PARAMETERS',
                           arc=dict(center_uv='FINITE_CONTINUOUS_MAY_BE_OUTSIDE_SUPPORT_DOMAIN',
                                    radius='FINITE_STRICTLY_POSITIVE', phase='EXPLICIT',
                                    sweep='NONZERO_SIGNED_CONTINUOUS_WITH_EXPLICIT_PARTIAL_FULL_TURN',
                                    winding_limit='FINITE_GLOBAL_GEOMETRY_SPAN_CAPACITY'),
                           polyline='ORDERED_REFERENCES_TO_FINITE_GEOMETRY_NODE_SLOTS'),
        child=dict(parent='EXACT_PARENT_SLOT_REFERENCE', order='EXPLICIT_ORDERED_INDEX',
                   geometry='ORDERED_ONE_OR_MORE_GEOMETRY_SPANS',
                   geometry_span_corners_are_not_business_cuts=True),
        parent=dict(children='ORDERED_CHILD_SLOT_REFERENCES',
                    shared_cut='SAME_BOUNDARY_SLOT_AND_PREDICTED_COORDINATE',
                    positive_gap='DISTINCT_BOUNDARY_SLOTS_AND_STRICTLY_POSITIVE_FORWARD_INTERVAL',
                    full_turn='DISTINCT_LOGICAL_START_END_WITH_EXPLICIT_SHARED_PHYSICAL_ANCHOR',
                    coordinate_equality_does_not_infer_closure=True),
        geometry_validation='CONCRETE_CURVE_BIDIRECTIONAL_CONTINUOUS_3MM_AND_OWN_DOMAIN_COVERAGE',
        separate_gates=['SURFACE_UNION_SUPPORT', 'JOINT_GEOMETRY_EXPRESSION', 'SEMANTIC_IDENTIFIABILITY', 'REAL_TRAINABILITY', 'FULL_CONTRACT'],
        actual_trained_locator=False, actual_joint_expression_verified=False, full_contract_pass=False)


def canonical_plane_key(normal, offset):
    values = np.asarray([*normal, offset], dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Finite plane equation required')
    if abs(float(values[:3] @ values[:3]) - 1.) > 2e-9:
        raise ValueError('Numerical unit computational normal required')
    first = next((x for x in values[:3] if x != 0), None)
    if first is None:
        raise ValueError('Nonzero numerical plane normal required')
    if first < 0:
        values *= -1
    values[values == 0] = 0.
    return struct.pack('<4d', *values)


def anchor_key(kind, raw_ids, plane_key=None):
    ids = tuple(sorted(map(int, raw_ids)))
    if any(i < 0 for i in ids):
        raise ValueError('Original nonnegative representative row IDs required')
    if kind == 'RAW_REPRESENTATIVE' and len(ids) == 1:
        return b'R' + struct.pack('<q', ids[0])
    if kind == 'TRIANGLE_CENTROID' and len(ids) == 3 and isinstance(plane_key, bytes) and len(plane_key) == 32:
        return b'T' + plane_key + struct.pack('<3q', *ids)
    raise ValueError('Declared original raw or finite triangle anchor identity required')


class SurfaceSlotRegistry:
    def __init__(self, connection, raw_xyz_mm):
        self.db = connection
        self.raw = np.asarray(raw_xyz_mm)
        if self.raw.ndim != 2 or self.raw.shape[1] != 3 or not np.isfinite(self.raw).all():
            raise ValueError('Complete original finite XYZ array required')
        self.db.executescript('''
          CREATE TABLE domains(face_id INTEGER PRIMARY KEY, descriptor_json TEXT NOT NULL, plane_key BLOB NOT NULL);
          CREATE TABLE anchors(anchor_id INTEGER PRIMARY KEY, canonical_key BLOB NOT NULL UNIQUE,
            kind TEXT NOT NULL, seed_x REAL NOT NULL, seed_y REAL NOT NULL, seed_z REAL NOT NULL,
            raw_a INTEGER NOT NULL, raw_b INTEGER, raw_c INTEGER, plane_key BLOB);
          CREATE TABLE triangles(face_id INTEGER NOT NULL, triangle_ordinal INTEGER NOT NULL,
            raw_a INTEGER NOT NULL, raw_b INTEGER NOT NULL, raw_c INTEGER NOT NULL,
            centroid_anchor_id INTEGER NOT NULL, PRIMARY KEY(face_id,triangle_ordinal));
          CREATE TABLE regions(anchor_id INTEGER NOT NULL, face_id INTEGER NOT NULL,
            triangle_ordinal INTEGER NOT NULL, source_role INTEGER NOT NULL,
            PRIMARY KEY(anchor_id,face_id,triangle_ordinal,source_role)) WITHOUT ROWID;
        ''')
        self.counts = dict(domains=0, original_triangles=0, original_triangle_corners=0,
                           region_memberships=0, canonical_anchors=0,
                           canonical_raw_anchors=0, canonical_triangle_anchors=0,
                           triangle_alias_reuses=0)

    def _anchor(self, kind, ids, plane_key=None):
        canonical = anchor_key(kind, ids, plane_key)
        existing = self.db.execute('SELECT anchor_id FROM anchors WHERE canonical_key=?', (canonical,)).fetchone()
        if existing:
            if kind == 'TRIANGLE_CENTROID':
                self.counts['triangle_alias_reuses'] += 1
            return existing[0]
        sorted_ids = tuple(sorted(map(int, ids)))
        seed = self.raw[list(sorted_ids)].mean(axis=0)
        padded = list(sorted_ids) + [None] * (3 - len(sorted_ids))
        cursor = self.db.execute('''INSERT INTO anchors(canonical_key,kind,seed_x,seed_y,seed_z,
                     raw_a,raw_b,raw_c,plane_key) VALUES(?,?,?,?,?,?,?,?,?)''',
                     (canonical, kind, *map(float, seed), *padded, plane_key))
        self.counts['canonical_anchors'] += 1
        self.counts['canonical_raw_anchors' if kind == 'RAW_REPRESENTATIVE' else 'canonical_triangle_anchors'] += 1
        return cursor.lastrowid

    def add_face(self, face, representative_ids, triangle_rows):
        face_id = int(face['face_id'])
        ids = np.asarray(representative_ids)
        triangles = np.asarray(triangle_rows)
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
                or (ids < 0).any() or (ids >= len(self.raw)).any()
                or triangles.ndim != 2 or triangles.shape[1] != 3
                or not np.issubdtype(triangles.dtype, np.integer)
                or (triangles < 0).any() or (triangles >= len(ids)).any()):
            raise ValueError('Complete original accepted triangle/representative rows required')
        plane_key = canonical_plane_key(face['normal'], face['offset_mm'])
        descriptor = json.dumps(face, sort_keys=True, separators=(',', ':'), allow_nan=False)
        self.db.execute('INSERT INTO domains VALUES(?,?,?)', (face_id, descriptor, plane_key))
        raw_lookup = {int(i): self._anchor('RAW_REPRESENTATIVE', [i]) for i in np.unique(ids[triangles])}
        memberships = []
        triangle_records = []
        for ordinal, original in enumerate(ids[triangles]):
            original = tuple(map(int, original))
            centroid = self._anchor('TRIANGLE_CENTROID', original, plane_key)
            triangle_records.append((face_id, ordinal, *original, centroid))
            memberships.extend((raw_lookup[raw_id], face_id, ordinal, corner)
                               for corner, raw_id in enumerate(original))
            memberships.append((centroid, face_id, ordinal, 3))
            if len(triangle_records) >= 4096:
                self.db.executemany('INSERT INTO triangles VALUES(?,?,?,?,?,?)', triangle_records)
                self.db.executemany('INSERT INTO regions VALUES(?,?,?,?)', memberships)
                triangle_records.clear(); memberships.clear()
        self.db.executemany('INSERT INTO triangles VALUES(?,?,?,?,?,?)', triangle_records)
        self.db.executemany('INSERT INTO regions VALUES(?,?,?,?)', memberships)
        self.counts['domains'] += 1
        self.counts['original_triangles'] += len(triangles)
        self.counts['original_triangle_corners'] += 3 * len(triangles)
        self.counts['region_memberships'] += 4 * len(triangles)
        return dict(self.counts)

    def finish(self):
        self.db.executescript('''
          CREATE INDEX regions_by_face_triangle ON regions(face_id,triangle_ordinal,source_role);
          CREATE INDEX triangle_centroid_aliases ON triangles(centroid_anchor_id);
        ''')
        self.db.commit()
        names = {'domains': 'domains', 'original_triangles': 'triangles',
                 'region_memberships': 'regions', 'canonical_anchors': 'anchors'}
        for count, table in names.items():
            assert self.db.execute(f'SELECT count(*) FROM {table}').fetchone()[0] == self.counts[count]
        assert self.counts['canonical_raw_anchors'] + self.counts['canonical_triangle_anchors'] == self.counts['canonical_anchors']
        assert self.counts['canonical_triangle_anchors'] + self.counts['triangle_alias_reuses'] == self.counts['original_triangles']
        return {**self.counts, 'slot_capacities': {role: count * self.counts['canonical_anchors'] for role, count in CAPACITIES.items()}}


def validate_slot_capacity(connection, assignments):
    """Independent role capacities, permitting explicit shared boundary refs.

    Duplicate defining outputs or out-of-manifest slots are invalid. A caller
    stores each shared coordinate once and then references it from children.
    This check does not replace parent-order or continuous-geometry checking.
    """
    seen = set()
    for assignment in assignments:
        anchor = assignment['anchor_id']
        role = assignment['role']
        ordinal = assignment['local_slot_ordinal']
        if (type(anchor) is not int or role not in CAPACITIES or type(ordinal) is not int
                or not 0 <= ordinal < CAPACITIES[role]
                or not connection.execute('SELECT 1 FROM anchors WHERE anchor_id=?', (anchor,)).fetchone()):
            raise ValueError('CAPACITY_DEFICIT_OR_SLOT_NOT_IN_FROZEN_MANIFEST')
        identity = (anchor, role, ordinal)
        if identity in seen:
            raise ValueError('DUPLICATE_DEFINING_SLOT_OUTPUT')
        seen.add(identity)
    return dict(unique_defining_outputs=len(seen), capacity_verified=True,
                complete_joint_geometry_or_parent_assignment_verified=False)


def local_region_triangles(connection, anchor_id, face_id):
    return [r[0] for r in connection.execute('''SELECT DISTINCT triangle_ordinal
            FROM regions WHERE anchor_id=? AND face_id=? ORDER BY triangle_ordinal''', (anchor_id, face_id))]
