"""Loss targets for the full source graph. Never imported by runtime input builders."""
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .decoder_output_slots_247 import CAPACITIES

VERSION = 'FULL_SOURCE_TYPED_GRAPH_SUPERVISION_275_v1'
FIELDS = ('coordinates', 'geometry_spans', 'children', 'parents')
SLOT_COLUMNS = (('parent', 0), ('boundary', 0), ('boundary', 1),
                ('geometry_span', 0), ('geometry_node', 0), ('child', 0))
CATEGORY_CLASSES = {'span_kind': ('line', 'arc', 'polyline', 'carrier_slice'),
                    'world_kind': ('line', 'arc', 'polyline', 'compound'),
                    'weld_type': ('horizontal', 'vertical'),
                    'closed': (False, True), 'direction': (-1, 1),
                    'position_kind': ('local_position', 'curve_endpoint', 'physical_reference')}
REF_MARKER = '__source_graph_slot_row_275__'
REF_KEYS = {'anchor_id', 'role', 'local_slot_ordinal'}


def digest(path):
    value = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def identity(ref, anchor_count):
    if set(ref) != REF_KEYS:
        raise ValueError('Exact canonical source slot reference required')
    anchor, role, ordinal = ref['anchor_id'], ref['role'], ref['local_slot_ordinal']
    if type(anchor) is not int or not 1 <= anchor <= anchor_count or role not in CAPACITIES:
        raise ValueError('Source slot outside original frozen pool')
    if type(ordinal) is not int or not 0 <= ordinal < CAPACITIES[role]:
        raise ValueError('Source role local capacity exceeded')
    column = SLOT_COLUMNS.index((role, ordinal))
    return anchor, column, ordinal


def scalar(value):
    if isinstance(value, dict) and set(value) == {'numerator', 'denominator'}:
        return float(Fraction(int(value['numerator']), int(value['denominator'])))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('An explicit finite number or exact rational is required')
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('Nonfinite supervision')
    return value


def flatten_graph(graph, anchor_count, original_mapping, outcomes):
    """Every record/ref remains recoverable, including exact fractions and zero gaps."""
    definitions = []
    for field_index, field in enumerate(FIELDS):
        for original_index, row in enumerate(graph[field]):
            definitions.append((identity(row['slot'], anchor_count), field_index, original_index, row))
    definitions.sort(key=lambda value: value[0])
    keys = [row[0] for row in definitions]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate source role definition adds capacity')
    lookup = {key: index for index, key in enumerate(keys)}
    slot_records = [dict(row[3]['slot']) for row in definitions]
    relationships = []

    def encode(value, owner=None, path=()):
        if isinstance(value, dict) and set(value) == REF_KEYS:
            key = identity(value, anchor_count)
            if key not in lookup:
                raise ValueError('Dangling role reference; do not drop unresolved relations')
            row_index = lookup[key]
            if owner is not None and path != ('slot',):
                relationships.append({'source_row': owner, 'target_row': row_index,
                                      'path': list(path)})
            return {REF_MARKER: row_index}
        if isinstance(value, dict):
            return {key: encode(v, owner, path + (key,)) for key, v in value.items()}
        if isinstance(value, list):
            return [encode(v, owner, path + (index,)) for index, v in enumerate(value)]
        return value

    encoded = {key: value for key, value in graph.items() if key not in FIELDS}
    for field_index, field in enumerate(FIELDS):
        encoded[field] = []
        for index, row in enumerate(graph[field]):
            encoded[field].append(encode(row, lookup[identity(row['slot'], anchor_count)]))
    packed_mapping = encode(original_mapping)
    category_names = tuple(CATEGORY_CLASSES)
    categories = np.full((len(keys), len(category_names)), -1, np.int64)
    numerical = np.zeros((len(keys), 12), np.float64)
    numerical_known = np.zeros_like(numerical, dtype=np.bool_)
    semantics, domain_records = [], []
    domains = np.full(len(keys), -1, np.int64)
    domain_lookup = {}

    def set_category(index, name, value):
        if value not in CATEGORY_CLASSES[name]:
            raise ValueError('Unrepresented actual graph category: ' + name)
        categories[index, category_names.index(name)] = CATEGORY_CLASSES[name].index(value)

    for index, (key, field_index, original_index, record) in enumerate(definitions):
        names, values = [], []
        if 'domain' in record:
            domain_key = canonical(record['domain'])
            if domain_key not in domain_lookup:
                domain_lookup[domain_key] = len(domain_records)
                domain_records.append(record['domain'])
            domains[index] = domain_lookup[domain_key]
        if field_index == 0:
            if 'position' in record:
                set_category(index, 'position_kind', 'local_position')
                position = record['position']
                if 'uv_mm' in position:
                    names = ['uv_u_m', 'uv_v_m']
                    values = [scalar(v) / 1000.0 for v in position['uv_mm']]
                elif 'barycentric' in position:
                    names = ['barycentric_' + str(i) for i in range(3)]
                    values = [scalar(v) for v in position['barycentric']]
                # Other exact position forms are retained without inventing targets.
            elif 'curve_endpoint' in record:
                set_category(index, 'position_kind', 'curve_endpoint')
                names = ['analytic_endpoint_index']
                values = [scalar(record['curve_endpoint']['endpoint_index'])]
            elif 'physical_ref' in record:
                set_category(index, 'position_kind', 'physical_reference')
        elif field_index == 1:
            set_category(index, 'span_kind', record['kind'])
            set_category(index, 'closed', record['is_closed'])
            if record['kind'] == 'arc':
                p = record['parameters']
                names = ['center_u_m', 'center_v_m', 'radius_m', 'u_axis_u', 'u_axis_v',
                         'v_axis_u', 'v_axis_v', 'sweep_rad', 'interval_start', 'interval_end']
                values = ([scalar(v) / 1000 for v in p['center_mm']]
                          + [scalar(p['radius_mm']) / 1000]
                          + [scalar(v) for v in p['u_axis'] + p['v_axis']]
                          + [scalar(p['sweep_rad'])]
                          + [scalar(v) for v in record.get('parameter_interval', [0, 1])])
        elif field_index == 2:
            set_category(index, 'world_kind', record['world_kind'])
            set_category(index, 'weld_type', record['weld_type'])
            names, values = ['original_parent_order_index'], [scalar(record['order_index'])]
        else:
            set_category(index, 'closed', record['is_closed'])
            set_category(index, 'direction', record['direction'])
            names, values = ['turn_count'], [scalar(record['turn_count'])]
            if record['frame']['kind'] == 'line':
                vector = np.array([scalar(v) for v in record['frame']['axis_world']])
                length = np.linalg.norm(vector)
                if not length > 0:
                    raise ValueError('Zero original parent direction axis')
                names += ['parent_axis_x', 'parent_axis_y', 'parent_axis_z']
                values += (vector / length).tolist()
        if len(values) != len(names) or len(values) > 12:
            raise ValueError('Numeric target schema mismatch')
        numerical[index, :len(values)] = values
        numerical_known[index, :len(values)] = True
        semantics.append(names)
    outcome_ids = [row['truth_child_id'] for row in outcomes]
    if len(outcome_ids) != len(set(outcome_ids)):
        raise ValueError('Duplicate original child denominator')
    mapped_ids = [row['truth_child_id'] for row in original_mapping]
    if len(mapped_ids) != len(set(mapped_ids)) or not set(mapped_ids).issubset(outcome_ids):
        raise ValueError('Dropped or duplicate original supervision identities')
    if len(mapped_ids) != len(graph['children']):
        raise ValueError('Each represented original child requires its own graph child')
    return dict(version=VERSION, anchor_count=anchor_count, slot_columns=[list(v) for v in SLOT_COLUMNS],
                slots=np.asarray(keys, dtype=np.int64).reshape(-1, 3),
                definition_locations=np.asarray([[v[1], v[2]] for v in definitions], np.int64).reshape(-1, 2),
                categories=categories, numerical_targets=numerical, numerical_known=numerical_known,
                domain_indices=domains, domain_records=domain_records, numeric_semantics=semantics,
                category_classes=CATEGORY_CLASSES, relations=relationships, slot_records=slot_records,
                exact_encoded_graph=encoded, exact_encoded_original_mapping=packed_mapping,
                all_original_outcomes=outcomes, original_children=len(outcomes),
                represented_original_children=len(mapped_ids), unknown_original_children=len(outcomes) - len(mapped_ids),
                graph_categories=dict(Counter((r['weld_type'] + '_' + r['world_kind']) for r in graph['children'])),
                complete_negative_role_supervision=(len(mapped_ids) == len(outcomes)),
                supervision_only=True, neural_initial_inputs_created=False,
                actual_optimizer_updates=0, full_contract_pass=False)


def restore(value, slot_records):
    if isinstance(value, dict) and set(value) == {REF_MARKER}:
        index = value[REF_MARKER]
        if type(index) is not int or not 0 <= index < len(slot_records):
            raise ValueError('Invalid source row pointer')
        return dict(slot_records[index])
    if isinstance(value, dict):
        return {key: restore(v, slot_records) for key, v in value.items()}
    if isinstance(value, list):
        return [restore(v, slot_records) for v in value]
    return value


def save_pack(output, packed):
    output = Path(output)
    output.mkdir(exist_ok=False)
    arrays = {key: value for key, value in packed.items() if isinstance(value, np.ndarray)}
    np.savez(output / 'complete_typed_graph_targets.npz', **arrays)
    manifest = {key: value for key, value in packed.items() if not isinstance(value, np.ndarray)}
    manifest['target_array_sha256'] = digest(output / 'complete_typed_graph_targets.npz')
    occupancy = np.lib.format.open_memmap(output / 'full_original_anchor_role_targets.npy', mode='w+',
                     dtype=np.int8, shape=(packed['anchor_count'], len(SLOT_COLUMNS)))
    occupancy[:] = 0 if packed['complete_negative_role_supervision'] else -1
    for anchor, column, ordinal in packed['slots']:
        occupancy[anchor - 1, column] = 1
    occupancy.flush()
    del occupancy
    manifest['full_original_anchor_role_targets_sha256'] = digest(output / 'full_original_anchor_role_targets.npy')
    manifest['all_original_slot_denominator'] = packed['anchor_count'] * len(SLOT_COLUMNS)
    manifest['graph_positive_definitions'] = len(packed['slots'])
    manifest['known_negative_slot_count'] = (manifest['all_original_slot_denominator'] - len(packed['slots'])
                          if packed['complete_negative_role_supervision'] else 0)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return manifest
