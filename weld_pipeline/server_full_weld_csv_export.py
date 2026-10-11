"""Export complete model predictions without changing geometry or field states.

The canonical CSV is a development interchange format, not an unconfirmed
customer spreadsheet layout. Raw child and parent JSON preserve all metadata.
Unknown child flags stay unknown; parent-level Any flags are never broadcast.
Signed endpoint axes are neither normalized nor flipped to improve a score.
"""
import csv
import io
import json
from pathlib import Path

import numpy as np

from .runtime_csv_export_287 import atomic_bytes, atomic_json

VERSION = 'COMPLETE_MODEL_ORDERED_CHILD_CSV_V1'
FIELDS = ('front_wall', 'back_wall', 'start_height', 'end_height',
          'start_hole_height', 'end_hole_height', 'start_base_edge', 'end_base_edge')
STATES = ('EXACT_MM', 'DISTANCE_GT_50', 'NO_HOLE', 'NO_BLOCKER_OR_DISTANCE_GT_50')
FLAGS = ('has_eave', 'side_plate_split', 'segment_reason', 'hole_edge')
BASE_COLUMNS = ('workpiece_id', 'parent_id', 'weld_id', 'segment_index', 'segment_count',
    'geometry_type', 'weld_type', 'predicted_side', 'is_closed', 'parent_is_closed', 'length_mm',
    'start_x_mm', 'start_y_mm', 'start_z_mm', 'end_x_mm', 'end_y_mm', 'end_z_mm',
    'gap_before', 'curve_file', 'parent_metadata_file', 'arc_parameters_json',
    'endpoint_attributes_json', 'raw_frame_axes_state', 'raw_frame_origins_state',
    'frame_business_semantics_certified', 'unmodeled_child_flags_state', 'status', 'reason')
FRAME_COLUMNS = tuple(
    f'{end}_frame_origin_{coord}_mm' for end in ('start', 'end') for coord in ('x', 'y', 'z')) + tuple(
    f'{end}_frame_{axis}_{coord}' for end in ('start', 'end')
    for axis in ('X', 'Y', 'Z') for coord in ('x', 'y', 'z'))
FIELD_COLUMNS = tuple(f'{name}_{column}' for name in FIELDS
                      for column in ('value_mm', 'state', 'legacy_code', 'encoding_valid', 'required'))
COLUMNS = BASE_COLUMNS + FRAME_COLUMNS + FIELD_COLUMNS


def _array(value, shape):
    try:
        result = np.asarray(value, np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result.shape == shape and np.isfinite(result).all() else None


def field_encoding(name, attribute, required):
    """Keep raw states and values; only emit a compatible legacy scalar code."""
    if not isinstance(attribute, dict):
        attribute = {}
    state = attribute.get('state', 'UNKNOWN')
    value = _array(attribute.get('value_mm'), ()) if state == 'EXACT_MM' else None
    valid, code = False, None
    if state == 'EXACT_MM' and value is not None and float(value) >= 0:
        valid = True
        if name in ('front_wall', 'back_wall', 'start_base_edge', 'end_base_edge'):
            valid = float(value) <= 50
        if name in ('start_hole_height', 'end_hole_height'):
            valid = float(value) > 0
        if valid:
            code = float(value)
    elif state == 'NO_BLOCKER_OR_DISTANCE_GT_50' and name in ('front_wall', 'back_wall'):
        valid, code = True, -1
    elif state == 'DISTANCE_GT_50' and name in ('start_base_edge', 'end_base_edge'):
        valid, code = True, 51
    elif state == 'NO_HOLE' and name in ('start_hole_height', 'end_hole_height'):
        valid, code = True, 0
    elif state == 'NOT_REQUIRED' and required is False:
        valid = True
    return dict(value_mm=float(value) if value is not None else None, state=state,
                legacy_code=code, encoding_valid=valid, required=required)


def export_decoded(directory, workpiece_id, decoded):
    if decoded.get('GT_or_STEP_reads') != 0:
        raise ValueError('Input-only decoded predictions are required')
    parents = decoded['parents']
    invalid = decoded.get('invalid_predictions', [])
    if invalid:
        raise ValueError('This complete-model decoder requires explicit valid selected parents; invalid outputs cannot be silently removed')
    expected = sum(len(parent['segments']) for parent in parents)
    if expected != decoded['total_prediction_children']:
        raise ValueError('Output child denominator differs from complete parent contents')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rows, unknown_flags, missing_axes, missing_origins, incompatible_codes = [], 0, 0, 0, 0
    atomic_json(directory / 'parents.json', parents)
    for pi, parent in enumerate(parents):
        if parent['child_count'] != len(parent['segments']) or parent['child_count'] < 1:
            raise ValueError('Parent child count changed')
        for ci, child in enumerate(parent['segments']):
            points = np.asarray(child['points'], np.float64)
            if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2
                    or not np.isfinite(points).all() or child['kind'] not in ('line', 'arc')
                    or not np.isfinite(child['length_mm']) or child['length_mm'] <= 0):
                raise ValueError('Malformed selected geometry must be repaired before export')
            weld_id = f'w{len(rows):06d}'
            curve = Path('curves') / (weld_id + '.json')
            atomic_json(directory / curve, child)
            row = dict.fromkeys(COLUMNS, '')
            row.update(workpiece_id=workpiece_id, parent_id=f'p{pi:06d}', weld_id=weld_id,
                segment_index=ci, segment_count=parent['child_count'], geometry_type=child['kind'],
                weld_type=child.get('weld_type', 'UNKNOWN'), predicted_side=child.get('predicted_side', 'UNKNOWN'),
                is_closed=child['is_closed'], parent_is_closed=parent['is_closed'], length_mm=child['length_mm'],
                gap_before='START' if ci == 0 else parent['gaps'][ci - 1],
                curve_file=curve.as_posix(), parent_metadata_file='parents.json#' + str(pi),
                arc_parameters_json=json.dumps(child['arc_parameters'], allow_nan=False) if child['kind'] == 'arc' else '',
                endpoint_attributes_json=json.dumps(child.get('endpoint_attributes', {}), ensure_ascii=False, allow_nan=False),
                frame_business_semantics_certified=child.get('frame_business_semantics_certified') is True,
                unmodeled_child_flags_state='UNKNOWN_SEE_ORIGINAL_CHILD_JSON',
                status='MODEL_PREDICTION_FIELDS_AND_BUSINESS_FRAMES_UNCERTIFIED')
            for column, value in zip(('start_x_mm', 'start_y_mm', 'start_z_mm',
                                      'end_x_mm', 'end_y_mm', 'end_z_mm'), points[[0, -1]].ravel()):
                row[column] = float(value)
            axes = _array(child.get('raw_export_frame_axes'), (2, 3, 3))
            origins = _array(child.get('frame_origins_mm'), (2, 3))
            row['raw_frame_axes_state'] = 'RAW_SIGNED_MODEL_AXES' if axes is not None else 'UNKNOWN_OR_MALFORMED'
            row['raw_frame_origins_state'] = 'RAW_MODEL_ORIGINS_MM' if origins is not None else 'UNKNOWN_OR_MALFORMED'
            missing_axes += int(axes is None); missing_origins += int(origins is None)
            for side, end in enumerate(('start', 'end')):
                if origins is not None:
                    for coord, value in zip(('x', 'y', 'z'), origins[side]):
                        row[f'{end}_frame_origin_{coord}_mm'] = float(value)
                if axes is not None:
                    for axis, vector in zip(('X', 'Y', 'Z'), axes[side]):
                        for coord, value in zip(('x', 'y', 'z'), vector):
                            row[f'{end}_frame_{axis}_{coord}'] = float(value)
            for name in FIELDS:
                weld = child.get('weld_type')
                required = True if weld == 'horizontal' else name == 'back_wall' if weld == 'vertical' else 'UNKNOWN'
                encoded = field_encoding(name, child.get('endpoint_attributes', {}).get(name, {}), required)
                for column, value in encoded.items():
                    row[f'{name}_{column}'] = value if value is not None else ''
                incompatible_codes += int(encoded['state'] in STATES and not encoded['encoding_valid'])
            # Existing flags are kept verbatim in child JSON. No parent flag is
            # interpreted as a per-child reason or duplicated into a child row.
            unknown_flags += sum(name not in child for name in FLAGS)
            rows.append(row)
    if len(rows) != expected:
        raise ValueError('Complete model export lost a child')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    atomic_bytes(directory / 'welds.csv', stream.getvalue().encode('utf-8-sig'))
    schema = dict(version=VERSION, units='mm', angle_units='rad', columns=COLUMNS,
        count_unit='One selected ordered child per row; no minimum-length deletion or overlap merge',
        parent_metadata='Original JSON preserves all gaps, nodes and parent flags',
        child_metadata='Original sampled child JSON preserves every original field and signed circular sweep',
        curve_normal='Analytic circle orientation; never a substitute for plate normals',
        frames='Original ordered row axes X/Y/Z and endpoint origins, with signs unchanged',
        unknown_values='Blank scalar plus explicit original state; never filled with zero',
        legacy_codes='Only compatible states encode wall -1, outer edge 51 or no-hole 0; these are not exact distances',
        incompatible_legacy_codes=incompatible_codes, missing_raw_axes=missing_axes,
        missing_raw_origins=missing_origins, absent_child_flags=unknown_flags,
        published_rows=len(rows), published_parents=len(parents), original_predictions_changed=False,
        customer_csv_layout_frozen=False, segmentation_business_semantics_certified=False,
        full_contract_pass=False)
    atomic_json(directory / 'output_schema.json', schema)
    return schema


def verify_roundtrip(directory, decoded):
    """Authenticate complete row order, every original JSON and signed scalar."""
    directory = Path(directory)
    saved_parents = json.loads((directory / 'parents.json').read_text())
    if saved_parents != decoded['parents']:
        raise ValueError('Parent metadata changed during CSV export')
    with (directory / 'welds.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames) != COLUMNS:
            raise ValueError('CSV schema changed')
        rows = list(reader)
    children = [(pi, ci, child) for pi, parent in enumerate(decoded['parents'])
                for ci, child in enumerate(parent['segments'])]
    if len(rows) != len(children) or len(rows) != decoded['total_prediction_children']:
        raise ValueError('CSV export lost or repeated children')
    for row, (pi, ci, child) in zip(rows, children):
        if row['parent_id'] != f'p{pi:06d}' or int(row['segment_index']) != ci:
            raise ValueError('CSV order differs from original parents or children')
        path = directory / row['curve_file']
        path.resolve().relative_to(directory.resolve())
        if json.loads(path.read_text()) != child:
            raise ValueError('Raw child geometry or metadata changed')
        positions = np.asarray([[float(row[f'{end}_{coord}_mm']) for coord in ('x', 'y', 'z')]
                                for end in ('start', 'end')])
        if positions.tobytes() != np.asarray(child['points'], np.float64)[[0, -1]].tobytes():
            raise ValueError('Signed ordered endpoint coordinates changed')
        for name, shape, columns in [
            ('raw_export_frame_axes', (2, 3, 3),
             [f'{end}_frame_{axis}_{coord}' for end in ('start', 'end') for axis in ('X', 'Y', 'Z') for coord in ('x', 'y', 'z')]),
            ('frame_origins_mm', (2, 3),
             [f'{end}_frame_origin_{coord}_mm' for end in ('start', 'end') for coord in ('x', 'y', 'z')])]:
            original = _array(child.get(name), shape)
            if original is None:
                if any(row[column] for column in columns):
                    raise ValueError('Missing frame was invented')
            elif np.asarray([float(row[column]) for column in columns]).reshape(shape).tobytes() != original.tobytes():
                raise ValueError('Raw signed frame changed')
        if json.loads(row['endpoint_attributes_json']) != child.get('endpoint_attributes', {}):
            raise ValueError('Original endpoint states or confidence changed')
    return dict(complete_child_rows=len(rows), all_parent_child_json_unchanged=True,
                ordered_positions_and_signed_frames_bitwise_equal=True,
                original_endpoint_states_unchanged=True)
