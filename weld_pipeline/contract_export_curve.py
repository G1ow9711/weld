"""Independent analytic-curve development export; E attributes stay unknown."""
import csv
import json
from pathlib import Path

import numpy as np

from weld_pipeline.contract_curve_geometry import CurveGeometryError, verify_sampled_arc
from weld_pipeline.contract_export import ATTRIBUTES

VERSION = 'contract_children_analytic_curve_v1'


def save_children(welds, outdir, workpiece_id):
    rows = []
    curves = {}
    for parent in welds:
        for index, segment in enumerate(parent['segments']):
            points = np.asarray(segment['points'], float)
            if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2
                    or not np.isfinite(points).all()):
                raise CurveGeometryError('INVALID_OUTPUT_COORDINATES: finite ordered 3D points required')
            sid = str(parent['id']) + '_' + str(index + 1)
            if sid in curves:
                raise CurveGeometryError('DUPLICATE_OUTPUT_ID: weld_id must be unique within a workpiece')
            record = {'kind': segment['kind'], 'ordered_points_mm': points.tolist(),
                      'order_indices': list(range(len(points)))}
            if segment['kind'] == 'arc':
                audit = verify_sampled_arc(segment)
                length = audit['analytic_length_mm']
                closed = bool(segment['arc_parameters']['is_closed'])
                record.update(arc_parameters=segment['arc_parameters'],
                              sample_positions_mm=segment['sample_positions_mm'],
                              sampling_verification=audit, length_mm=length, is_closed=closed,
                              length_basis='analytic circular arc')
            elif segment['kind'] == 'line':
                if len(points) != 2 or np.array_equal(points[0], points[1]):
                    raise CurveGeometryError('INVALID_LINE: distinct start and end points required')
                length = float(np.linalg.norm(points[-1] - points[0]))
                closed = False
                record.update(length_mm=length, is_closed=False, length_basis='line endpoints')
            else:
                raise CurveGeometryError('UNSUPPORTED_GEOMETRY: explicit line or analytic arc required')
            if not np.isfinite(length) or length <= 0.:
                raise CurveGeometryError('INVALID_OUTPUT_LENGTH: positive finite length required')
            curves[sid] = record
            row = {'workpiece_id': workpiece_id, 'weld_id': sid, 'parent_id': parent['id'],
                   'segment_index': index, 'weld_type': parent['type'], 'geometry_type': segment['kind'],
                   'weld_side': 'unknown', 'is_closed': closed,
                   **dict(zip(['start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z'],
                              [*points[0], *points[-1]])),
                   'length_mm': length, 'curve_reference': 'child_curves.json#' + sid,
                   'confidence': parent.get('confidence'),
                   'status': 'geometry_prediction_attributes_incomplete',
                   'reason_code': 'ATTRIBUTE_ESTIMATOR_AND_SIDE_SCHEMA_PENDING'}
            for name in ATTRIBUTES:
                for field, value in [('value', None), ('status', 'unknown'), ('source', None),
                                     ('confidence', None), ('reason', 'ESTIMATOR_NOT_IMPLEMENTED')]:
                    row[name + '_' + field] = value
            rows.append(row)
    fields = ['workpiece_id', 'weld_id', 'parent_id', 'segment_index', 'weld_type', 'geometry_type',
              'weld_side', 'is_closed', 'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z',
              'length_mm', 'curve_reference', 'confidence', 'status', 'reason_code']
    fields += [name + '_' + field for name in ATTRIBUTES
               for field in ['value', 'status', 'source', 'confidence', 'reason']]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'predicted_children.csv').open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / 'child_curves.json').write_text(json.dumps(curves, ensure_ascii=False, indent=2), encoding='utf8')
    schema = {'version': VERSION, 'units': 'mm', 'angle_units': 'rad',
              'count_unit': 'one predicted geometric child per row', 'attributes': ATTRIBUTES,
              'unknown_values': 'Empty CSV value plus explicit unknown status; never valid or not-applicable.',
              'length_basis': {'line': 'Euclidean endpoint distance',
                               'arc': 'radius_mm * abs(sweep_rad) from delivered arc_parameters; not polyline chord sum'},
              'arc_plane_normal_semantics': 'circle geometric orientation only; not either material-side surface normal',
              'numeric_precision': 'IEEE754 float64, round-trip decimal serialization; geometric residual tolerance 1e-9 mm',
              'd4_arc_sampling': {'max_step_mm': 1., 'max_chord_error_mm': .1, 'max_angle_step_rad': float(np.pi / 2.)},
              'full_contract_output_compliant': False,
              'c1_all_fields_complete': False,
              'missing_features': ['weld_side', 'reviewed segmentation reasons', 'attribute estimators',
                                   'per-attribute reviewed truth and side conventions'],
              'columns': fields}
    (out / 'output_schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding='utf8')
    return len(rows)
