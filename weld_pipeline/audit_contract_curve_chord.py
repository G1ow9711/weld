"""CPU-only deterministic representation audit; no inference or training.

Real source arc points are fitted only after strict 3D residual/topology checks.
This script neither writes truth nor reports learned extraction accuracy.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from weld_pipeline.contract_curve_geometry import (
    CircularArc, fit_ordered_circle, sample_arc, verify_sampled_arc)
from weld_pipeline.contract_export_curve import save_children
from weld_pipeline.continuous_event_decoder_chord import decode_event_sets


OLD_HASHES = {
    'weld_pipeline/continuous_event_decoder.py': '4e48ac7dbb9380883523f6db63462ac2b15a1ca30b20d1ace322315188858c07',
    'weld_pipeline/contract_export.py': '4b6ae45fd8d1932a36fd485c005d2e221e20bfb96e3e3a81e9bdc827db318560',
    'weld_pipeline/curved_sequences.py': '722f5cf977dc0caeb9f5322b07150bec7be53dddd8207737975eb3b62e2db99b',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fit_source_audit(arc, points):
    points = np.asarray(points, float)
    radial = points - arc.center_mm
    plane = radial @ arc.plane_normal
    projected = radial - plane[:, None] * arc.plane_normal
    radial_error = np.linalg.norm(projected, axis=1) - arc.radius_mm
    direction = arc.start_direction
    tangent = np.cross(arc.plane_normal, direction)
    angles = np.unwrap(np.arctan2(projected @ tangent, projected @ direction))
    positions = np.abs(angles - angles[0]) * arc.radius_mm
    # Last parameter comes from the recovered sweep, with no independent
    # rounding of endpoints. Every source point is evaluated against the fit.
    positions[0] = 0.
    positions[-1] = arc.length_mm
    residual = np.linalg.norm(points - arc.point_at(positions), axis=1)
    return {'source_point_count': len(points), 'max_plane_residual_mm': float(np.abs(plane).max()),
            'max_radial_residual_mm': float(np.abs(radial_error).max()),
            'max_ordered_3d_fit_residual_mm': float(residual.max()),
            'source_polyline_length_mm': float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())}


def synthetic_cases():
    rows = []
    # Five radii and varied topology; no model or per-workpiece configuration.
    normal = np.asarray([1., 2., 3.]); normal /= np.linalg.norm(normal)
    basis = np.cross(normal, [0., 0., 1.]); basis /= np.linalg.norm(basis)
    for radius in [.02, .1, .5, 1., 100.]:
        for label, sweep, tilted in [('short', 1e-5, False), ('open', .2, False),
                                     ('negative', -2., False), ('tilted', 2., True),
                                     ('closed_positive', 2. * math.pi, True),
                                     ('closed_negative', -2. * math.pi, True)]:
            arc = CircularArc([10., -20., 30.], radius, normal if tilted else [0., 0., 1.],
                              basis if tilted else [1., 0., 0.], sweep,
                              label.startswith('closed'))
            cuts = [arc.length_mm * .271, arc.length_mm * .623]
            segment = sample_arc(arc, breakpoints_mm=cuts)
            row = verify_sampled_arc(segment)
            assert all(cut in segment['sample_positions_mm'] for cut in cuts)
            assert np.array_equal(segment['points'][0], arc.point_at(0.))
            assert np.array_equal(segment['points'][-1], arc.point_at(arc.length_mm))
            rows.append({'radius_mm': radius, 'case': label, **row,
                         'exact_breakpoints_retained': True, 'exact_endpoints_retained': True})
    # Crossing the 2*pi angle origin is geometry, not a discontinuity.
    arc = CircularArc([0., 0., 0.], .02, [0., 0., 1.],
                      [math.cos(6.2), math.sin(6.2), 0.], .3)
    rows.append({'radius_mm': .02, 'case': 'cross_zero_angle', **verify_sampled_arc(sample_arc(arc))})
    # Tiny genuine gaps and cuts are retained by the actual opt-in decoder.
    radius = .02
    complete = CircularArc([0., 0., 0.], radius, [0., 0., 1.], [1., 0., 0.], 4.)
    seq = {'curve_kind': 'arc', 'type': 'horizontal', 'sample_step_mm': .01,
           'xyz': complete.point_at(np.linspace(0., complete.length_mm, 9)),
           'sequence_arc_parameters': complete.to_parameters()}
    events = ([[.001, .9999], [.03000001, .9999]],
              [[.03, .9999], [.06, .9999]], [[.0100001, .9999]])
    decoded = decode_event_sets([seq], [np.full(9, .95)], [events])
    intervals = [segment['source_interval_mm'] for weld in decoded for segment in weld['segments']]
    assert intervals == [[.001, .0100001], [.0100001, .03], [.03000001, .06]]
    assert np.array_equal(decoded[0]['segments'][0]['points'][-1], decoded[0]['segments'][1]['points'][0])
    for weld in decoded:
        for segment in weld['segments']:
            verify_sampled_arc(segment)
    return rows, {'retained_intervals_mm': intervals, 'exact_common_cut': True,
                  'positive_gap_retained_mm': .03000001 - .03, 'segments': len(intervals)}


def run(source_manifest, outdir):
    started = time.perf_counter()
    source_manifest = Path(source_manifest)
    outdir = Path(outdir)
    old_before = {name: sha256(name) for name in OLD_HASHES}
    if old_before != OLD_HASHES:
        raise RuntimeError('FROZEN_SOURCE_CHANGED: original decoder/export/curved_sequences hash mismatch')
    source_hash_before = sha256(source_manifest)
    manifest = json.loads(source_manifest.read_text(encoding='utf8'))
    real = []; fixture_welds = []
    for parent in manifest['truth']:
        segments = []
        for index, source in enumerate(parent['segments']):
            if source['kind'] != 'arc':
                continue
            arc = fit_ordered_circle(source['points'], tolerance_mm=1e-6,
                                     is_closed=np.array_equal(source['points'][0], source['points'][-1]))
            fit_audit = _fit_source_audit(arc, source['points'])
            if fit_audit['max_ordered_3d_fit_residual_mm'] > 1e-6:
                raise RuntimeError('SOURCE_FIT_RESIDUAL_FAILED: no analytic representation was certified')
            segment = sample_arc(arc)
            # Real endpoints survive in the exported fixture, bit for bit.
            segment['points'][0] = source['points'][0]
            segment['points'][-1] = source['points'][-1]
            checked = verify_sampled_arc(segment)
            old_length = float(source['length_mm'])
            real.append({'parent_id': parent['id'], 'segment_index': index, **fit_audit, **checked,
                         'source_length_field_mm': old_length,
                         'analytic_minus_source_length_field_mm': arc.length_mm - old_length,
                         'radius_mm': arc.radius_mm, 'sweep_rad': arc.sweep_rad,
                         'exact_source_endpoints_retained': True,
                         'status': 'ANALYTIC_REPRESENTATION_VERIFIED_FROM_ORDERED_POINTS'})
            segments.append(segment)
        if segments:
            fixture_welds.append({'id': parent['id'], 'type': parent['type'], 'segments': segments})
    # Labels never enter the decoder: this is a labelled geometry-export fixture.
    exported = save_children(fixture_welds, outdir / 'gt_representation_fixture_export', 'TD2-LB8A')
    export_path = outdir / 'gt_representation_fixture_export'
    curves = json.loads((export_path / 'child_curves.json').read_text(encoding='utf8'))
    with (export_path / 'predicted_children.csv').open(encoding='utf8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        parameters = curves[row['weld_id']]['arc_parameters']
        delivered = CircularArc.from_parameters(parameters)
        if float(row['length_mm']) != delivered.length_mm or row['weld_side'] != 'unknown':
            raise RuntimeError('EXPORT_ROUNDTRIP_FAILED')
        if any(row[name + '_status'] != 'unknown' for name in ['normal_1', 'normal_2', 'thickness_1', 'thickness_2']):
            raise RuntimeError('UNKNOWN_ATTRIBUTE_SCHEMA_FAILED')
    synthetic, gaps = synthetic_cases()
    new_sources = ['weld_pipeline/contract_curve_geometry.py', 'weld_pipeline/continuous_event_decoder_chord.py',
                   'weld_pipeline/contract_export_curve.py', 'weld_pipeline/audit_contract_curve_chord.py']
    tests = Path('weld_pipeline/tests/test_contract_curve_geometry.py')
    if tests.exists():
        new_sources.append(str(tests).replace('\\', '/'))
    frozen_after = {name: sha256(name) for name in OLD_HASHES}
    report = {'status': 'REPRESENTATION_CHECKS_PASS',
              'scope': 'CPU deterministic geometry and serialization only; not learned extraction accuracy or formal acceptance',
              'model_inference_run': False, 'model_training_run': False,
              'default_inference_changed': False, 'step_runtime_input': False,
              'full_contract_c1_complete': False, 'full_contract_e_complete': False,
              'runtime': {'python': sys.executable, 'torch_loaded': 'torch' in sys.modules,
                          'threads': {key: os.environ.get(key) for key in
                                      ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
                          'temp': os.environ.get('TEMP'), 'pycache': os.environ.get('PYTHONPYCACHEPREFIX'),
                          'elapsed_s': time.perf_counter() - started},
              'hashes': {'source_manifest': {str(source_manifest): source_hash_before},
                         'source_gt_unchanged': sha256(source_manifest) == source_hash_before,
                         'old_sources_before': old_before, 'old_sources_after': frozen_after,
                         'old_sources_unchanged': old_before == frozen_after,
                         'new_sources': {name: sha256(name) for name in new_sources},
                         'contract_text': sha256('weld_pipeline/reports/contract_child_20260930/contract_text.txt')},
              'real_checks': {'arc_segments': len(real), 'fixture_export_rows': exported,
                              'analytic_parameters_source': 'strict 3D ordered-point circle fit, max residual <=1e-6 mm',
                              'max_source_fit_residual_mm': max(item['max_ordered_3d_fit_residual_mm'] for item in real),
                              'max_arc_step_mm': max(item['max_arc_step_mm'] for item in real),
                              'max_chord_error_mm': max(item['max_chord_error_mm'] for item in real),
                              'max_length_field_roundtrip_error_mm': 0., 'details': real},
              'synthetic_checks': {'arc_cases': len(synthetic), 'details': synthetic, 'event_gap_cut_case': gaps},
              'limits': ['source truth fixture is not a prediction or accuracy score',
                         'new decoder/export are opt-in and require explicit integration',
                         'arbitrary-plane geometry support is not a trained arbitrary-plane arc detector',
                         'material-side normals, weld_side, E attributes and reviewed segmentation reasons remain unknown',
                         'general splines, multi-turn circles and degenerate or numerically collapsed geometry are explicit errors']}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'representation_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': report['status'], 'real_arc_checks': len(real), 'synthetic_arc_checks': len(synthetic),
                      'max_source_fit_residual_mm': report['real_checks']['max_source_fit_residual_mm'],
                      'max_real_arc_step_mm': report['real_checks']['max_arc_step_mm'],
                      'max_real_chord_error_mm': report['real_checks']['max_chord_error_mm'],
                      'old_sources_unchanged': report['hashes']['old_sources_unchanged'],
                      'source_gt_unchanged': report['hashes']['source_gt_unchanged'],
                      'torch_loaded': report['runtime']['torch_loaded']}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-manifest', default='weld_pipeline/data/mixed_surface_v1/TD2-LB8A/manifest.json')
    parser.add_argument('--outdir', default='weld_pipeline/reports/curve_chord_178_20261002')
    args = parser.parse_args()
    run(args.source_manifest, args.outdir)
