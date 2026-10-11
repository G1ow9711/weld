"""Reference-only STEP topology/plate audit. Never imported by prediction.

CAD measurements here are candidate reference evidence, not reviewed weld-
side attribute labels. No original STEP/PCD/Excel is modified.
"""
import argparse
import json
import re
import time
from pathlib import Path

import gmsh
import numpy as np

from training.labels import sha256


def planar_face(tag):
    lower, upper = gmsh.model.getParametrizationBounds(2, tag)
    lower, upper = np.array(lower), np.array(upper)
    fractions = np.array([[.3, .3], [.3, .7], [.5, .5], [.7, .3], [.7, .7]])
    parameters = lower + fractions * (upper - lower)
    normals = np.array(gmsh.model.getNormal(tag, parameters.ravel().tolist())).reshape(-1, 3)
    normal = normals[2]
    if not np.isfinite(normals).all() or np.linalg.norm(normal) < 1e-9:
        return None
    normal /= np.linalg.norm(normal)
    spread = np.rad2deg(np.arccos(np.clip(normals @ normal, -1., 1.))).max()
    if spread > .02:
        return None
    center = np.array(gmsh.model.occ.getCenterOfMass(2, tag))
    points = np.array(gmsh.model.getValue(2, tag, parameters.ravel().tolist())).reshape(-1, 3)
    residual = np.abs((points - center) @ normal).max()
    if residual > .01:
        return None
    return {'face': tag, 'type': gmsh.model.getType(2, tag), 'area_mm2': gmsh.model.occ.getMass(2, tag),
            'center_mm': center.tolist(), 'parametric_normal': normal.tolist(),
            'normal_spread_degrees': float(spread), 'planarity_residual_mm': float(residual)}


def volume_record(tag):
    mass = gmsh.model.occ.getMass(3, tag)
    center = gmsh.model.occ.getCenterOfMass(3, tag)
    inertia = np.array(gmsh.model.occ.getMatrixOfInertia(3, tag)).reshape(3, 3)
    eigenvalues, vectors = np.linalg.eigh(inertia)
    variance = eigenvalues.sum() / (2. * mass) - eigenvalues / mass
    estimate = float(np.sqrt(max(0., 12. * variance.min())))
    planar, errors = [], []
    boundary = gmsh.model.getBoundary([(3, tag)], combined=False, oriented=True)
    for dim, signed_tag in boundary:
        try:
            face = planar_face(abs(signed_tag))
        except Exception as e:
            errors.append({'face': abs(signed_tag), 'error': str(e)})
            continue
        if face:
            face['boundary_orientation'] = 1 if signed_tag > 0 else -1
            face['boundary_oriented_normal'] = (np.array(face['parametric_normal']) * face['boundary_orientation']).tolist()
            planar.append(face)
    pairs = []
    maximum_area = max((f['area_mm2'] for f in planar), default=0.)
    for i, a in enumerate(planar):
        if a['area_mm2'] < maximum_area * .5:
            continue
        n = np.array(a['parametric_normal'])
        for b in planar[i + 1:]:
            if b['area_mm2'] < maximum_area * .5 or abs(n @ b['parametric_normal']) < .99999:
                continue
            distance = abs(n @ (np.array(b['center_mm']) - a['center_mm']))
            if .1 < distance < 100.:
                pairs.append({'faces': [a['face'], b['face']], 'plane_separation_mm': float(distance),
                              'inertia_estimate_error_mm': float(abs(distance - estimate)),
                              'normal_axis': n.tolist()})
    consistent = [p for p in pairs if p['inertia_estimate_error_mm'] <= .05]
    return {'volume': tag, 'name': gmsh.model.getEntityName(3, tag), 'volume_mm3': mass,
            'center_mm': list(center), 'bbox_mm': list(gmsh.model.getBoundingBox(3, tag)),
            'inertia_thickness_estimate_mm': estimate, 'inertia_normal_axis_unsigned': vectors[:, -1].tolist(),
            'planar_faces': planar, 'parallel_major_face_pairs': pairs,
            'constant_thickness_supported': bool(consistent),
            'supported_thickness_mm': consistent[0]['plane_separation_mm'] if consistent else None,
            'face_errors': errors,
            'normal_scope': 'CAD-parametric and boundary-oriented normals; weld side/identity/sign convention requires separate mapping and review.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', nargs='*')
    ap.add_argument('--outdir', default='weld_pipeline/reports/step_geometry_20260930')
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path('DataSet').rglob('*.stp')) + sorted(Path('DataSet').rglob('*.step'))
    if args.groups:
        paths = [p for p in paths if p.parent.name in args.groups]
    gmsh.initialize([], readConfigFiles=False)
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setString('Geometry.OCCTargetUnit', 'MM')
    summary = []
    try:
        # Verify inertia convention against a translated constant-thickness box.
        box = gmsh.model.occ.addBox(10000., 20000., 30000., 100., 60., 10.)
        gmsh.model.occ.synchronize()
        calibration = volume_record(box)
        if not calibration['constant_thickness_supported'] or abs(calibration['supported_thickness_mm'] - 10.) > .01:
            raise ValueError('CAD thickness/inertia calibration failed')
        (out / 'calibration_box.json').write_text(json.dumps(calibration, indent=2), encoding='utf8')
        for path in paths:
            start = time.perf_counter()
            gmsh.clear()
            imported = gmsh.model.occ.importShapes(str(path.resolve()))
            gmsh.model.occ.synchronize()
            volumes = [volume_record(t) for d, t in gmsh.model.getEntities(3)]
            text = path.read_text('utf8', errors='replace')
            step_names = re.findall(r"MANIFOLD_SOLID_BREP\s*\('([^']*)'", text)
            report = {'step': str(path), 'sha256': sha256(path), 'gmsh_version': gmsh.__version__,
                'scope': 'Reference CAD structural/attribute feasibility audit; not inference input or reviewed attribute acceptance truth.',
                'unit_policy': 'OCC target MM; STEP declarations retained below.',
                'length_unit_declarations': re.findall(r'SI_UNIT\s*\([^)]*METRE[^)]*\)', text)[:8],
                'bbox_mm': list(gmsh.model.getBoundingBox(-1, -1)), 'solid_names_in_step': step_names,
                'entity_counts': {str(d): len(gmsh.model.getEntities(d)) for d in range(4)},
                'volumes': volumes, 'elapsed_seconds': time.perf_counter() - start,
                'coordinate_alignment_to_pcd': 'NOT_VERIFIED_YET'}
            (out / (path.parent.name + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
            thicknesses = sorted({round(v['supported_thickness_mm'], 4) for v in volumes if v['constant_thickness_supported']})
            row = {'name': path.parent.name, 'step_sha256': report['sha256'], 'counts': report['entity_counts'],
                   'bbox_mm': report['bbox_mm'], 'solids_with_supported_thickness': sum(v['constant_thickness_supported'] for v in volumes),
                   'thickness_values_mm': thicknesses, 'elapsed_seconds': report['elapsed_seconds']}
            summary.append(row)
            (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
            print('STEP_AUDIT', json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        gmsh.finalize()


if __name__ == '__main__':
    main()
