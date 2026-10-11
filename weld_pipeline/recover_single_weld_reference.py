"""Audit an ordered, single-weld reference PLY; never use during inference."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.truth import excel_rows


def read_reference_ply(path):
    with Path(path).open('r', encoding='ascii') as stream:
        header = []
        for line in stream:
            header.append(line.strip())
            if line.strip() == 'end_header':
                break
        if 'format ascii 1.0' not in header or 'comment unit millimeter; frame reference_workpiece' not in header:
            raise ValueError('Unsupported reference PLY format or coordinate frame')
        count = int(next(s for s in header if s.startswith('element vertex ')).split()[-1])
        points = np.loadtxt(stream, usecols=(0, 1, 2), ndmin=2)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError('Invalid PLY vertex array')
    return points


def endpoint_transforms(local, world):
    """Enumerate both planar handedness choices; dense geometry must select one."""
    a, b = local; c, d = world
    if abs(b[2]-a[2]) > 1e-6 or abs(d[2]-c[2]) > 1e-6:
        raise ValueError('This audit supports horizontal planar curves only')
    u = b-a; v = d-c
    if abs(np.linalg.norm(u)-np.linalg.norm(v)) > 1e-5:
        raise ValueError('Endpoint distances disagree')
    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    normal = np.array([0., 0., 1.])
    local_basis = np.column_stack([u, np.cross(normal, u), normal])
    result = []
    for sign in [1., -1.]:
        target_basis = np.column_stack([v, sign*np.cross(normal, v), sign*normal])
        r = target_basis @ local_basis.T
        transform = np.eye(4); transform[:3, :3] = r; transform[:3, 3] = c-r@a
        result.append(transform)
    return result


def fit_primitive(points):
    a, b = points[0], points[-1]; direction = b-a
    chord = np.linalg.norm(direction)
    if chord <= 1e-8 or len(points) < 3:
        raise ValueError('Degenerate segment')
    line_error = np.linalg.norm(np.cross(points-a, direction/chord), axis=1).max()
    if line_error < 1e-5:
        return {'kind': 'line', 'lengthMm': float(chord), 'maxFitErrorMm': float(line_error)}
    if np.ptp(points[:, 2]) > 1e-6:
        raise ValueError('Non-horizontal arc')
    origin = points[:, :2].mean(axis=0); xy = points[:, :2]-origin
    coef = np.linalg.lstsq(np.column_stack([2*xy, np.ones(len(xy))]), (xy**2).sum(axis=1), rcond=None)[0]
    center = coef[:2]; radius = np.sqrt(coef[2]+np.dot(center, center))
    error = np.abs(np.linalg.norm(xy-center, axis=1)-radius).max()
    angles = np.unwrap(np.arctan2(xy[:, 1]-center[1], xy[:, 0]-center[0]))
    delta = np.diff(angles)
    if error > .001 or not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError('Curve is not a monotone circular arc')
    return {'kind': 'arc', 'lengthMm': float(radius*abs(angles[-1]-angles[0])),
            'centerXY': (center+origin).tolist(), 'radiusMm': float(radius), 'maxFitErrorMm': float(error)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--folder', required=True); ap.add_argument('--outdir', required=True)
    args = ap.parse_args(); folder = Path(args.folder); out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    xlsx = next(folder.glob('*.xlsx')); rows = excel_rows(xlsx)
    if len(rows) != 1:
        raise ValueError('Multiple welds require explicit per-vertex identities')
    row = rows[0]; ply = next(folder.glob('*welds.ply')); workpiece = next(folder.glob('*workpiece.ply'))
    pcd = next(folder.glob('*AllFaces.pcd'))
    q = read_reference_ply(ply)
    start = np.array([row['StartMatrix.Translation.'+c] for c in 'XYZ'], float)
    end = np.array([row['EndMatrix.Translation.'+c] for c in 'XYZ'], float)
    world = np.array([[row['Source '+side+'Matrix.Translation.'+c] for c in 'XYZ'] for side in ['Start','End']], float)
    if np.linalg.norm(q[0]-start) > 1e-5 or np.linalg.norm(q[-1]-end) > 1e-5:
        raise ValueError('Reference order disagrees with Excel endpoints')
    cut = np.flatnonzero(np.linalg.norm(np.diff(q, axis=0), axis=1) < 1e-8)+1
    chunks = np.split(q, cut)
    if len(chunks) != row['SegmentCount']:
        raise ValueError('Duplicate boundaries do not match declared segment count')
    segments = [dict(fit_primitive(p), points=p.tolist()) for p in chunks]
    length = sum(s['lengthMm'] for s in segments)
    if abs(length-row['Length']) > .01 or any(s['kind']=='arc' for s in segments) != bool(row['HasMid (Any)']):
        raise ValueError('Recovered geometry disagrees with Excel length/arc flag')
    dense = read_reference_ply(workpiece); tree = cKDTree(dense)
    raw = read_xyz_pcd(pcd)*1000
    transforms = endpoint_transforms(np.array([start,end]), world); checks = []
    for matrix in transforms:
        local = (raw-matrix[:3,3]) @ matrix[:3,:3]
        distances = tree.query(local, workers=2)[0]
        checks.append({'pcd_points': len(raw), 'median_mm': float(np.median(distances)),
                       'p95_mm': float(np.quantile(distances,.95)), 'p99_mm': float(np.quantile(distances,.99)),
                       'max_mm': float(distances.max()), 'exportToWorld': matrix.tolist()})
    chosen = int(np.argmin([c['p95_mm'] for c in checks])); other = 1-chosen
    # Predeclared conservative gates: every PCD sample supports the alignment,
    # and the alternate handedness is clearly worse. Fail closed otherwise.
    accepted = checks[chosen]['max_mm'] <= 3 and checks[other]['p95_mm'] > checks[chosen]['p95_mm']+5
    receipt = {'status': 'RECOVERED_REFERENCE' if accepted else 'FRAME_REQUIRES_REVIEW',
               'scope': 'Derived customer reference geometry only; not model predictions or independent accuracy.',
               'group': folder.name, 'source_hashes': {str(p): sha256(p) for p in [xlsx,ply,workpiece,pcd,Path(__file__)]},
               'reference_vertices': len(q), 'duplicate_boundary_indices': cut.tolist(),
               'segment_kinds': [s['kind'] for s in segments], 'length_excel_mm': row['Length'], 'length_recovered_mm': length,
               'coordinate_checks': checks, 'selected_transform': chosen if accepted else None}
    if accepted:
        recovered = {'unit':'millimeter','matrixConvention':'row_major_column_vector','coordinateFrame':'reference_workpiece',
                     'exportToWorld':transforms[chosen].tolist(),'provenance':receipt,
                     'welds':[{'id':row['ObjectId'],'name':row['Name'],'segments':segments}]}
        dest = out/(folder.name+'_recovered_welds.json')
        dest.write_text(json.dumps(recovered,ensure_ascii=False,indent=2),encoding='utf-8')
        receipt['recovered_file'] = str(dest); receipt['recovered_sha256'] = sha256(dest)
    (out/'recovery_audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
