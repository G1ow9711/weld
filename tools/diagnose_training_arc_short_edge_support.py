"""Post-freeze TRAIN diagnosis of observed mesh topology, never runtime selection."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree, Delaunay
from tools.diagnose_training_arc_boundary_support import segment_distances
from threadpoolctl import threadpool_limits
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

BASE = Path(__file__).resolve().parents[1] / 'server_rebuild/expansion_v1'

def main():
    key = 'new_ac2008a03607'
    out = BASE / 'training_arc_short_edge_support_diagnostic_v1'
    out.mkdir(exist_ok=False)
    folder = BASE / 'frozen_features/legacy_runtime' / ('source_' + key) / 'run185' / ('source_' + key)
    freeze = read(folder / 'input_freeze.json')
    path = BASE / 'inputs' / key / 'xyz_mm.npy'
    assert digest(path) == freeze['input_sha256']
    for name in ('surfaces.json', 'surface_support.npz'):
        assert digest(folder / name) == freeze['files'][name]
    membership_path = BASE / 'training_arc_support_membership_diagnostic_v1/report.json'
    membership = read(membership_path)
    totals = {}
    for row in membership['rows']:
        for face in row['faces']:
            totals[face['face']] = totals.get(face['face'], 0) + face['matched_samples']
    selected = sorted(totals, key=lambda f: (-totals[f], f))[:3]
    xyz = np.load(path, mmap_mode='r')
    faces = read(folder / 'surfaces.json')
    truthpath = BASE / 'labels' / key / 'targets.json'
    truth = read(truthpath)
    tree = cKDTree(xyz)
    arcs = []
    for pi, parent in enumerate(truth['parents']):
        for ci, child in enumerate(parent['children']):
            if child['representation']['kind'] == 'arc':
                arc = CircularArc.from_parameters(child['representation']['arc_parameters'])
                points = arc.point_at(np.linspace(0, arc.length_mm, int(np.ceil(arc.length_mm)) + 1))
                arcs.append((pi, ci, tree.query(points, workers=2)[1], points))
    rows = []
    with np.load(folder / 'surface_support.npz') as support:
        for fi in selected:
            face = faces[fi]
            ids = support[face['point_ids_key']]
            points = np.asarray(xyz[ids], np.float64)
            n = np.asarray(face['normal'], float); n /= np.linalg.norm(n)
            origin = points.mean(0)
            axis = np.eye(3)[np.argmin(abs(n))]; u = axis - n * (axis @ n); u /= np.linalg.norm(u)
            basis = np.column_stack((u, np.cross(n, u)))
            uv, inverse = np.unique((points - origin) @ basis, axis=0, return_inverse=True)
            nearest = cKDTree(uv).query(uv, k=2, workers=2)[0][:, 1]
            spacing = float(np.median(nearest[nearest > 1e-6]))
            tri = Delaunay(uv).simplices
            vertices = uv[tri]
            keep = np.linalg.norm(vertices - np.roll(vertices, 1, axis=1), axis=2).max(1) <= 6 * spacing
            full_edges = np.sort(np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1)
            full_edges = np.unique(full_edges, axis=0)
            short_edges = full_edges[np.linalg.norm(uv[full_edges[:, 0]] - uv[full_edges[:, 1]], axis=1) <= 6 * spacing]
            segments = origin + uv[short_edges] @ basis.T
            kept = tri[keep]
            degree = np.bincount(kept.ravel(), minlength=len(uv))
            edges = np.sort(np.concatenate((kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]])), axis=1)
            unique, counts = np.unique(edges, axis=0, return_counts=True)
            boundary_vertex = np.zeros(len(uv), bool); boundary_vertex[unique[counts == 1].ravel()] = True
            lookup = dict(zip(ids.tolist(), inverse.tolist()))
            arc_rows = []
            for pi, ci, raw_ids, reference_points in arcs:
                vi = np.asarray([lookup[int(i)] for i in raw_ids if int(i) in lookup], dtype=np.int64)
                arc_rows.append(dict(parent=pi, child=ci, matched_samples=len(vi),
                    short_edge_distance_quantiles_mm=np.quantile(segment_distances(reference_points, segments), [0, .5, .95, 1]).tolist(),
                    boundary_samples=int(boundary_vertex[vi].sum()),
                    interior_samples=int(((degree[vi] > 0) & ~boundary_vertex[vi]).sum()),
                    no_kept_triangle_samples=int((degree[vi] == 0).sum())))
            row = dict(face=fi, proposal_branch=face.get('proposal_branch'), support_points=len(ids),
                unique_uv=len(uv), spacing_mm=spacing, cutoff_mm=6*spacing,
                triangles=len(tri), kept_triangles=len(kept), short_edges=len(short_edges), arc_rows=arc_rows)
            rows.append(row)
            print(fi, 'spacing', spacing, 'counts', {k: sum(a[k] for a in arc_rows) for k in ('matched_samples', 'boundary_samples', 'interior_samples', 'no_kept_triangle_samples')}, flush=True)
    write(out / 'report.json', dict(rows=rows, selection='Top three faces by nearest reference sample membership; TRAIN diagnostic only, forbidden as runtime selection.',
        source_freeze_sha256=digest(folder / 'input_freeze.json'), membership_sha256=digest(membership_path),
        target_sha256=digest(truthpath), boundary_rule='Diagnostic graph of unique-UV Delaunay edges individually <=6*median positive nearest spacing, including edges of rejected triangles; not material boundary or contact certification.',
        fit_calls=0, DEV_reads=0, heldout_reads=0))

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
