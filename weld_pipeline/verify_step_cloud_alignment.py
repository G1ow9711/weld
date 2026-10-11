"""Visual and geometric STEP/PCD cross-check for reference diagnosis only."""
import argparse
import itertools
import json
import time
from pathlib import Path

import gmsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.truth import load_truth


def mesh_step(path):
    gmsh.clear()
    gmsh.model.occ.importShapes(str(path.resolve()))
    gmsh.model.occ.synchronize()
    biggest = max(gmsh.model.getEntities(3), key=lambda x: gmsh.model.occ.getMass(*x))
    inertia = np.array(gmsh.model.occ.getMatrixOfInertia(*biggest)).reshape(3, 3)
    eigenvalues, basis = np.linalg.eigh(inertia)
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1.
    gmsh.option.setNumber('Mesh.MeshSizeMin', .5)
    gmsh.option.setNumber('Mesh.MeshSizeMax', 55.)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 64)
    gmsh.option.setNumber('Mesh.MinimumCirclePoints', 64)
    gmsh.option.setNumber('Mesh.Algorithm', 6)
    gmsh.model.mesh.generate(2)
    node_ids, coords, _ = gmsh.model.mesh.getNodes()
    vertices = np.array(coords).reshape(-1, 3)
    # Native node tags need not be contiguous.
    order = np.argsort(node_ids)
    keys = np.asarray(node_ids)[order]
    triangles, owners, face_ids = [], [], []
    volumes = gmsh.model.getEntities(3)
    ownership = {abs(tag): v for dim, v in volumes
                 for d, tag in gmsh.model.getBoundary([(dim, v)], combined=False, oriented=False)}
    for dim, tag in gmsh.model.getEntities(2):
        types, element_tags, nodes = gmsh.model.mesh.getElements(2, tag)
        for kind, connectivity in zip(types, nodes):
            name, edim, eorder, count, local, primary = gmsh.model.mesh.getElementProperties(kind)
            if primary != 3:
                raise ValueError('Only linear triangular CAD surface mesh is supported')
            indices = order[np.searchsorted(keys, np.asarray(connectivity).reshape(-1, count)[:, :3])]
            triangles.extend(indices.tolist())
            owners.extend([ownership.get(tag, -1)] * len(indices))
            face_ids.extend([tag] * len(indices))
    return vertices, np.array(triangles, np.int32), np.array(owners), np.array(face_ids), basis


def cloud(points):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(points)
    return p


def align(vertices, triangles, basis, xyz):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()
    o3d.utility.random.seed(130)
    source = np.asarray(mesh.sample_points_uniformly(70000).points)
    rng = np.random.default_rng(130)
    sample = xyz[rng.choice(len(xyz), min(150000, len(xyz)), replace=False)]
    z, count = np.unique(xyz[:, 2].round(3), return_counts=True)
    level = z[np.argmax(count)]
    flat = xyz[np.abs(xyz[:, 2] - level) < .01]
    values, vectors = np.linalg.eigh(np.cov(flat[:, :2].T))
    u = np.r_[vectors[:, -1], 0.]
    v = np.cross([0., 0., 1.], u)
    target_basis = np.column_stack([u, v, [0., 0., 1.]])
    tree = cKDTree(sample)
    candidates = []
    for swap in [False, True]:
        permute = np.eye(3)[:, [1, 0, 2]] if swap else np.eye(3)
        for signs in itertools.product([-1., 1.], repeat=3):
            middle = permute @ np.diag(signs)
            if np.linalg.det(middle) < 0:
                continue
            rotation = target_basis @ middle @ basis.T
            rotated = vertices @ rotation.T
            translation = (xyz.min(0) + xyz.max(0) - rotated.min(0) - rotated.max(0)) / 2.
            trial = source[::7] @ rotation.T + translation
            distances = tree.query(trial, workers=2)[0]
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation, translation
            candidates.append({'initial_mean_distance_mm': float(np.minimum(distances, 100.).mean()),
                               'transform': transform})
    candidates.sort(key=lambda c: c['initial_mean_distance_mm'])
    target = cloud(sample).voxel_down_sample(3.)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=45., max_nn=50))
    src = cloud(source)
    results = []
    for initial in candidates[:4]:
        result = o3d.pipelines.registration.registration_icp(src, target, 45., initial['transform'],
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.HuberLoss(3.)),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        aligned = vertices @ result.transformation[:3, :3].T + result.transformation[:3, 3]
        final_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(aligned), o3d.utility.Vector3iVector(triangles))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(final_mesh))
        d = scene.compute_distance(o3d.core.Tensor(sample.astype(np.float32))).numpy()
        results.append({'initial_mean_distance_mm': initial['initial_mean_distance_mm'],
                        'icp_fitness': result.fitness, 'icp_rmse_mm': result.inlier_rmse,
                        'point_to_cad_median_mm': float(np.median(d)), 'point_to_cad_p95_mm': float(np.percentile(d, 95)),
                        'point_to_cad_max_mm': float(d.max()), 'within_3mm_rate': float((d <= 3.).mean()),
                        'transform': result.transformation.tolist()})
    results.sort(key=lambda r: (r['point_to_cad_p95_mm'], r['point_to_cad_median_mm']))
    return results


def draw(name, vertices, triangles, owners, xyz, truth, out):
    fig = plt.figure(figsize=(16, 10))
    cmap = plt.get_cmap('tab20')
    colors = np.array([cmap(int(v) % 20) for v in owners])
    center = (vertices.min(0) + vertices.max(0)) / 2.
    span = np.ptp(vertices, axis=0)
    for panel, (elevation, azimuth) in enumerate([(30, -65), (90, -90), (0, -90), (0, 0)], start=1):
        ax = fig.add_subplot(2, 2, panel, projection='3d')
        collection = Poly3DCollection(vertices[triangles], facecolor=colors, edgecolor='none', alpha=.7, rasterized=True)
        ax.add_collection3d(collection)
        for w in truth:
            if w['type'] != 'vertical':
                continue
            for segment in w['segments']:
                p = np.array(segment['points'])
                ax.plot(p[:, 0], p[:, 1], p[:, 2], color='red', linewidth=1.5)
                ax.scatter(p[[0, -1], 0], p[[0, -1], 1], p[[0, -1], 2], color='black', s=8)
        for j, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            setter(center[j] - span[j] * .53 - 1., center[j] + span[j] * .53 + 1.)
        ax.set_box_aspect(np.maximum(span, .1))
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xlabel('X / mm'); ax.set_ylabel('Y / mm'); ax.set_zlabel('Z / mm')
        ax.set_title(f'{name}: CAD solids; red = reference vertical weld; dots = cuts')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', nargs='+', default=['224-GR24A1', '527-FR146A'])
    args = ap.parse_args()
    out = Path('weld_pipeline/reports/step_alignment_20260930')
    out.mkdir(parents=True, exist_ok=True)
    gmsh.initialize([], readConfigFiles=False)
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setString('Geometry.OCCTargetUnit', 'MM')
    try:
        for name in args.groups:
            start = time.perf_counter()
            folder = Path('DataSet/9月20日') / name
            step, pcd = next(folder.glob('*.stp')), next(folder.glob('*AllFaces.pcd'))
            vertices, triangles, owners, face_ids, basis = mesh_step(step)
            xyz = read_xyz_pcd(pcd) * 1000.
            results = align(vertices, triangles, basis, xyz)
            transform = np.array(results[0]['transform'])
            aligned = vertices @ transform[:3, :3].T + transform[:3, 3]
            np.savez_compressed(out / (name + '_mesh.npz'), vertices=vertices, triangles=triangles, owners=owners,
                                face_ids=face_ids, aligned_vertices=aligned, transform=transform)
            truth, excluded = load_truth(folder)
            draw(name, aligned, triangles, owners, xyz, truth, out / (name + '_cad_weld_views.png'))
            report = {'name': name, 'step': str(step), 'pcd': str(pcd), 'step_sha256': sha256(step),
                      'pcd_sha256': sha256(pcd), 'source_sha256': sha256(__file__),
                      'vertices': len(vertices), 'triangles': len(triangles), 'raw_cad_vertex_bounds_mm': [vertices.min(0).tolist(), vertices.max(0).tolist()],
                      'pcd_bounds_mm': [xyz.min(0).tolist(), xyz.max(0).tolist()], 'alignment_trials': results,
                      'alignment_status': 'REFERENCE_ALIGNMENT_SUPPORTED' if results[0]['point_to_cad_p95_mm'] <= 1. else 'ALIGNMENT_OR_GEOMETRY_DIFFERENCE_UNRESOLVED',
                      'scope': 'CAD reference diagnosis only, no inference or attribute acceptance claim.',
                      'elapsed_seconds': time.perf_counter() - start}
            (out / (name + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
            print('CAD_ALIGN', json.dumps({k: report[k] for k in ['name', 'vertices', 'triangles', 'alignment_status', 'elapsed_seconds']}),
                  json.dumps(results[0]), flush=True)
    finally:
        gmsh.finalize()


if __name__ == '__main__':
    main()
