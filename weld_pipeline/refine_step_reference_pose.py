"""Reference-only gravity-constrained point-to-CAD pose refinement."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.truth import load_truth


def refinement(vertices, triangles, transform, xyz, normal):
    offset = vertices.mean(0)
    shifted = vertices - offset
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(shifted), o3d.utility.Vector3iVector(triangles))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rotation = transform[:3, :3].copy()
    gravity = np.array([0., 0., np.sign((rotation @ normal)[2])])
    # Project one CAD tangent onto the target horizontal plane, mapping the
    # exact CAD baseplate normal to the verified flat PCD gravity plane.
    tangent = np.cross(normal, [1., 0., 0.])
    if np.linalg.norm(tangent) < .1:
        tangent = np.cross(normal, [0., 1., 0.])
    tangent /= np.linalg.norm(tangent)
    cad_basis = np.column_stack([tangent, np.cross(normal, tangent), normal])
    world_tangent = rotation @ tangent
    world_tangent[2] = 0.
    world_tangent /= np.linalg.norm(world_tangent)
    world_basis = np.column_stack([world_tangent, np.cross(gravity, world_tangent), gravity])
    rotation = world_basis @ cad_basis.T
    rotated = shifted @ rotation.T
    translation = (xyz.min(0) + xyz.max(0) - rotated.min(0) - rotated.max(0)) / 2.
    center = (xyz.min(0) + xyz.max(0)) / 2.
    history = []
    for iteration in range(60):
        p_cad = (xyz - translation) @ rotation
        closest = scene.compute_closest_points(o3d.core.Tensor(p_cad.astype(np.float32)))
        q = closest['points'].numpy().astype(float) @ rotation.T + translation
        n = closest['primitive_normals'].numpy().astype(float) @ rotation.T
        distances = np.linalg.norm(q - xyz, axis=1)
        use = distances < 8.
        residual = np.sum((q[use] - xyz[use]) * n[use], axis=1)
        spin = np.cross([0., 0., 1.], q[use] - center)
        j = np.column_stack([np.sum(spin * n[use], axis=1), n[use]])
        weight = np.minimum(1., .5 / np.maximum(np.abs(residual), 1e-8))
        root = np.sqrt(weight)
        delta = np.linalg.lstsq(j * root[:, None], -residual * root, rcond=None)[0]
        angle = float(np.clip(delta[0], -.005, .005))
        c, s = np.cos(angle), np.sin(angle)
        update = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        rotation = update @ rotation
        translation = update @ (translation - center) + center + delta[1:]
        history.append({'iteration': iteration, 'p95_mm': float(np.percentile(distances, 95)),
                        'median_mm': float(np.median(distances)), 'delta': delta.tolist()})
        if abs(delta[0]) < 1e-10 and np.linalg.norm(delta[1:]) < 1e-5:
            break
    final = np.eye(4)
    final[:3, :3] = rotation
    final[:3, 3] = translation - rotation @ offset
    p_cad = (xyz - translation) @ rotation
    distances = scene.compute_distance(o3d.core.Tensor(p_cad.astype(np.float32))).numpy()
    return {'transform': final.tolist(), 'history': history, 'point_to_cad_median_mm': float(np.median(distances)),
            'point_to_cad_p95_mm': float(np.percentile(distances, 95)), 'point_to_cad_max_mm': float(distances.max()),
            'within_1mm_rate': float((distances <= 1.).mean()), 'within_3mm_rate': float((distances <= 3.).mean())}


def draw(name, verts, triangles, owners, truth, path):
    normals = np.cross(verts[triangles[:, 1]] - verts[triangles[:, 0]], verts[triangles[:, 2]] - verts[triangles[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    colors = np.array([plt.get_cmap('tab20')(int(o) % 20) for o in owners])
    fig = plt.figure(figsize=(15, 10))
    for panel, axes, title in [(1, [1, 0], 'Top view'), (2, [1, 2], 'Side view')]:
        ax = fig.add_subplot(3, 1, panel)
        other = ({0, 1, 2} - set(axes)).pop()
        mask = np.abs(normals[:, other]) > .7
        ax.add_collection(PolyCollection(verts[triangles[mask]][:, :, axes], facecolors=colors[mask],
                                         edgecolors='#56616d', linewidths=.12, alpha=.45))
        for w in truth:
            if w['type'] == 'vertical':
                for seg in w['segments']:
                    p = np.array(seg['points'])
                    ax.plot(p[:, axes[0]], p[:, axes[1]], color='red', lw=1.3)
        ax.autoscale(); ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(name + ': ' + title + '; CAD solids / red reference welds', fontsize=11)
        ax.set_xlabel('XYZ'[axes[0]] + ' / mm'); ax.set_ylabel('XYZ'[axes[1]] + ' / mm')
        ax.tick_params(labelsize=8)
    focus = next(w for w in truth if w['type'] == 'vertical' and
                 abs(np.linalg.norm(np.array(w['segments'][-1]['points'][-1]) - w['segments'][0]['points'][0]) - 120.) < .01)
    start, end = np.array(focus['segments'][0]['points'][0]), np.array(focus['segments'][-1]['points'][-1])
    center = (start + end) / 2.
    half = np.array([70., 70., 90.])
    tri = verts[triangles]
    mask = np.all(tri.max(1) >= center - half, axis=1) & np.all(tri.min(1) <= center + half, axis=1)
    ax = fig.add_subplot(3, 1, 3, projection='3d')
    ax.add_collection3d(Poly3DCollection(tri[mask], facecolor=colors[mask], edgecolor='#48505b',
                                       linewidth=.08, alpha=.55))
    for index, seg in enumerate(focus['segments']):
        p = np.array(seg['points'])
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color='red', lw=3.)
        ax.scatter(p[[0, -1], 0], p[[0, -1], 1], p[[0, -1], 2], color='black', s=20)
        ax.text(*p[-1], ' ' + str(index + 1), fontsize=9)
    ax.set_xlim(center[0] - half[0], center[0] + half[0]); ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_zlim(center[2] - half[2], center[2] + half[2]); ax.set_box_aspect(half)
    ax.view_init(23, -55); ax.tick_params(labelsize=7)
    ax.set_title('120 mm weld zoom: ' + str(len(focus['segments'])) + ' reference segment(s)', fontsize=11)
    ax.set_xlabel('X / mm'); ax.set_ylabel('Y / mm'); ax.set_zlabel('Z / mm')
    fig.subplots_adjust(top=.96, bottom=.05, left=.09, right=.94, hspace=.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--groups', nargs='+', default=['224-GR24A1', '527-FR146A'])
    args = ap.parse_args()
    out = Path('weld_pipeline/reports/step_alignment_20260930')
    for name in args.groups:
        archive = np.load(out / (name + '_mesh.npz'))
        source_report = json.loads((out / (name + '.json')).read_text('utf8'))
        cad = json.loads((Path('weld_pipeline/reports/step_geometry_20260930') / (name + '.json')).read_text('utf8'))
        biggest = max(cad['volumes'], key=lambda v: v['volume_mm3'])
        pair = biggest['parallel_major_face_pairs'][0]
        normal = np.array(pair['normal_axis'])
        folder = Path('DataSet/9月20日') / name
        pcd = next(folder.glob('*AllFaces.pcd'))
        xyz = read_xyz_pcd(pcd) * 1000.
        rng = np.random.default_rng(131)
        sample = xyz[rng.choice(len(xyz), min(70000, len(xyz)), replace=False)]
        trials = []
        for r in source_report['alignment_trials'][:2]:
            trials.append(refinement(archive['vertices'], archive['triangles'], np.array(r['transform']), sample, normal))
        trials.sort(key=lambda r: (r['point_to_cad_p95_mm'], r['point_to_cad_median_mm']))
        best = trials[0]
        t = np.array(best['transform'])
        aligned = archive['vertices'] @ t[:3, :3].T + t[:3, 3]
        np.savez_compressed(out / (name + '_mesh_refined.npz'), **{k: archive[k] for k in ['vertices', 'triangles', 'owners', 'face_ids']},
                            aligned_vertices=aligned, transform=t)
        truth, excluded = load_truth(folder)
        draw(name, aligned, archive['triangles'], archive['owners'], truth, out / (name + '_verified_views.png'))
        report = {'name': name, 'method': 'Point-to-trimmed-CAD-mesh; gravity locked to exact largest-plate normal; robust yaw + translation refinement.',
                  'source_mesh_sha256': sha256(out / (name + '_mesh.npz')), 'source_sha256': sha256(__file__),
                  'step_sha256': cad['sha256'], 'pcd_sha256': sha256(pcd), 'trials': trials,
                  'alignment_status': 'REFERENCE_ALIGNMENT_SUPPORTED' if best['point_to_cad_p95_mm'] <= 1. else 'UNRESOLVED',
                  'scope': 'Reference inspection, no inference or contract-attribute acceptance claim.'}
        (out / (name + '_refined.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        print('REFINED', name, json.dumps({k: v for k, v in best.items() if k != 'history'}), flush=True)


if __name__ == '__main__':
    main()
