"""Offline CAD-to-pure-cloud rigid alignment; no weld truth is consumed."""
from itertools import permutations, product
import gmsh
import numpy as np
from scipy.spatial import cKDTree


def rotation_vector_matrix(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-15:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def surface_samples(count=40000):
    gmsh.option.setNumber("Mesh.MeshSizeMin", 2.)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 50.)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 32)
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 32)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(2)
    ids, coords, _ = gmsh.model.mesh.getNodes()
    vertices = np.asarray(coords).reshape(-1, 3)
    order = np.argsort(ids)
    sorted_ids = np.asarray(ids)[order]
    triangles = []
    types, _, connectivity = gmsh.model.mesh.getElements(2)
    for element_type, nodes in zip(types, connectivity):
        _, _, _, total_nodes, _, primary = gmsh.model.mesh.getElementProperties(element_type)
        if primary != 3:
            raise ValueError("Registration requires triangular surface elements")
        tags = np.asarray(nodes).reshape(-1, total_nodes)[:, :3]
        triangles.append(order[np.searchsorted(sorted_ids, tags)])
    triangles = np.concatenate(triangles)
    tri = vertices[triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    good = twice_area > 1e-12
    tri, cross, twice_area = tri[good], cross[good], twice_area[good]
    rng = np.random.default_rng(302)
    selected = rng.choice(len(tri), count, p=twice_area / twice_area.sum())
    u, v = rng.random((2, count))
    outside = u + v > 1
    u[outside], v[outside] = 1 - u[outside], 1 - v[outside]
    chosen = tri[selected]
    sampled = chosen[:, 0] + u[:, None] * (chosen[:, 1] - chosen[:, 0]) + v[:, None] * (chosen[:, 2] - chosen[:, 0])
    normals = cross[selected] / twice_area[selected, None]
    return vertices, sampled, normals


def principal_basis(points):
    _, vectors = np.linalg.eigh(np.cov(points.T))
    basis = vectors[:, ::-1]
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    return basis


def refine(reference, normals, target, initial, iterations=60):
    reference_tree = cKDTree(reference)
    transform = initial.copy()
    rotation, translation = transform[:3, :3], transform[:3, 3]
    pivot = target.mean(axis=0)
    history = []
    for iteration in range(iterations):
        local = (target - translation) @ rotation
        distances, indices = reference_tree.query(local, workers=1)
        kept = distances <= max(20., 70. * (.90 ** iteration))
        if kept.sum() < 100:
            break
        world = reference[indices[kept]] @ rotation.T + translation
        normal_world = normals[indices[kept]] @ rotation.T
        residual = np.einsum("ij,ij->i", world - target[kept], normal_world)
        jacobian = np.column_stack([np.cross(world - pivot, normal_world), normal_world])
        weights = np.minimum(1., 2. / np.maximum(np.abs(residual), 1e-10))
        delta, _, rank, _ = np.linalg.lstsq(jacobian * np.sqrt(weights[:, None]),
                                         -residual * np.sqrt(weights), rcond=None)
        if rank < 6 or not np.isfinite(delta).all():
            break
        angle = float(np.linalg.norm(delta[:3]))
        shift = float(np.linalg.norm(delta[3:]))
        delta *= min(1., .1 / max(angle, 1e-30), 30. / max(shift, 1e-30))
        update = rotation_vector_matrix(delta[:3])
        rotation, translation = update @ rotation, update @ (translation - pivot) + pivot + delta[3:]
        history.append(dict(iteration=iteration, point_plane_p95_mm=float(np.percentile(np.abs(residual), 95)),
                            rotation_step_radians=float(np.linalg.norm(delta[:3])), translation_step_mm=float(np.linalg.norm(delta[3:]))))
        if np.linalg.norm(delta[:3]) < 1e-10 and np.linalg.norm(delta[3:]) < 1e-5:
            break
    transform[:3, :3], transform[:3, 3] = rotation, translation
    return transform, history


def pure_cloud_alignment(points, catalogue):
    vertices, reference, normals = surface_samples()
    rng = np.random.default_rng(302)
    target = np.asarray(points[rng.choice(len(points), min(20000, len(points)), replace=False)])
    target_tree = cKDTree(target)
    target_bases = [principal_basis(target)]
    rounded, counts = np.unique(target[:, 2].round(3), return_counts=True)
    flat = target[np.abs(target[:, 2] - rounded[np.argmax(counts)]) < .01]
    if len(flat) >= 500:
        _, axes = np.linalg.eigh(np.cov(flat[:, :2].T))
        first = np.r_[axes[:, -1], 0.]
        target_bases.append(np.column_stack([first, np.cross([0., 0., 1.], first), [0., 0., 1.]]))
    biggest = max(catalogue, key=lambda v: v["volume_mm3"])
    volume = int(biggest["volume"])
    inertia = np.asarray(gmsh.model.occ.getMatrixOfInertia(3, volume)).reshape(3, 3)
    _, volume_basis = np.linalg.eigh(inertia)
    if np.linalg.det(volume_basis) < 0:
        volume_basis[:, 2] *= -1
    reference_bases = [volume_basis, principal_basis(reference)]
    target_min, target_max = points.min(0), points.max(0)
    initials = []
    for target_basis in target_bases:
        for reference_basis in reference_bases:
            for perm in permutations(range(3)):
                permutation = np.eye(3)[:, perm]
                for signs in product((-1., 1.), repeat=3):
                    middle = permutation @ np.diag(signs)
                    if np.linalg.det(middle) < 0:
                        continue
                    rotation = target_basis @ middle @ reference_basis.T
                    rotated = vertices @ rotation.T
                    translation = (target_min + target_max - rotated.min(0) - rotated.max(0)) / 2
                    trial = reference[::20] @ rotation.T + translation
                    distances = target_tree.query(trial, workers=1)[0]
                    transform = np.eye(4)
                    transform[:3, :3], transform[:3, 3] = rotation, translation
                    initials.append((float(np.minimum(distances, 100).mean()), transform))
    initials.sort(key=lambda v: v[0])
    results = []
    reference_tree = cKDTree(reference)
    for initial_score, initial in initials[:6]:
        transform, history = refine(reference, normals, target, initial)
        local = (target - transform[:3, 3]) @ transform[:3, :3]
        distances, indices = reference_tree.query(local, workers=1)
        residual = np.abs(np.einsum("ij,ij->i", reference[indices] - local, normals[indices]))
        score = float(np.percentile(residual, 95)) + 5 * float(np.mean(distances > 20))
        results.append(dict(transform=transform.tolist(), initial_cloud_distance_mm=initial_score,
            final_point_plane_p95_mm=float(np.percentile(residual, 95)), correspondence_gt20_fraction=float(np.mean(distances > 20)),
            selection_score=score, history=history,
            fitted_from_pure_cloud_and_CAD_surfaces_only=True, weld_labels_used_for_registration=False))
    return sorted(results, key=lambda v: v["selection_score"])
