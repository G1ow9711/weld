"""Contact-local input geometry; no weld-target access in conversion."""
import numpy as np
import torch

VECTOR_BLOCKS = (0, 4, 7, 17, 20)


def local_arrays(arrays, geometry):
    nodes = arrays['nodes'].copy(); patches = arrays['patches'].copy()
    contacts, faces = geometry['contacts'], geometry['faces']
    bases, centers, tangents = [], [], []
    for i, c in enumerate(contacts):
        tangent = np.asarray(c['direction'], float)
        tangent /= np.linalg.norm(tangent)
        side = np.cross([0., 0., 1.], tangent)
        if np.linalg.norm(side) < 1e-4:
            side = np.asarray(faces[c['face_indices'][0]]['normal'], float)
            side = side-tangent*(side@tangent)
        side /= np.linalg.norm(side)
        third = np.cross(tangent, side)
        basis = np.stack([tangent, side, third], axis=1)
        center = (np.asarray(c['p0'])+c['p1'])*.5
        bases.append(basis); centers.append(center); tangents.append(tangent)
        # Local gravity retains horizontal/vertical semantics without heading.
        nodes[i, 0:3] = np.asarray([0., 0., 1.])@basis
        for k, fi in enumerate(c['face_indices']):
            face = faces[fi]
            nodes[i, 4+3*k:7+3*k] = np.asarray(face['normal'])@basis
            boundary = np.concatenate(face['lines'])
            local = (boundary-center)@basis
            nodes[i, 11+3*k:14+3*k] = np.ptp(local, axis=0)/1000.
            nodes[i, 17+3*k:20+3*k] = (np.asarray(face['center'])-center)@basis/1000.
        begin, end = arrays['ptr'][i:i+2]
        patches[begin:end] = arrays['patches'][begin:end]@basis
    bases=np.asarray(bases); centers=np.asarray(centers); tangents=np.asarray(tangents)
    src,dst=arrays['edge_index']
    relation = np.eye(2, dtype=np.float32)[arrays['edge_type']-1]
    relative_tangent = np.einsum('ei,eij->ej', tangents[src], bases[dst])
    relative_position = np.einsum('ei,eij->ej', (centers[src]-centers[dst])/1000., bases[dst])
    edges = np.c_[relation, relative_tangent, relative_position].astype(np.float32)
    return nodes, patches, edges


def load_source(si):
    from weld_pipeline.sjg_train_780 import load_source as original
    from weld_pipeline.sjg_geometry_776 import read
    from weld_pipeline.sjg_prepare_776 import OUT
    a = original(si)
    nodes, patches, edges = local_arrays(a, read(OUT/f'source_{si:02d}/geometry.json'))
    a['nodes'], a['patches'], a['edge_type'] = nodes, patches, edges
    a['nodes_gpu'] = torch.from_numpy(nodes).cuda()
    a['edge_type_gpu'] = torch.from_numpy(edges).cuda()
    return a


def normalization_nodes(data, indices):
    pieces=[]
    for i in indices:
        original=data[i]['nodes']; reversed_nodes=original.copy()
        for start in VECTOR_BLOCKS: reversed_nodes[:, start:start+2] *= -1
        pieces.extend([original, reversed_nodes])
    return np.concatenate(pieces)


def reversed_batch(a, ids, rng):
    from weld_pipeline.sjg_train_780 import tensors
    patch, query, index, target, offset, mask = tensors(a, ids)
    reverse = torch.from_numpy(rng.random(len(a['nodes'])) < .5).cuda()
    signs = torch.ones((len(reverse), 3), device='cuda')
    signs[reverse, :2] = -1
    nodes = a['nodes_gpu'].clone(); edge_features = a['edge_type_gpu'].clone()
    for start in VECTOR_BLOCKS: nodes[:, start:start+3] *= signs
    src, dst = a['edge_index_gpu']
    edge_features[:, 2:5] *= signs[dst]*signs[src, 0:1]
    edge_features[:, 5:8] *= signs[dst]
    patch *= signs[index, None, :]
    flip = reverse[index]
    query[flip, 0] = 1-query[flip, 0]
    query[flip, 1:3] = query[flip, 1:3].flip(1)
    target[flip, 1:3] = target[flip, 1:3].flip(1)
    offset[flip] = -offset[flip].flip(1)
    mask[flip] = mask[flip].flip(1)
    return nodes, edge_features, patch, query, index, target, offset, mask
