"""Small official PTv3 backbone; inputs and voxel mappings derive only from XYZ."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from training.third_party.ptv3_801.model import PointTransformerV3


CONFIG = dict(in_channels=3, order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
    stride=(2, 2, 2), enc_depths=(1, 1, 2, 2), enc_channels=(32, 64, 128, 256),
    enc_num_head=(2, 4, 8, 16), enc_patch_size=(128,)*4,
    dec_depths=(1, 1, 1), dec_channels=(32, 64, 128),
    dec_num_head=(2, 4, 8), dec_patch_size=(128,)*3,
    drop_path=0.05, shuffle_orders=False, enable_flash=False,
    upcast_attention=True, upcast_softmax=True)


def build_backbone():
    model=PointTransformerV3(**CONFIG)
    # Upstream constructor does not forward shuffle_orders to its pooling
    # modules. Explicitly disable their independent randomization as well.
    for module in model.modules():
        if hasattr(module, 'shuffle_orders'):
            module.shuffle_orders=False
    return model


def xyz_block(xyz_mm, point_count, seed=801):
    """Input-selected spatial crop. No weld labels, CAD or material IDs."""
    rng=np.random.default_rng(seed)
    anchor=int(rng.integers(len(xyz_mm)))
    ids=np.asarray(cKDTree(xyz_mm).query(xyz_mm[anchor], k=min(point_count,len(xyz_mm)))[1],np.int64).reshape(-1)
    return ids, voxelize(xyz_mm[ids])


def voxelize(xyz_mm, grid_mm=1.):
    p=np.asarray(xyz_mm,np.float64)
    grid=np.floor((p-p.min(0))/grid_mm).astype(np.int32)
    unique, representative, inverse=np.unique(grid,axis=0,return_index=True,return_inverse=True)
    selected=p[representative]
    coord=((selected-selected.mean(0))/100.).astype(np.float32)
    assert np.array_equal(unique[inverse],grid)
    return dict(coord=coord,grid_coord=unique,feat=coord.copy(),
        inverse=inverse.astype(np.int64),representative=representative.astype(np.int64),
        grid_mm=float(grid_mm),original_count=len(p))


def cuda_input(block):
    return {k:torch.from_numpy(np.ascontiguousarray(block[k])).cuda()
            for k in ('coord','grid_coord','feat')} | {
                'offset':torch.tensor([len(block['coord'])],device='cuda',dtype=torch.long)}
