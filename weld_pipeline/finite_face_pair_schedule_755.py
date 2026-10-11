"""Exact row-set sharing and vectorized original broad phase; no face pruning."""
from collections import OrderedDict
from pathlib import Path
import hashlib
import numpy as np


def group_exact_faces(faces):
    groups=OrderedDict()
    comparison=('original_row_count','centroid_mm','normal','plane_offset_mm','covariance_eigenvalues_mm2',
        'fit_residual_max_mm','fit_residual_rms_mm','original_aabb_min_mm','original_aabb_max_mm')
    # Spacing controls witness bands, so equal row IDs alone do not suffice.
    for face in faces:
        if not face['eligible_finite_2D']:continue
        key=(face['original_rows_sha256'],face['coverage']['original_resolution_mm'])
        if key not in groups:groups[key]=dict(representative=face,aliases=[])
        group=groups[key]
        assert all(group['representative'][k]==face[k] for k in comparison),face['face_id']
        group['aliases'].append(dict(face_id=face['face_id'],plane_index=face['plane_index'],
            payload_path=face['payload_path'],row_array_key=face['row_array_key']))
    return list(groups.values())


def broad_pair_indices(groups):
    """Exactly 752's nonparallel/AABB gates, with O(F) temporary storage."""
    n=np.asarray([g['representative']['normal'] for g in groups])
    lo=np.asarray([g['representative']['original_aabb_min_mm'] for g in groups])
    hi=np.asarray([g['representative']['original_aabb_max_mm'] for g in groups])
    spacing=np.asarray([g['representative']['coverage']['original_resolution_mm'] for g in groups])
    for ai,g in enumerate(groups):
        cross=np.linalg.norm(np.cross(n[ai],n[ai+1:]),axis=1)
        distance=np.linalg.norm(np.maximum(np.maximum(lo[ai],lo[ai+1:])-np.minimum(hi[ai],hi[ai+1:]),0.),axis=1)
        reach=1.5*(spacing[ai]+spacing[ai+1:])+.3
        candidates=np.flatnonzero((cross>.15)&(distance<=reach))+ai+1
        first_planes={a['plane_index'] for a in g['aliases']}
        candidates=[int(bi) for bi in candidates if len(first_planes|{a['plane_index'] for a in groups[bi]['aliases']})>1]
        yield ai,candidates


class TwoPlaneRows:
    """Pin first payload and retain at most one other decompressed payload."""
    def __init__(self,folder):
        self.folder=Path(folder);self.primary_path=self.secondary_path=None
        self.primary={};self.secondary={};self.loads=0
    def load(self,path):
        self.loads+=1
        with np.load(self.folder/path,allow_pickle=False) as data:
            return {k:data[k] for k in data.files if k.endswith('_original_row_indices')}
    def pin(self,face):
        path=face['payload_path']
        if self.primary_path!=path:
            self.primary={};self.secondary={};self.secondary_path=None
            self.primary=self.load(path);self.primary_path=path
        return self.checked(face,self.primary)
    def get(self,face):
        path=face['payload_path']
        if path==self.primary_path:return self.checked(face,self.primary)
        if path!=self.secondary_path:
            self.secondary={};self.secondary=self.load(path);self.secondary_path=path
        return self.checked(face,self.secondary)
    @staticmethod
    def checked(face,arrays):
        rows=arrays[face['row_array_key']]
        assert len(rows)==face['original_row_count']
        assert hashlib.sha256(rows.tobytes()).hexdigest()==face['original_rows_sha256']
        return rows
