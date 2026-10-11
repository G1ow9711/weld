"""TRAIN diagnostic: trace nearest raw arc support into frozen face membership."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

BASE = Path(__file__).resolve().parents[1] / 'server_rebuild/expansion_v1'

def main(key='new_ac2008a03607'):
    out = BASE / ('training_arc_support_membership_diagnostic_v1' if key=='new_ac2008a03607' else 'training_arc_support_membership_fc_diagnostic_v1')
    out.mkdir(exist_ok=False)
    folder = BASE / 'frozen_features/legacy_runtime' / ('source_' + key) / 'run185' / ('source_' + key)
    freeze = read(folder / 'input_freeze.json')
    path = BASE / 'inputs' / key / 'xyz_mm.npy'
    assert digest(path) == freeze['input_sha256']
    for name in ('surfaces.json', 'surface_support.npz'):
        assert digest(folder / name) == freeze['files'][name]
    xyz = np.load(path, mmap_mode='r')
    faces = read(folder / 'surfaces.json')
    tree = cKDTree(xyz)
    truthpath = BASE / 'labels' / key / 'targets.json'
    truth = read(truthpath)
    rows = []
    with np.load(folder / 'surface_support.npz') as support:
        for pi, parent in enumerate(truth['parents']):
            for ci, child in enumerate(parent['children']):
                if child['representation']['kind'] != 'arc':
                    continue
                arc = CircularArc.from_parameters(child['representation']['arc_parameters'])
                points = arc.point_at(np.linspace(0, arc.length_mm, int(np.ceil(arc.length_mm)) + 1))
                distance, ids = tree.query(points, workers=2)
                memberships = np.zeros(len(ids), dtype=np.int64)
                face_rows = []
                for fi, face in enumerate(faces):
                    skey = face.get('point_ids_key')
                    if skey not in support:
                        continue
                    mask = np.isin(ids, support[skey])
                    if not mask.any():
                        continue
                    memberships += mask
                    n = np.asarray(face.get('normal', [0, 0, 0]), float)
                    norm = np.linalg.norm(n)
                    if norm == 0:
                        continue
                    n /= norm
                    offset = float(face.get('offset', 0)) / norm
                    face_rows.append(dict(face=fi, matched_samples=int(mask.sum()),
                        abs_normal_dot_arc_normal=float(abs(n @ arc.plane_normal)),
                        raw_point_plane_residual_quantiles_mm=np.quantile(abs(xyz[ids[mask]] @ n - offset), [0, .5, 1]).tolist(),
                        reference_plane_residual_quantiles_mm=np.quantile(abs(points[mask] @ n - offset), [0, .5, 1]).tolist()))
                row = dict(parent=pi, child=ci, samples=len(ids),
                    raw_distance_max_mm=float(distance.max()), no_face_samples=int((memberships == 0).sum()),
                    membership_count_quantiles=np.quantile(memberships, [0, .5, 1]).tolist(), faces=face_rows)
                rows.append(row)
                print(pi, ci, 'no_face', row['no_face_samples'], 'faces', len(face_rows), flush=True)
    write(out / 'report.json', dict(key=key, rows=rows, target_sha256=digest(truthpath),
        source_freeze_sha256=digest(folder / 'input_freeze.json'),
        scope='TRAIN reference diagnostic only: nearest raw sample memberships, not all nearby support or contact proof; no candidate generation.',
        DEV_reads=0, heldout_reads=0, fit_calls=0))

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--key',choices=['new_ac2008a03607','new_fcaf2af1abee'],default='new_ac2008a03607');args=parser.parse_args();main(args.key)
